# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from typing import Any

import torch

from datasets import Dataset, load_dataset, Audio
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from transformers import DataCollatorForSeq2Seq


def audio_array_to_text(
    audio_array: torch.tensor,
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    max_seconds: int = 20,
) -> str:
    # truncate the audio array to the expected length
    if audio_array.shape[-1] > max_seconds * feature_extractor.sampling_rate:
        audio_array = audio_array[: max_seconds * feature_extractor.sampling_rate]
        #
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(audio_tokenizer.device)
    with torch.no_grad():
        # Encode the audio input to get the audio codes
        # This will return a tensor of shape (batch_size, num_quantizers, sequence_length)
        # where each quantizer's output is in a separate dimension
        encoder_outputs = audio_tokenizer.encode(
            inputs["input_values"],
            inputs["padding_mask"],
            num_quantizers=num_quantizers,
        )
    flatten_audio_codes = encoder_outputs.audio_codes.transpose(1, 2).reshape(-1)
    assert flatten_audio_codes.numel() % num_quantizers == 0
    steps = []
    for i in range(0, flatten_audio_codes.numel(), num_quantizers):
        group = [
            f"<{flatten_audio_codes[i + j].item()}_{j}>" for j in range(num_quantizers)
        ]
        steps.append(group)

    parts = [tok for step in steps for tok in step]

    text = "".join(parts)

    del inputs, encoder_outputs, flatten_audio_codes
    torch.cuda.empty_cache()
    return f"<audio>{text}</audio>"


def process_audio(
    sample: dict[str, Any],
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
    task: str = "a2a",
) -> str:
    audio_sample = sample["audio"]["array"]
    text = audio_array_to_text(
        audio_sample,
        audio_tokenizer,
        feature_extractor,
        num_quantizers,
    )
    if task == "tts":
        transcription = sample["text"]
        text = transcription + text
    return text


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: BaseTokenizer,
        audio_tokenizer=None,
        feature_extractor=None,
        num_quantizers: int = 4,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        task: str = "a2a",
    ) -> None:
        if dataset_name == "peoples_speech":
            ds = load_dataset(
                "parquet",
                data_dir="data/peoples_speech/clean",
                split="train",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "librispeech_asr_train":
            ds = load_dataset(
                "openslr/librispeech_asr",
                split="train.other.500",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )
        elif dataset_name == "librispeech_asr_test":
            ds = load_dataset(
                "openslr/librispeech_asr",
                split="test.clean",
                streaming=True,
            )
            ds = ds.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate)
            )

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported. ")

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._data = self._data.shuffle(seed=42, buffer_size=10_000)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.feature_extractor = feature_extractor
        self.num_quantizers = num_quantizers
        self.seq_len = seq_len
        self.task = task
        self.infinite = infinite

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        while True:
            data_iter = self._get_data_iter()
            while True:
                try:
                    sample = next(data_iter)
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(
                        f"Error while iterating over dataset {self.dataset_name}: {e}"
                    )
                    self._sample_idx += 1
                    continue

                try:
                    sample_text = process_audio(
                        sample,
                        self.audio_tokenizer,
                        self.feature_extractor,
                        self.num_quantizers,
                        self.task,
                    )
                    self._sample_idx += 1
                    yield self.tokenizer(
                        sample_text,
                        max_length=self.seq_len,
                        padding="max_length",
                        truncation=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Error while processing sample in dataset {self.dataset_name}: {e}"
                    )
                    self._sample_idx += 1
                    continue

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    audio_tokenizer,
    feature_extractor,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        feature_extractor=feature_extractor,
        num_quantizers=job_config.model.num_quantizers,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        task=job_config.training.task,
    )

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        max_length=seq_len,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )


def build_hf_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    audio_tokenizer,
    feature_extractor,
    job_config: JobConfig,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets."""
    dataset_name = job_config.validation.dataset
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        feature_extractor=feature_extractor,
        num_quantizers=job_config.model.num_quantizers,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
        task=job_config.training.task,
    )

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        max_length=seq_len,
    )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
