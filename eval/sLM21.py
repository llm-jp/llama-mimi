from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor
import os
from torchtitan.datasets.hf_datasets import audio_array_to_text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import json


def parse_args():
    parser = ArgumentParser(description="Audio completion generation script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-jp/Llama-Mimi-1.3B",
        help="Run name for the model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    return parser.parse_args()


def compute_loss(
    audio, audio_tokenizer, feature_extractor, num_quantizers, tokenizer, model, device
):
    text = audio_array_to_text(
        audio, audio_tokenizer, feature_extractor, num_quantizers
    )
    inputs = tokenizer(text, return_tensors="pt")

    labels = inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    mod_mask = torch.zeros_like(labels)
    mod_mask[:, 2 :: model.config.num_quantizers] = 1
    labels[mod_mask == 0] = -100
    # print([tokenizer.decode(ids) for ids in neg_labels[0].tolist() if ids != -100])
    inputs = inputs.to(device)
    outputs = model(input_ids=inputs.input_ids, labels=labels)
    loss = outputs.loss
    return loss


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    model = (
        AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
        .eval()
        .to(device)
    )
    num_quantizers = model.config.num_quantizers
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = ["sWUGGY", "sBLIMP"]
    result = {}
    for task in tasks:
        ds = load_dataset(
            f"speed/{task}", split="train"
        )  # .shuffle(seed=42).select(range(1000))
        ds = ds.cast_column(
            "negative", Audio(sampling_rate=feature_extractor.sampling_rate)
        )
        ds = ds.cast_column(
            "positive", Audio(sampling_rate=feature_extractor.sampling_rate)
        )

        total_correct = 0
        total_samples = 0

        for example in tqdm(ds):
            negative_audio = example["negative"]["array"]
            positive_audio = example["positive"]["array"]

            neg_loss = compute_loss(
                negative_audio,
                audio_tokenizer,
                feature_extractor,
                num_quantizers,
                tokenizer,
                model,
                device,
            )
            pos_loss = compute_loss(
                positive_audio,
                audio_tokenizer,
                feature_extractor,
                num_quantizers,
                tokenizer,
                model,
                device,
            )
            # print(f"Neg loss: {neg_loss.item()}, Pos loss: {pos_loss.item()}")

            total_correct += (neg_loss > pos_loss).item()
            total_samples += 1

        acc = total_correct / total_samples
        result[task] = acc
        print(f"Accuracy for {task}: {acc}")

    output_path = os.path.join(
        args.output_dir, "sLM21", args.model_name, "accuracy.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
