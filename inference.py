from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MimiModel,
    AutoFeatureExtractor,
    StoppingCriteria,
)
import torch
import torchaudio
import re
import requests
import io


def audio_array_to_text(
    audio_array: torch.tensor,
    audio_tokenizer,
    feature_extractor,
    num_quantizers: int,
) -> str:
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(audio_tokenizer.device)
    with torch.no_grad():
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

    return f"<audio>{text}</audio>"


def text_to_audio_values(
    text: str,
    num_quantizers: int,
    output_file: str,
    audio_tokenizer,
    feature_extractor,
):
    # Extract (val, idx) pairs from the <val_idx> format in the text
    matches = re.findall(r"<(\d+)_(\d+)>", text)
    vals = []
    for i in range(0, len(matches), num_quantizers):
        chunk = matches[i : i + num_quantizers]
        if len(chunk) < num_quantizers:
            break
        indices = [int(idx) for _, idx in chunk]
        if indices == list(range(num_quantizers)):
            vals.extend(int(val) for val, _ in chunk)
        else:
            break
    vals = vals[: len(vals) - len(vals) % num_quantizers]
    tensor_bt4 = torch.tensor(vals).reshape(1, -1, num_quantizers)  # (B, T, 4)
    tensor_b4t = tensor_bt4.transpose(1, 2)  # (B, 4, T)
    audio_values = audio_tokenizer.decode(tensor_b4t)[0]
    torchaudio.save(
        output_file,
        audio_values[0].detach().cpu(),
        feature_extractor.sampling_rate,
    )


class StopOnAudioEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.target_text = "</audio>"
        self.target_ids = tokenizer(
            self.target_text, add_special_tokens=False
        ).input_ids

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < len(self.target_ids):
            return False
        return input_ids[0][-len(self.target_ids) :].tolist() == self.target_ids


temperature = 0.8
top_k = 30
do_sample = True
max_length = 1024
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llm-jp/Llama-Mimi-1.3B"
model = (
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    .eval()
    .to(device)
)
num_quantizers = model.config.num_quantizers
tokenizer = AutoTokenizer.from_pretrained(model_id)
audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
stopping_criteria = StopOnAudioEnd(tokenizer)

audio_url = (
    "https://speed1313.github.io/llama-mimi/data/prompt/natural/great_day_gt.wav"
)
response = requests.get(audio_url)
response.raise_for_status()
waveform, sample_rate = torchaudio.load(io.BytesIO(response.content))
if sample_rate != feature_extractor.sampling_rate:
    waveform = torchaudio.transforms.Resample(
        sample_rate, feature_extractor.sampling_rate
    )(waveform)
    sample_rate = feature_extractor.sampling_rate
prompt_array = waveform.squeeze().cpu().numpy()

text = audio_array_to_text(
    prompt_array, audio_tokenizer, feature_extractor, num_quantizers
)

text = text.replace("</audio>", "")
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        stopping_criteria=[stopping_criteria],
    )

generated_text = tokenizer.decode(generated[0])

text_to_audio_values(
    generated_text,
    num_quantizers=num_quantizers,
    output_file="output.wav",
    audio_tokenizer=audio_tokenizer,
    feature_extractor=feature_extractor,
)
