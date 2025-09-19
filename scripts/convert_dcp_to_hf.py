from transformers import AutoModelForCausalLM, AutoTokenizer
from torchtitan.components.checkpoint import ModelWrapper
import torch.distributed.checkpoint as dcp
import torch
from torchtitan.train import expand_tokenizer_with_unit_tokens

if __name__ == "__main__":
    num_quantizers = 4
    checkpoint_id = f"outputs/Llama-3.2-1B_peoples_speech-q4-s1024/checkpoint/step-5000"
    output_dir = f"models/Llama-3.2-1B_peoples_speech-q4-s1024"
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer = expand_tokenizer_with_unit_tokens(
        tokenizer,
        codebook_size=2048,
        num_quantizers=num_quantizers,
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    wrapped = ModelWrapper(model)
    print(wrapped)
    dcp.load(wrapped.state_dict(), checkpoint_id=checkpoint_id)
    model.config.num_quantizers = num_quantizers
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
