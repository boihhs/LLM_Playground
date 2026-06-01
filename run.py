from tokenizers import Tokenizer
from train import Config
import torch


def load_tokenizer(configs: Config):
    return Tokenizer.from_file(configs.tokenizer_path)

def load_model(configs: Config):
    return torch.load(configs.model_path, weights_only=False)

def generate_text(prompt, model, tokenizer, configs, length):
    prompt_ids = tokenizer.encode(prompt).ids
    # (B, L) = (1, L)
    ids = torch.tensor(prompt_ids, dtype=torch.int, device=configs.device).unsqueeze(0)
    out_ids = model.generate(ids, length)
    return tokenizer.decode(out_ids[0].tolist())

def main():
    configs = Config()
    tokenizer = load_tokenizer(configs)
    model = load_model(configs)
    length = 200
    prompt = input("Type your prompt below:\n")
    response = generate_text(prompt, model, tokenizer, configs, length)
    print(response)

if __name__ == "__main__":
    main()