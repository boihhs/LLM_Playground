from src.dataset.dataloader import DataLoader
from src.models.autregressive_model import Autoreg_Model
from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


@dataclass
class Config:
    train_path: str = "data/TinyStoriesV2-GPT4-train-tokens.bin"
    valid_path: str = "data/TinyStoriesV2-GPT4-valid-tokens.bin"
    tokenizer_path: str = "data/tokenizer-TinyStories.json"
    model_path: str = "model.pth"

    vocab_size: int = 16384
    pad_id: int = 2
    eot_id: int = 1

    B: int = 32
    L: int = 256

    E: int = 256
    nheads: int = 8
    nlayers: int = 6
    dropout: float = 0.1

    shared_encoder_decoder = False
    canon = False
    canon_length: int = 3
    is_causal = True
    bias = True

    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    num_steps: int = 10000
    eval_interval: int = 500

    token_dtype = np.uint16
    dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    

def config_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

@torch.no_grad()
def eval(model: Autoreg_Model, eval_batcher: DataLoader, configs: Config):
    model.eval()
    x, y = eval_batcher.get_batch(configs.B, configs.L)
    _, loss = model(x, y)

    return loss.item()


def train(model: Autoreg_Model, train_batcher: DataLoader, eval_batcher: DataLoader, optimizer, configs: Config, writer: SummaryWriter):
    model.train()

    for i in range(configs.num_steps):
        if i % configs.eval_interval == 0:
            eval_loss = eval(model, eval_batcher, configs)
            model.train()
            writer.add_scalar("Loss/eval", eval_loss, i)
            writer.flush()
            torch.save(model, configs.model_path)

        x, y = train_batcher.get_batch(configs.B, configs.L)

        _, loss = model(x, y)
        train_loss = loss.item()

        writer.add_scalar("Loss/train", train_loss, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    config = Config()
    writer = SummaryWriter()

    train_batcher = DataLoader(config.train_path, config.token_dtype, config.device)
    eval_batcher = DataLoader(config.train_path, config.token_dtype, config.device)

    model = Autoreg_Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    optimizer = config_optimizer(model, config)
    train(model, train_batcher, eval_batcher, optimizer, config, writer)


if __name__ == "__main__":
    main()