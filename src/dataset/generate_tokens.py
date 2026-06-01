from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
import os

"""
From: https://huggingface.co/docs/tokenizers/quicktour
"""

# First "train" the byte pair encoder on the train and validation
def train_tokenizer(files, tokenizer_out, vocab_size, eot_token):
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", eot_token, "<pad>"],
        min_frequency=2,
    )
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files, trainer)
    tokenizer.save(tokenizer_out)

# Tokenize the train and validation data and store in bin files
def tokenize_dataset(files, files_w, tokenizer_out, eot_token, dtype=np.uint16, chunk_size=1024*1024):
    end_bytes=eot_token.encode("utf-8")
    tokenizer=Tokenizer.from_file(tokenizer_out)

    for file, file_w in zip(files, files_w):
        print(f"Starting file: {file}...")

        length_bytes=os.path.getsize(file)
        comp_bytes=0
        tracker=.1
        pending=b""

        with open(file_w, "wb") as f_w:
            with open(file, "rb") as f:
                while True:
                    chunk=f.read(chunk_size)

                    if chunk == b"":
                        if pending:
                            text=pending.decode("utf-8")
                            output=np.array(tokenizer.encode(text).ids, dtype=dtype)
                            f_w.write(output.tobytes())
                        break

                    comp_bytes += len(chunk)
                    pending += chunk

                    parts=pending.split(end_bytes)

                    for part in parts[:-1]:
                        text=(part + end_bytes).decode("utf-8")
                        output=np.array(tokenizer.encode(text).ids, dtype=dtype)
                        f_w.write(output.tobytes())

                    pending=parts[-1]

                    while length_bytes > 0 and comp_bytes/length_bytes >= tracker:
                        print(f"{tracker*100:.0f}% Completed ({comp_bytes}/{length_bytes})")
                        tracker += .1

files = [f"data/TinyStoriesV2-GPT4-{split}.txt" for split in ["train", "valid"]]
files_w = [f"data/TinyStoriesV2-GPT4-{split}-tokens.bin" for split in ["train", "valid"]]
eot = "<|endoftext|>"
tokenizer_out = "data/tokenizer-TinyStories.json"
vocab_size=16384

assert vocab_size <= np.iinfo(np.uint16).max + 1

# First "train" the byte pair encoder on the train and validation
train_tokenizer(files, tokenizer_out, vocab_size, eot)

# Tokenize the train and validation data and store in bin files
tokenize_dataset(files, files_w, tokenizer_out, eot)

