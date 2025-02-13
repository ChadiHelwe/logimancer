import csv
import json
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Block, Config, merge_lora_weights
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    lazy_load,
    quantization,
)
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False


def eval(
    lora_path: Path = Path("out/lora/logimancer/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-hf"),
    quantize: Optional[
        Literal[
            "bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"
        ]
    ] = None,
    max_new_tokens: int = 1024,
    top_k: int = 200,
    temperature: float = 0.8,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(
        checkpoint_dir / "lit_config.json",
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    fabric.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    with lazy_load(checkpoint_path) as checkpoint, lazy_load(
        lora_path
    ) as lora_checkpoint:
        checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
        model.load_state_dict(checkpoint, strict=quantize is None)
    fabric.print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )

    model.eval()
    merge_lora_weights(model)
    model = fabric.setup(model)
    tokenizer = Tokenizer(checkpoint_dir)

    with open("results/results_amr.csv", "w") as f:
        out = csv.writer(f)
        out.writerow(["instruction", "input", "output", "prediction"])
        with open("datasets/test_amr_logimancer_dataset.json", "r") as f:
            data = json.load(f)
            for i in tqdm(data):
                instruction = i["instruction"]
                input = i["input"]
                output = i["output"]
                instruction_input = {"instruction": instruction, "input": input}

                try:
                    prompt = generate_prompt(instruction_input)
                    encoded = tokenizer.encode(prompt, device=fabric.device)
                    prompt_length = encoded.size(0)
                    max_returned_tokens = prompt_length + max_new_tokens

                    with fabric.init_tensor():
                        # set the max_seq_length to limit the memory usage to what we need
                        model.max_seq_length = max_returned_tokens
                        # enable the kv cache
                        model.set_kv_cache(batch_size=1)

                    # t0 = time.perf_counter()
                    y = generate(
                        model,
                        encoded,
                        max_returned_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        eos_id=tokenizer.eos_id,
                    )
                    # t = time.perf_counter() - t0

                    prediction = tokenizer.decode(y)
                    prediction = prediction.split("### Response:")[1].strip()
                    out.writerow([instruction, input, output, prediction])
                except Exception as e:
                    print("Error:", e)
                # tokens_generated = y.size(0) - prompt_length
                # fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
                # if fabric.device.type == "cuda":
                #     fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    eval()
