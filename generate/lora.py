# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
import json
from pathlib import Path
from typing import Literal, Optional
from tqdm import tqdm

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.lora import GPT, Config, merge_lora_weights
from lit_gpt.utils import CLI, check_valid_checkpoint_dir, get_default_supported_precision, lazy_load
from scripts.prepare_swisscom import generate_prompt


def main(
    swisscom_eval_dataset_path: Path = Path("eval_qa_dataset.json"),
    generated_contexts_path: Path = Path("generated_contexts.json"),
    lora_path: Path = Path("out/lora/swisscom-pkg-phi-2/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/microsoft/phi-2"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,

    max_new_tokens: int = 600,
    top_k: Optional[int] = 40,
    temperature: float = 0.7,

    precision: Optional[str] = None,

    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = False,
    lora_value: bool = True,
    lora_projection: bool = False,
    lora_mlp: bool = False,
    lora_head: bool = False,
) -> None:
    """Generates contexts for a given evaluation dataset.
    See `finetune/lora.py`.

    Args:
        swisscom_eval_dataset_path: Path to the JSON with Swisscom evaluation (question-answer) dataset.
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
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

    checkpoint_path = checkpoint_dir / "lit_model.pth"

    tokenizer = Tokenizer(checkpoint_dir)

    with open(swisscom_eval_dataset_path, "r") as f:
        eval_qa_dataset = json.load(f)

    samples = [{"input" : qa["question"]} for qa in eval_qa_dataset]
    prompts = [generate_prompt(sample) for sample in samples]
    encodeds = [tokenizer.encode(prompt, device=fabric.device) for prompt in prompts]
    max_returned_tokens = max([encoded.size(0) for encoded in encodeds]) + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
    model.eval()

    t0 = time.perf_counter()
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    merge_lora_weights(model)
    model = fabric.setup(model)

    L.seed_everything(1234)

    t0 = time.perf_counter()
    tokens_generated = 0

    contexts = []
    for encoded in tqdm(encodeds):
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
        
        output = tokenizer.decode(y)
        output = output.split("### Context:")[1].strip()
        contexts.append(output)

        tokens_generated += y.size(0) - prompt_length
        
        
    t = time.perf_counter() - t0
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

    with open(generated_contexts_path, "w") as f:
        json.dump(contexts, f, indent=1)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    CLI(main)
