"""
Fine-tune Llama-3-8B with LoRA on financial QA data.

Hardware profiles:
  --profile rtx3070  ->  RTX 3070 Super (12GB VRAM)
  --profile h100     ->  H100 (80GB VRAM)

Usage:
  uv run train.py --profile rtx3070
  uv run train.py --profile h100
  uv run train.py --profile rtx3070 --no-wandb
"""

import argparse

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

PROFILES = {
    "rtx3070": {
        "max_seq_length": 4096,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "dataset_num_proc": 4,
    },
    "h100": {
        "max_seq_length": 8192,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "dataset_num_proc": 8,
    },
}

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_prompts(examples, eos_token):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(ALPACA_PROMPT.format(instruction, inp, output) + eos_token)
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B LoRA")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="rtx3070",
        help="Hardware profile (default: rtx3070)",
    )
    parser.add_argument(
        "--dataset", default="ckerf/arb-instruct-v1", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--model", default="unsloth/llama-3-8b-bnb-4bit", help="Base model name"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--hub-model", default="ckerf/arbagent-llama3-8b-lora", help="HF Hub model ID for pushing"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing to HF Hub")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    max_seq_length = profile["max_seq_length"]

    # W&B setup
    if not args.no_wandb:
        import wandb

        wandb.init(
            project="Arb-Agent-v1",
            name=f"llama3-8b-lora-{args.profile}",
            job_type="training",
            config={"profile": args.profile, **profile},
        )

    # Load model
    print(f"Loading model with profile: {args.profile}")
    print(f"  max_seq_length={max_seq_length}, batch_size={profile['per_device_train_batch_size']}, "
          f"grad_accum={profile['gradient_accumulation_steps']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load and format dataset
    dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(
        lambda examples: format_prompts(examples, tokenizer.eos_token),
        batched=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=profile["dataset_num_proc"],
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=profile["per_device_train_batch_size"],
            gradient_accumulation_steps=profile["gradient_accumulation_steps"],
            num_train_epochs=args.epochs,
            warmup_steps=10,
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            report_to="wandb" if not args.no_wandb else "none",
            run_name=f"llama3-8b-lora-{args.profile}",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    trainer.train()

    # Push to Hub
    if not args.no_push:
        print(f"Pushing model to {args.hub_model}...")
        model.push_to_hub(args.hub_model)
        tokenizer.push_to_hub(args.hub_model)
        print("Done.")
    else:
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
