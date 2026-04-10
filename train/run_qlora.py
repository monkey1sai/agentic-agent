from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma 4 E4B QLoRA finetuning for coding domain.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_dependencies() -> dict[str, bool]:
    modules = ["torch", "transformers", "datasets", "peft", "accelerate", "trl", "bitsandbytes"]
    return {name: bool(importlib.util.find_spec(name)) for name in modules}


def build_chat_text(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
    return "\n".join(chunks)


def resolve_target_modules(base_targets: list[str], model_name: str) -> list[str]:
    if "gemma-4" not in model_name.lower():
        return base_targets
    return [f"{target}.linear" for target in base_targets]


def load_training_modules() -> dict[str, Any]:
    return {
        "torch": importlib.import_module("torch"),
        "load_dataset": importlib.import_module("datasets").load_dataset,
        "LoraConfig": importlib.import_module("peft").LoraConfig,
        "get_peft_model": importlib.import_module("peft").get_peft_model,
        "prepare_model_for_kbit_training": importlib.import_module("peft").prepare_model_for_kbit_training,
        "AutoModelForCausalLM": importlib.import_module("transformers").AutoModelForCausalLM,
        "AutoTokenizer": importlib.import_module("transformers").AutoTokenizer,
        "BitsAndBytesConfig": importlib.import_module("transformers").BitsAndBytesConfig,
        "Trainer": importlib.import_module("transformers").Trainer,
        "TrainingArguments": importlib.import_module("transformers").TrainingArguments,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    deps = check_dependencies()
    payload = {
        "config": config,
        "dependencies": deps,
        "python_ready": all(deps.values()),
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not all(deps.values()):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        missing = [name for name, ok in deps.items() if not ok]
        raise SystemExit(f"Missing dependencies for QLoRA: {', '.join(missing)}")

    modules = load_training_modules()
    torch = modules["torch"]
    load_dataset = modules["load_dataset"]
    LoraConfig = modules["LoraConfig"]
    get_peft_model = modules["get_peft_model"]
    prepare_model_for_kbit_training = modules["prepare_model_for_kbit_training"]
    AutoModelForCausalLM = modules["AutoModelForCausalLM"]
    AutoTokenizer = modules["AutoTokenizer"]
    BitsAndBytesConfig = modules["BitsAndBytesConfig"]
    Trainer = modules["Trainer"]
    TrainingArguments = modules["TrainingArguments"]

    train_path = Path(config["train_path"]).resolve()
    valid_path = Path(config["valid_path"]).resolve()
    target_modules = resolve_target_modules(config["target_modules"], config["base_model"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, config.get("torch_dtype", "bfloat16")),
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(valid_path)})

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [build_chat_text(messages) for messages in batch["messages"]]
        tokenized = tokenizer(texts, truncation=True, max_length=config["max_seq_length"], padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["save_steps"],
        max_steps=config.get("max_steps", -1),
        bf16=config.get("torch_dtype") == "bfloat16",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
