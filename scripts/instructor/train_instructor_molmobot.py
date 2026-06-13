#!/usr/bin/env python3
"""Fine-tune Qwen3.5-2B VLM for MolmoBot caption generation via LoRA SFT.

Usage:
    # Single GPU
    python scripts/instructor/train_instructor.py --data-dir data/instructor_training_mix5

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=8 scripts/instructor/train_instructor.py --data-dir data/instructor_training_mix5
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
)
from transformers import Trainer
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_PROJECT = "/path/to/workspace/molmo/MolmoBot/MolmoBot-Pi0"
DEFAULT_DATA_DIR = f"{_PROJECT}/data/instructor_training_mix5"
DEFAULT_MODEL_NAME = "/path/to/models/Qwen3.5-2B"
DEFAULT_OUTPUT_DIR = f"{_PROJECT}/checkpoints/instructor/sft"

# Image field names in the training JSONL.  MolmoBot uses 2 cameras (exo + wrist);
# Robocasa used 3 (image_left + image_right + image_wrist).  Override with
# --image-keys to use a different set.
DEFAULT_IMAGE_KEYS = ("image_exo", "image_wrist")

SYSTEM_PROMPT = (
    "You are a robot action instructor. Given camera images and a task description "
    "for a robotic manipulation task, generate a concise caption describing the "
    "robot's action and scene. Respond with only the caption text, nothing else."
)


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def build_messages(example: dict, data_dir: str, image_keys: tuple[str, ...]) -> list[dict]:
    """Build chat messages with image content for a training example.

    Each key in ``image_keys`` must name a field in ``example`` whose value
    is a path (relative to ``data_dir``) to a JPEG/PNG camera frame.
    """
    content: list[dict] = []
    for k in image_keys:
        img = Image.open(os.path.join(data_dir, example[k])).convert("RGB")
        content.append({"type": "image", "image": img})
    content.append({
        "type": "text",
        "text": (
            f"Task: {example['task_description']}\n"
            "Generate a concise caption describing the robot's action and "
            "scene for this task."
        ),
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "assistant", "content": example["caption"]},
    ]


class InstructorDataCollator:
    """Data collator that builds VLM inputs from chat messages with images.

    Each example is processed through the Qwen3VL processor which handles
    image tokenization and chat template formatting.  The labels are masked
    for all tokens except the assistant response.
    """

    def __init__(self, processor, data_dir: str, image_keys: tuple[str, ...],
                 max_seq_length: int = 512):
        self.processor = processor
        self.data_dir = data_dir
        self.image_keys = image_keys
        self.max_seq_length = max_seq_length

    def __call__(self, features: list[dict]) -> dict:
        texts = []
        all_images = []

        for feat in features:
            messages = build_messages(feat, self.data_dir, self.image_keys)

            # Extract images from messages for the processor.
            images = []
            for msg in messages:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            images.append(item["image"])

            # Apply chat template (text only, images handled separately).
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_images.append(images)

        # Process all examples in a batch.
        batch = self.processor(
            text=texts,
            images=all_images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Build labels: mask everything except assistant response tokens.
        labels = batch["input_ids"].clone()
        # Mask padding tokens.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask all tokens before the assistant response.
        # The assistant response starts after "<|im_start|>assistant\n"
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            # Find the last occurrence of <|im_start|> which marks the assistant turn.
            last_im_start = -1
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] == im_start_id:
                    last_im_start = j
                    break
            if last_im_start >= 0:
                # Mask everything up to and including "assistant\n" (typically +2 tokens after im_start).
                # Find the first non-special token after the assistant header.
                mask_end = last_im_start + 3  # im_start + "assistant" + "\n"
                labels[i, :mask_end] = -100

        batch["labels"] = labels

        # Reward weights for reward-weighted SFT (if present in data).
        if "reward_weight" in features[0]:
            batch["reward_weights"] = torch.tensor(
                [f["reward_weight"] for f in features], dtype=torch.float32,
            )

        return batch


class RewardWeightedTrainer(Trainer):
    """Trainer that scales per-example loss by a reward weight.

    When the batch contains a ``reward_weights`` tensor (shape [B]), the
    cross-entropy loss for each example is multiplied by its weight before
    averaging.  This implements reward-weighted SFT: examples with higher
    downstream task success rates contribute more to the gradient.

    Falls back to standard Trainer behavior when no weights are present.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        reward_weights = inputs.pop("reward_weights", None)

        outputs = model(**inputs)

        if reward_weights is None:
            loss = outputs.loss
        else:
            # Compute per-token loss, then weight per-example.
            logits = outputs.logits
            labels = inputs["labels"]
            # Shift for causal LM: predict next token.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Per-token CE (no reduction).
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            # Mask out padding (-100 labels).
            mask = shift_labels != -100
            # Per-example mean loss.
            loss_per_example = (loss_per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # Apply reward weights.
            weighted_loss = (loss_per_example * reward_weights.to(loss_per_example.device)).mean()
            loss = weighted_loss

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B VLM instructor with LoRA SFT")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="HuggingFace model ID or local path")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with train.jsonl, val.jsonl, images/")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--per-device-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--image-keys",
        default=",".join(DEFAULT_IMAGE_KEYS),
        help="Comma-separated image field names (order matters). "
             f"Default: {','.join(DEFAULT_IMAGE_KEYS)}",
    )
    parser.add_argument(
        "--ignore-reward-weights",
        action="store_true",
        help="Ignore reward_weight field in data (force uniform-loss SFT "
             "even when data was prepared with reward weights).",
    )
    args = parser.parse_args()

    image_keys = tuple(k.strip() for k in args.image_keys.split(",") if k.strip())
    logger.info("Image keys: %s", image_keys)

    # -----------------------------------------------------------------------
    # Load model and processor
    # -----------------------------------------------------------------------
    logger.info("Loading model: %s", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
    )

    # Freeze vision encoder — only train language model via LoRA.
    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
        logger.info("Froze vision encoder (%d params)", sum(p.numel() for p in model.visual.parameters()))

    # -----------------------------------------------------------------------
    # Apply LoRA to language model
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    train_examples = load_jsonl(data_dir / "train.jsonl")
    val_examples = load_jsonl(data_dir / "val.jsonl")
    logger.info("Train: %d examples, Val: %d examples", len(train_examples), len(val_examples))

    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)

    data_collator = InstructorDataCollator(
        processor, str(data_dir), image_keys, max_seq_length=args.max_seq_length,
    )

    # -----------------------------------------------------------------------
    # Training config
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size // 4),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        seed=args.seed,
        report_to="wandb",
        run_name="qwen_instructor",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    # Use RewardWeightedTrainer if data has reward weights AND they aren't
    # explicitly ignored; else standard Trainer.
    has_rewards = (
        "reward_weight" in train_examples[0]
        and not args.ignore_reward_weights
    )
    if args.ignore_reward_weights:
        logger.info("--ignore-reward-weights: forcing uniform-loss SFT.")
        # Also strip from the collator input so the 'reward_weights' tensor
        # isn't built and accidentally used.
        for ex in train_examples:
            ex.pop("reward_weight", None)
        for ex in val_examples:
            ex.pop("reward_weight", None)
        # Rebuild HF datasets to reflect the strip
        train_ds = Dataset.from_list(train_examples)
        val_ds = Dataset.from_list(val_examples)
    trainer_cls = RewardWeightedTrainer if has_rewards else Trainer
    logger.info("Using %s (reward_weighted=%s)", trainer_cls.__name__, has_rewards)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final LoRA adapters + tokenizer.
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    logger.info("Saved final model to %s", final_dir)

    # Merge LoRA into base model for easy inference.
    logger.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_dir = output_dir / "merged"
    merged_model.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    logger.info("Saved merged model to %s", merged_dir)


if __name__ == "__main__":
    main()
