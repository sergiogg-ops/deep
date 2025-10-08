import os
import argparse
import yaml
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    get_linear_schedule_with_warmup,
)
import sacrebleu
from peft import LoraConfig, get_peft_model, TaskType


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        tokenizer: AutoTokenizer,
        src_max_length: int = 128,
        tgt_max_length: int = 128,
    ):
        self.examples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length

        with open(src_file, "r", encoding="utf-8") as src_f, open(tgt_file, "r", encoding="utf-8") as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                src = src_line.strip()
                tgt = tgt_line.strip()
                if src and tgt:
                    self.examples.append({"src": src, "tgt": tgt})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        src_text = example["src"] + self.tokenizer.eos_token
        tgt_text = example["tgt"] + self.tokenizer.eos_token
        input_text = src_text + tgt_text

        enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.src_max_length + self.tgt_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        enc_src = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.src_max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        src_input_ids = enc_src.input_ids.squeeze(0)
        src_attention_mask = enc_src.attention_mask.squeeze(0)

        src_len = (src_input_ids != self.tokenizer.pad_token_id).sum().item()

        labels = input_ids.clone()
        labels[:src_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "src_input_ids": src_input_ids,
            "src_attention_mask": src_attention_mask,
            "src_len": src_len,
            "tgt_text": example["tgt"],
        }


class TranslationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_src: str,
        train_tgt: str,
        val_src: str,
        val_tgt: str,
        test_src: str,
        test_tgt: str,
        model_name_or_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        src_max_length: int = 128,
        tgt_max_length: int = 128,
    ):
        super().__init__()
        self.train_src = train_src
        self.train_tgt = train_tgt
        self.val_src = val_src
        self.val_tgt = val_tgt
        self.test_src = test_src
        self.test_tgt = test_tgt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.src_max_length = src_max_length
        self.tgt_max_length = tgt_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,token='hf_token')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TranslationDataset(
            self.train_src, self.train_tgt, self.tokenizer, self.src_max_length, self.tgt_max_length
        )
        self.val_dataset = TranslationDataset(
            self.val_src, self.val_tgt, self.tokenizer, self.src_max_length, self.tgt_max_length
        )
        self.test_dataset = TranslationDataset(
            self.test_src, self.test_tgt, self.tokenizer, self.src_max_length, self.tgt_max_length
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class TranslationFineTuner(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        total_steps: int = 1000,
        adam_epsilon: float = 1e-8,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        src_max_length: int = 128,
        tgt_max_length: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            #device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=False,
            token='hf_token'
        )
        if self.hparams.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,token='hf_token')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.val_preds: List[str] = []
        self.val_refs: List[str] = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("val_loss", outputs.loss, prog_bar=True)
        gen_ids = self.model.generate(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            max_length=self.hparams.src_max_length + self.hparams.tgt_max_length,
        )
        for i in range(gen_ids.size(0)):
            src_len = batch["src_len"][i]
            pred_ids = gen_ids[i, src_len:]
            pred_str = self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            self.val_preds.append(pred_str)
            self.val_refs.append(batch["tgt_text"][i])

    def on_validation_epoch_end(self):
        bleu = sacrebleu.corpus_bleu(self.val_preds, [self.val_refs])
        self.log("val_bleu", bleu.score, prog_bar=True)
        self.val_preds.clear()
        self.val_refs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("test_loss", outputs.loss)
    
    def predict_step(self, batch, batch_idx):
        gen_ids = self.model.generate(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            max_length=self.hparams.src_max_length + self.hparams.tgt_max_length,
        )
        preds = []
        for i in range(gen_ids.size(0)):
            src_len = batch["src_len"][i]
            pred_ids = gen_ids[i, src_len:]
            pred_str = self.tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            preds.append(pred_str)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            eps=float(self.hparams.adam_epsilon),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )
        return [optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA for translation with YAML config, LoRA, and early stopping support")
    parser.add_argument("--config", type=str, help="Path to YAML config file with parameters")
    # arguments
    parser.add_argument("--train_src", type=str)
    parser.add_argument("--train_tgt", type=str)
    parser.add_argument("--val_src", type=str)
    parser.add_argument("--val_tgt", type=str)
    parser.add_argument("--test_src", type=str)
    parser.add_argument("--test_tgt", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--src_max_length", type=int, default=128)
    parser.add_argument("--tgt_max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning", default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping", default=True)
    parser.add_argument("--patience", type=int, help="Early stopping patience epochs", default=3)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items(): setattr(args, key, val)

    required = ["train_src","train_tgt","val_src","val_tgt","test_src","test_tgt","model_name_or_path","output_dir"]
    for r in required:
        if not getattr(args, r, None): parser.error(f"--{r} is required")
    

    train_lines = sum(1 for _ in open(args.train_src, encoding="utf-8"))
    total_steps = (train_lines // args.batch_size) * args.max_epochs

    data_module = TranslationDataModule(
        train_src=args.train_src,
        train_tgt=args.train_tgt,
        val_src=args.val_src,
        val_tgt=args.val_tgt,
        test_src=args.test_src,
        test_tgt=args.test_tgt,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        src_max_length=args.src_max_length,
        tgt_max_length=args.tgt_max_length,
    )

    model = TranslationFineTuner(
        model_name_or_path=args.model_name_or_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r or 8,
        lora_alpha=args.lora_alpha or 16,
        lora_dropout=args.lora_dropout or 0.1
    )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="llama-translation-{epoch:02d}-{val_loss:.2f}-{val_bleu:.2f}",
        save_top_k=3,
        monitor="val_bleu",
        mode="max",
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_bleu", mode="max", patience=args.patience))

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        val_check_interval=0.25,
    )

    trainer.fit(model, data_module)
    trainer.test(model, datamodule=data_module)

    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
