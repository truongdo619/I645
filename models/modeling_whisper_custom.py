from transformers import WhisperForConditionalGeneration
import torch.nn as nn
from torch.nn.modules import CrossEntropyLoss

class MultiTaskWhisperModel(nn.Module):
    def __init__(self, model_args, config) -> None:
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                subfolder=model_args.subfolder,
                token=model_args.token,
                low_cpu_mem_usage=True,
                attn_implementation=
                model_args.attn_implementation,
            )
        self.ce_loss = CrossEntropyLoss()


    def forward(self, batch):
        outputs = self.whisper(input_features=batch["input_features"], labels=batch["labels"])
        batch_size, seq_len, vocab_size = outputs.logits.size()
        loss = self.ce_loss(outputs.logits.view(-1, vocab_size), batch["labels"].reshape(-1))
        return loss