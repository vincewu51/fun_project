import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from tags import NUM_TAGS

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask=None):
        attn_weights = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled

class Qwen2VLClassifier(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", use_lora=True,
                 lora_r=8, lora_alpha=16, use_gradient_checkpointing=True,
                 pooling_strategy='attention', loss_type='bce', class_weights=None):
        super().__init__()

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        if use_gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

        hidden_size = self.base_model.config.hidden_size
        self.pooling_strategy = pooling_strategy

        if pooling_strategy == 'attention':
            self.pooling = AttentionPooling(hidden_size)
        else:
            self.pooling = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_TAGS)
        )

        self.loss_type = loss_type
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            pos_weight = class_weights if class_weights is not None else None
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]

        if self.pooling_strategy == 'attention':
            pooled_output = self.pooling(hidden_states, attention_mask)
        else:
            pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}

    def predict(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.forward(pixel_values, input_ids, attention_mask)
            probs = torch.sigmoid(outputs['logits'])
        return probs

def get_processor(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
