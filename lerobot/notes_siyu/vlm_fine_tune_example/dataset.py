import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import json
from pathlib import Path
import numpy as np
import random

class MovieTrailerDataset(Dataset):
    def __init__(self, metadata_path, processor, augment=False):
        self.processor = processor
        self.augment = augment
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def augment_image(self, image):
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item['image_path']).convert('RGB')

        if self.augment:
            image = self.augment_image(image)

        text = item['text']
        label = torch.tensor(item['label'], dtype=torch.float32)

        prompt = (
            f"Scene description: {text}\n"
            f"Task: Identify which of the predefined object categories appear in this movie scene. "
            f"Output format: probability scores for each category."
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = label

        return inputs

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
