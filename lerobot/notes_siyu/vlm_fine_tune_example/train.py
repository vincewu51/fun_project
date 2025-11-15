import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from tqdm import tqdm
import json

from model import Qwen2VLClassifier, get_processor
from dataset import MovieTrailerDataset, collate_fn
from tags import YOLO_TAGS

def compute_metrics(preds, labels, threshold=0.5):
    preds_binary = (preds > threshold).astype(int)
    labels_binary = labels.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_binary, preds_binary, average='micro', zero_division=0
    )

    try:
        map_score = average_precision_score(labels_binary, preds, average='micro')
    except:
        map_score = 0.0

    try:
        roc_auc = roc_auc_score(labels_binary, preds, average='micro')
    except:
        roc_auc = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map': map_score,
        'roc_auc': roc_auc
    }

def train_epoch(model, dataloader, optimizer, device, scaler=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if scaler is not None:
            with autocast():
                outputs = model(pixel_values, input_ids, attention_mask, labels)
                loss = outputs['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(outputs['logits']).cpu().detach().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs['loss']

            total_loss += loss.item()

            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    return avg_loss, metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(__file__).parent / 'data'

    print("Loading processor...")
    processor = get_processor()

    print("Loading model...")
    model = Qwen2VLClassifier(
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        use_gradient_checkpointing=True,
        pooling_strategy='attention',
        loss_type='focal'
    )
    model = model.to(device)

    print("Loading datasets...")
    train_dataset = MovieTrailerDataset(
        data_dir / 'train' / 'metadata.json',
        processor,
        augment=True
    )
    val_dataset = MovieTrailerDataset(
        data_dir / 'val' / 'metadata.json',
        processor,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    num_epochs = 10
    warmup_epochs = 2
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler() if device.type == 'cuda' else None

    best_val_f1 = 0
    save_dir = Path(__file__).parent / 'checkpoints'
    save_dir.mkdir(exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, scaler)
        val_loss, val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val - Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, mAP: {val_metrics['map']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': best_val_f1
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            print(f"Saved best model with F1: {best_val_f1:.4f}")

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining completed!")

if __name__ == '__main__':
    main()
