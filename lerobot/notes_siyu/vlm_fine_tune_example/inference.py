import torch
from PIL import Image
import numpy as np
from pathlib import Path
import json

from model import Qwen2VLClassifier, get_processor
from tags import YOLO_TAGS, NUM_TAGS

class MovieTrailerPredictor:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading processor...")
        self.processor = get_processor()

        print(f"Loading model on {self.device}...")
        self.model = Qwen2VLClassifier(
            use_lora=True,
            use_gradient_checkpointing=False,
            pooling_strategy='attention',
            loss_type='focal'
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} with F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded checkpoint from {checkpoint_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path, text_description, threshold=0.3, top_k=10):
        image = Image.open(image_path).convert('RGB')

        prompt = (
            f"Scene description: {text_description}\n"
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

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()[0]

        probs = np.clip(probs, 0.0, 1.0)

        results = []
        for idx, prob in enumerate(probs):
            if prob > threshold:
                results.append({
                    'tag': YOLO_TAGS[idx],
                    'probability': float(prob),
                    'index': int(idx)
                })

        results.sort(key=lambda x: x['probability'], reverse=True)

        if len(results) > top_k:
            results = results[:top_k]

        return results

    def batch_predict(self, image_paths, text_descriptions, threshold=0.5, top_k=10):
        all_results = []

        for img_path, text_desc in zip(image_paths, text_descriptions):
            results = self.predict(img_path, text_desc, threshold, top_k)
            all_results.append({
                'image': str(img_path),
                'text': text_desc,
                'predictions': results
            })

        return all_results

def main():
    data_dir = Path(__file__).parent / 'data' / 'test'
    checkpoint_path = Path(__file__).parent / 'checkpoints' / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Using untrained model for demo...")
        checkpoint_path = None

    predictor = MovieTrailerPredictor(checkpoint_path)

    with open(data_dir / 'metadata.json', 'r') as f:
        test_data = json.load(f)

    print(f"\nRunning inference on {len(test_data)} test samples...\n")

    for i, item in enumerate(test_data[:5]):
        print(f"=== Sample {i+1} ===")
        print(f"Text: {item['text']}")
        print(f"Ground Truth Tags: {', '.join(item['tags'])}")

        results = predictor.predict(item['image_path'], item['text'], threshold=0.3, top_k=10)

        print("Predictions:")
        for pred in results:
            print(f"  {pred['tag']}: {pred['probability']:.4f}")
        print()

if __name__ == '__main__':
    main()
