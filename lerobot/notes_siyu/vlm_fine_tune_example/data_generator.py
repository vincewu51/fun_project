import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path
from tags import YOLO_TAGS, TAG_TO_IDX

MOVIE_SCENES = [
    {"desc": "A person walking down a dark street at night", "tags": ["person", "car", "traffic light"]},
    {"desc": "Group of people sitting at dining table eating pizza", "tags": ["person", "dining table", "pizza", "chair", "cup"]},
    {"desc": "Car chase through city with motorcycle", "tags": ["car", "motorcycle", "person", "bus", "truck"]},
    {"desc": "Dog running in park with frisbee", "tags": ["dog", "frisbee", "person", "bench"]},
    {"desc": "Beach scene with surfboard and people", "tags": ["person", "surfboard", "umbrella", "backpack"]},
    {"desc": "Office meeting with laptop and cell phone", "tags": ["person", "laptop", "cell phone", "chair", "cup", "keyboard"]},
    {"desc": "Kitchen cooking scene", "tags": ["person", "refrigerator", "microwave", "bowl", "knife", "bottle"]},
    {"desc": "Living room watching tv", "tags": ["person", "tv", "couch", "remote", "potted plant"]},
    {"desc": "Train station with people and suitcases", "tags": ["person", "train", "suitcase", "backpack", "bench", "clock"]},
    {"desc": "Airport scene with airplane", "tags": ["airplane", "person", "suitcase", "backpack", "bench"]},
    {"desc": "Horse riding in countryside", "tags": ["horse", "person"]},
    {"desc": "Teddy bear in bedroom", "tags": ["teddy bear", "bed", "book", "clock"]},
    {"desc": "Birthday party with cake", "tags": ["person", "cake", "dining table", "chair", "cup", "fork"]},
    {"desc": "Tennis match in progress", "tags": ["person", "tennis racket", "sports ball", "bench"]},
    {"desc": "Baseball game scene", "tags": ["person", "baseball bat", "baseball glove", "sports ball", "bench"]},
]

def generate_mock_image(scene_desc, tags, size=(224, 224)):
    img = Image.new('RGB', size, color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()

    y_offset = 10
    for tag in tags[:5]:
        draw.rectangle([10, y_offset, 100, y_offset + 15], fill=(255, 255, 255, 180))
        draw.text((12, y_offset), tag, fill=(0, 0, 0), font=font)
        y_offset += 20

    for _ in range(random.randint(3, 8)):
        x, y = random.randint(0, size[0]-30), random.randint(0, size[1]-30)
        w, h = random.randint(20, 50), random.randint(20, 50)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)

    return img

def create_label_vector(tags):
    label = np.zeros(len(YOLO_TAGS), dtype=np.float32)
    for tag in tags:
        if tag in TAG_TO_IDX:
            label[TAG_TO_IDX[tag]] = 1.0
    return label

def generate_dataset(output_dir, num_samples=100, split='train'):
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = []

    for i in range(num_samples):
        scene = random.choice(MOVIE_SCENES)

        extra_tags = random.sample(YOLO_TAGS, k=random.randint(0, 3))
        all_tags = list(set(scene['tags'] + extra_tags))

        img = generate_mock_image(scene['desc'], all_tags)
        img_path = output_path / f"frame_{i:04d}.png"
        img.save(img_path)

        label = create_label_vector(all_tags)

        dataset.append({
            'image_path': str(img_path),
            'text': scene['desc'],
            'tags': all_tags,
            'label': label.tolist()
        })

    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    return dataset

if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'data'

    print("Generating training data...")
    train_data = generate_dataset(output_dir, num_samples=120, split='train')
    print(f"Generated {len(train_data)} training samples")

    print("Generating validation data...")
    val_data = generate_dataset(output_dir, num_samples=30, split='val')
    print(f"Generated {len(val_data)} validation samples")

    print("Generating test data...")
    test_data = generate_dataset(output_dir, num_samples=20, split='test')
    print(f"Generated {len(test_data)} test samples")
