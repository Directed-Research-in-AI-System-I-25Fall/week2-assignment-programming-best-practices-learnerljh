import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetForImageClassification
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load the MNIST dataset
mnist = load_dataset("mnist")
images = mnist["test"]["image"]
labels = mnist["test"]["label"]
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
correct_predictions = 0

for image, true_label in zip(images, labels):
    image = Image.fromarray(image).resize((224, 224))

    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    if predicted_label == true_label:
        correct_predictions += 1

accuracy = correct_predictions / len(labels)
print(f"Accuracy: {accuracy:.2f}")