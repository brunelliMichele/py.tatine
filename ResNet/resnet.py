dataset_path= "data/training"


from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import random


tqdm.pandas()


transform = transforms.ToTensor()
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]


import os

image_files = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(root, file))

def display_random_images(image_files, num_images=5):
    random_images = random.sample(image_files, num_images)
    plt.figure(figsize=(15, 10))

    for i, img_path in enumerate(random_images):
        img = image.load_img(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')

    plt.show()

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(model, img_path):
    img_data = load_and_preprocess_image(img_path)
    features = model.predict(img_data, verbose=0)
    return features.flatten()

# ~ extract features for all images
image_features = []

for img_path in tqdm(image_files, desc="Extracting features"):
    features = extract_features(base_model, img_path)
    image_features.append(features)

image_features = np.array(image_features)

def search_similar_images(query_image_index, image_features, image_files, k=5):
    query_feature = image_features[query_image_index].reshape(1, -1)
    similarities = cosine_similarity(query_feature, image_features).flatten()
    top_k_indices = similarities.argsort()[-k-1:-1][::-1]  # ~ exclude the query image itself
    return top_k_indices

# EXAMPLE:
query_image_index = random.randint(0, len(image_files) - 1) # ~ get random image from

# ~ get top 5 similar images
top_k_indices = search_similar_images(query_image_index, image_features, image_files, k=5)

def display_images(query_image_path, similar_image_paths):
    plt.figure(figsize=(15, 5))

    # ~ show query image
    plt.subplot(1, len(similar_image_paths) + 1, 1)
    plt.imshow(image.load_img(query_image_path))
    plt.title("Query Image")
    plt.axis('off')

    # ~ show similar images
    for i, img_path in enumerate(similar_image_paths):
        plt.subplot(1, len(similar_image_paths) + 1, i + 2)
        plt.imshow(image.load_img(img_path))
        plt.title(f"Similar {i+1}")
        plt.axis('off')

    plt.show()

# ~ display results
query_image_path = image_files[query_image_index]
similar_image_paths = [image_files[idx] for idx in top_k_indices]
display_images(query_image_path, similar_image_paths)