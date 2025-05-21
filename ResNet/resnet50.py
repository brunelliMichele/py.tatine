import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

#set paths
data_path = "data/test"
query_path = os.path.join(data_path, "query")
gallery_path = os.path.join(data_path, "gallery")

#load and preprocess
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

def get_image_files(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(model, image_paths):
    features = []
    for img_path in tqdm(image_paths, desc=f"Extracting features from {len(image_paths)} images"):
        img_data = load_and_preprocess_image(img_path)
        feat = model.predict(img_data, verbose=0)
        features.append(feat.flatten())
    return np.array(features)

#load image paths
query_images = get_image_files(query_path)
gallery_images = get_image_files(gallery_path)

#extract features
query_features = extract_features(base_model, query_images)
gallery_features = extract_features(base_model, gallery_images)

#find top-10 similar images for each query
def display_results(query_path, top_k_paths):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 11, 1)
    plt.imshow(image.load_img(query_path))
    plt.title("Query")
    plt.axis('off')
    for i, path in enumerate(top_k_paths):
        plt.subplot(1, 11, i + 2)
        plt.imshow(image.load_img(path))
        plt.title(f"Top {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#loop through all query images and display results
try:
    for i, q_feat in enumerate(query_features):
        sims = cosine_similarity(q_feat.reshape(1, -1), gallery_features).flatten()
        top_k_indices = sims.argsort()[-10:][::-1]
        top_k_paths = [gallery_images[idx] for idx in top_k_indices]
        print(f"Query {i+1}/{len(query_images)}: {os.path.basename(query_images[i])}")
        display_results(query_images[i], top_k_paths)
        time.sleep(1)  # Pause between queries for better viewing
except KeyboardInterrupt:
    print("Interrupted by user.")

