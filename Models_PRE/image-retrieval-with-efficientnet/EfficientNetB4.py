
import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import random
from annoy import AnnoyIndex
import json
from metrics import build_filename_to_class_mapping, top_k_accuracy, precision_at_k

# --- Setup paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
#from submit import submit

# --- EfficientNetB4 model ---
efficientnet_model = tf.keras.applications.EfficientNetB4(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features.flatten()

def load_and_prepare_image(path, size=(224, 224)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.resize(img, size)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def create_image_strip(query_path, similar_paths, size=(224, 224)):
    query_img = load_and_prepare_image(query_path, size)
    thickness = 5
    color = (0, 0, 255)  # Red border
    query_img = cv2.rectangle(query_img.copy(), (0, 0), (query_img.shape[1]-1, query_img.shape[0]-1), color, thickness)

    images = [query_img]
    for path in similar_paths:
        img = load_and_prepare_image(path, size)
        images.append(img)

    strip = np.concatenate(images, axis=1)
    return strip

# --- Use one directory for all images (may contain subfolders) ---
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data_preEval", "training"))

# --- Collect all image paths ---
all_images = [
    os.path.join(root, file)
    for root, _, files in os.walk(IMAGE_DIR)
    for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

if len(all_images) < 20:
    raise ValueError("❌ Not enough images found in the folder. Need at least 20 images.")

print(f"📁 Found {len(all_images)} images in total.")

# --- Randomly select 20 query images ---
random.seed(42)
query_images = random.sample(all_images, 20)

# --- Extract features for all images ---
print("⏳ Extracting features for all images...")
features = [extract_features(path) for path in all_images]
print("✅ Feature extraction complete.")

# --- Build Annoy index ---
feature_dim = len(features[0])
annoy_index = AnnoyIndex(feature_dim, 'angular')
for i, feat in enumerate(features):
    annoy_index.add_item(i, feat)
annoy_index.build(200)
print("🧠 Annoy index built.")

top_n = 10
results = {}

# --- Process each query ---
for i, query_path in enumerate(query_images):
    print(f"\n🔎 Query {i+1}/20: {os.path.basename(query_path)}")
    query_idx = all_images.index(query_path)
    query_feature = features[query_idx]

    # Get top_n + 1 neighbors (include self) then exclude query
    similar_idxs = annoy_index.get_nns_by_vector(query_feature, top_n + 1)
    similar_idxs = [idx for idx in similar_idxs if idx != query_idx][:top_n]
    similar_paths = [all_images[idx] for idx in similar_idxs]

    # Record results
    query_fname = os.path.basename(query_path).replace("\\", "/")
    gallery_fnames = [os.path.basename(p).replace("\\", "/") for p in similar_paths]
    results[query_fname] = gallery_fnames

    # Visualize
    #strip_img = create_image_strip(query_path, similar_paths)
    #cv2.imshow(f"Query {i+1}: {os.path.basename(query_path)}", strip_img)
    #print("📸 Showing top 10 similar images.")
    #cv2.waitKey(2000)

#cv2.destroyAllWindows()

# --- Save results to JSON ---
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "EfficientNet", "B4", "submission.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Retrieval complete. Results saved to '{OUTPUT_FILE}'.")
#submit(results, "Py.tatine")

dataset_dir = os.path.join(BASE_DIR, "..","..", "data_preEval", "training")

filename_mapping = build_filename_to_class_mapping(dataset_dir)

acc= top_k_accuracy(results, filename_mapping)
prec= precision_at_k(results, filename_mapping)

print("Accuracy=", acc)
print("Precision=", prec)