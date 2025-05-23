import os
import json
import cv2
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex

# --- Load EfficientNetB4 (pretrained, no top) ---
efficientnet_model = tf.keras.applications.EfficientNetB4(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

# --- Feature extraction with L2 normalization ---
def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features.flatten()

# --- Load and prepare images with OpenCV ---
def load_and_prepare_image(path, size=(224, 224)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.resize(img, size)
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# --- Create image strip for visualization ---
def create_image_strip(query_path, similar_paths, size=(224, 224)):
    query_img = load_and_prepare_image(query_path, size)
    query_img = cv2.rectangle(query_img.copy(), (0, 0), (query_img.shape[1]-1, query_img.shape[0]-1), (0, 0, 255), 5)
    images = [query_img] + [load_and_prepare_image(p, size) for p in similar_paths]
    return np.concatenate(images, axis=1)

# --- Set up directories ---
base_data_dir = "data/test"
gallery_dir = os.path.join(base_data_dir, "gallery")
query_dir = os.path.join(base_data_dir, "query")
top_n = 10

# --- Extract gallery features ---
print("Extracting gallery features...")
gallery_images = [
    os.path.join(root, file)
    for root, _, files in os.walk(gallery_dir)
    for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]
gallery_features = [extract_features(p) for p in gallery_images]

# --- Build Annoy index ---
feature_dim = len(gallery_features[0])
annoy_index = AnnoyIndex(feature_dim, 'angular')
for i, feat in enumerate(gallery_features):
    annoy_index.add_item(i, feat)
annoy_index.build(200)
print("Annoy index built.")

# --- Process queries ---
query_images = [
    os.path.join(root, file)
    for root, _, files in os.walk(query_dir)
    for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]

results = []

for i, query_path in enumerate(query_images):
    print(f"\nProcessing query {i+1}/{len(query_images)}: {os.path.basename(query_path)}")
    query_feature = extract_features(query_path)
    similar_idxs = annoy_index.get_nns_by_vector(query_feature, top_n)
    similar_paths = [gallery_images[idx] for idx in similar_idxs]

    # Show image strip
    strip_img = create_image_strip(query_path, similar_paths)
    cv2.imshow(f"Query {i+1}: {os.path.basename(query_path)}", strip_img)
    cv2.waitKey(2000)

    # Save result
    results.append({
        "filename": query_path.replace("\\", "/"),
        "gallery_images": [p.replace("\\", "/") for p in similar_paths]
    })

cv2.destroyAllWindows()

# --- Save JSON results ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "retrieval_results_merged.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nRetrieval complete. Results saved to '{output_file}'.")
