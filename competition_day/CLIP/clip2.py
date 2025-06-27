import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "gallery")
OUTPUT_FILE = os.path.join(BASE_DIR, "submission.json")
TOP_K = 10

# Enhanced configuration
BATCH_SIZE = 64  # Process images in batches to avoid memory issues
USE_ENSEMBLE = True  # Use multiple CLIP models
USE_TTA = True  # Test Time Augmentation
RERANK_TOP_K = 50  # Rerank top candidates

# Load multiple CLIP models for ensemble
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

models_info = [
    ("ViT-B/32", 1.0),
    ("ViT-B/16", 1.2),  # Higher weight for better model
    ("ViT-L/14", 1.5) if torch.cuda.is_available() else ("RN50", 0.8),  # Use large model on GPU
]

models = []
preprocessors = []

print("🔄 Loading CLIP models...")
for model_name, weight in models_info:
    try:
        model, preprocess = clip.load(model_name, device=device)
        models.append((model, weight))
        preprocessors.append(preprocess)
        print(f"✅ Loaded {model_name}")
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}: {e}")

if not models:
    # Fallback to single model
    model, preprocess = clip.load("ViT-B/32", device=device)
    models = [(model, 1.0)]
    preprocessors = [preprocess]

def create_tta_transforms():
    """Create Test Time Augmentation transforms"""
    transforms = []
    
    # Original transform
    transforms.append(preprocessors[0])
    
    if USE_TTA:
        # Horizontal flip
        flip_transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        transforms.append(flip_transform)
        
        # Slight rotation and scale variations
        from torchvision.transforms import RandomRotation, RandomResizedCrop
        augment_transform = Compose([
            RandomResizedCrop(224, scale=(0.95, 1.0), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        transforms.append(augment_transform)
    
    return transforms

def load_images_from_folder_enhanced(folder, batch_size=BATCH_SIZE):
    """Enhanced image loading with batching and TTA"""
    filenames = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            filenames.append(fname)
    
    all_features = []
    tta_transforms = create_tta_transforms()
    
    print(f"🔄 Processing {len(filenames)} images with {len(tta_transforms)} augmentations...")
    
    for i in tqdm(range(0, len(filenames), batch_size), desc="Processing batches"):
        batch_files = filenames[i:i+batch_size]
        batch_features = []
        
        for transform_idx, transform in enumerate(tta_transforms):
            batch_images = []
            
            for fname in batch_files:
                path = os.path.join(folder, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Errore con {fname}: {e}")
                    # Use a dummy tensor for failed images
                    batch_images.append(torch.zeros(3, 224, 224))
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                
                # Extract features using ensemble of models
                ensemble_features = []
                for model, weight in models:
                    with torch.no_grad():
                        features = model.encode_image(batch_tensor).float()
                        features = F.normalize(features, p=2, dim=1)
                        ensemble_features.append(features * weight)
                
                # Weighted average of ensemble
                avg_features = torch.stack(ensemble_features).mean(dim=0)
                batch_features.append(avg_features)
        
        # Average features across TTA transforms
        if batch_features:
            tta_averaged = torch.stack(batch_features).mean(dim=0)
            all_features.append(tta_averaged)
    
    if all_features:
        final_features = torch.cat(all_features, dim=0)
        # Final normalization
        final_features = F.normalize(final_features, p=2, dim=1)
        return final_features, filenames
    else:
        return torch.empty(0, 512).to(device), []

def advanced_similarity_scoring(query_features, gallery_features):
    """Advanced similarity with multiple metrics"""
    # Standard cosine similarity
    cosine_sim = query_features @ gallery_features.T
    
    # Add small regularization to avoid extreme similarities
    cosine_sim = cosine_sim * 0.95
    
    # Optional: Add Euclidean distance component (inverted)
    query_expanded = query_features.unsqueeze(1)  # (num_query, 1, dim)
    gallery_expanded = gallery_features.unsqueeze(0)  # (1, num_gallery, dim)
    euclidean_dist = torch.norm(query_expanded - gallery_expanded, dim=2)
    euclidean_sim = 1 / (1 + euclidean_dist * 0.1)  # Convert to similarity
    
    # Combine similarities
    combined_sim = 0.8 * cosine_sim + 0.2 * euclidean_sim
    
    return combined_sim

def query_expansion(query_features, gallery_features, top_k=5, alpha=0.3):
    """Expand query with top similar gallery images"""
    similarity = query_features @ gallery_features.T
    _, top_indices = similarity.topk(top_k, dim=1)
    
    expanded_queries = []
    for i, query_feat in enumerate(query_features):
        top_gallery_feats = gallery_features[top_indices[i]]
        # Weighted combination: original query + top similar images
        expanded = (1 - alpha) * query_feat + alpha * top_gallery_feats.mean(dim=0)
        expanded = F.normalize(expanded.unsqueeze(0), p=2, dim=1)
        expanded_queries.append(expanded)
    
    return torch.cat(expanded_queries, dim=0)

def show_retrieval_results(query_dir, gallery_dir, results):
    """Enhanced visualization with similarity scores"""
    results_list = [{"filename": k, "samples": v} for k, v in results.items()]
    
    for item in results_list[:3]:  # Show only first 3 for brevity
        query_img = Image.open(os.path.join(query_dir, item["filename"])).convert("RGB")
        gallery_imgs = [Image.open(os.path.join(gallery_dir, fname)).convert("RGB") for fname in item["samples"][:5]]

        fig, axes = plt.subplots(1, len(gallery_imgs) + 1, figsize=(18, 4))
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query\n{item['filename']}", fontsize=10)
        axes[0].axis("off")

        for i, img in enumerate(gallery_imgs):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Top {i+1}\n{item['samples'][i]}", fontsize=8)
            axes[i + 1].axis("off")
        
        plt.tight_layout()
        plt.show()

# Main execution
print("🔄 Loading gallery images...")
gallery_features, gallery_filenames = load_images_from_folder_enhanced(GALLERY_DIR)
print(f"✅ {len(gallery_features)} gallery images processed")

print("🔄 Loading query images...")
query_features, query_filenames = load_images_from_folder_enhanced(QUERY_DIR)
print(f"✅ {len(query_features)} query images processed")

print("🔄 Performing query expansion...")
expanded_query_features = query_expansion(query_features, gallery_features)

print("🔄 Computing enhanced similarities...")
similarity_scores = advanced_similarity_scoring(expanded_query_features, gallery_features)

print("🔄 Retrieving top matches...")
topk_values, topk_indices = similarity_scores.topk(TOP_K, dim=1)

results = {}
for i, query_fname in enumerate(query_filenames):
    top_gallery_files = [gallery_filenames[idx] for idx in topk_indices[i]]
    results[query_fname] = top_gallery_files

print("💾 Saving results...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Enhanced retrieval complete! Output saved to {OUTPUT_FILE}")
print(f"📊 Improvements applied:")
print(f"   • Model ensemble ({len(models)} models)")
print(f"   • Test Time Augmentation: {USE_TTA}")
print(f"   • Query expansion")
print(f"   • Advanced similarity scoring")
print(f"   • Batch processing for efficiency")

# Visualize results
print("🎨 Showing retrieval examples...")
show_retrieval_results(QUERY_DIR, GALLERY_DIR, results)