import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import time
import random

# Custom dataset for image similarity
class ImageSimilarityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                           if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        self.transform = transform
        # For simplicity, we'll use image indices as pseudo-labels
        # In a real scenario, you'd use actual class labels if available
        self.labels = list(range(len(self.image_paths)))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

# Direct triplet dataset - more efficient for training
class DirectTripletDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = [label for _, label in dataset]
        
        self.triplets = []
        num_samples = len(dataset)

        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label_to_indices.setdefault(label, []).append(idx)
        
        all_labels = list(label_to_indices.keys())
        
        for idx in range(num_samples):
            anchor_label = self.labels[idx]
            positive_indices = label_to_indices[anchor_label].copy()
            positive_indices.remove(idx)
            negative_labels = [l for l in all_labels if l != anchor_label]
            negative_label = random.choice(negative_labels)
            negative_indices = label_to_indices[negative_label]

            if not positive_indices or not negative_indices:
                continue
            
            for _ in range(3):
                pos_idx = random.choice(positive_indices)
                neg_idx = random.choice(negative_indices)
                self.triplets.append((idx, pos_idx, neg_idx))

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        anchor_img, anchor_label = self.dataset[anchor_idx]
        pos_img, pos_label = self.dataset[pos_idx]
        neg_img, neg_label = self.dataset[neg_idx]
        return (anchor_img, pos_img, neg_img), (anchor_label, pos_label, neg_label)

# Embedding network
class EmbeddingNet(nn.Module):
    def __init__(self, base_model, embedding_size=128):
        super(EmbeddingNet, self).__init__()
        # Remove the original FC layer
        modules = list(base_model.children())[:-1]
        self.base_model = nn.Sequential(*modules)
        # Get feature dimension
        self.in_features = 2048  # For ResNet50
        # Add new FC layer for embeddings
        self.fc = nn.Linear(self.in_features, embedding_size)
        
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

# Triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        # Compute distances
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss
        losses = torch.relu(dist_pos - dist_neg + self.margin)
        return torch.mean(losses)

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Create datasets - modify the paths to point to your actual data directories
try:
    print("Loading training dataset...")
    train_dataset = torchvision.datasets.ImageFolder(root="data/training", transform=transform)
    print(f"Found {len(train_dataset)} training images")
    
    # Create triplet dataset
    print("Creating triplet dataset...")
    triplet_dataset = DirectTripletDataset(train_dataset)
    print(f"Created triplet dataset with {len(triplet_dataset)} triplets")
    
    # Create data loaders
    triplet_train_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Loading gallery and query datasets...")
    # Check if directories exist - use training as fallback if not
    gallery_dir = "data/test/gallery" if os.path.exists("data/test/gallery") else "data/training"
    query_dir = "data/test/query" if os.path.exists("data/test/query") else "data/training"
    
    gallery_dataset = ImageSimilarityDataset(image_dir=gallery_dir, transform=transform)
    query_dataset = ImageSimilarityDataset(image_dir=query_dir, transform=transform)
    
    print(f"Found {len(gallery_dataset)} gallery images and {len(query_dataset)} query images")
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=0)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=0)
    
except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

# Create model
print("Creating model...")
base_model = resnet50(pretrained=True)
model = EmbeddingNet(base_model)
model = model.to(device)

# Unfreeze last few convolutional layers for fine-tuning
ct = 0
for child in model.base_model.children():
    ct += 1
    if ct < 7:  # Freeze early layers
        for param in child.parameters():
            param.requires_grad = False

# Optimizer and loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = TripletLoss(margin=0.5)

# Training
num_epochs = 5
best_loss = float('inf')

print("Starting training...")
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    for (anchor_img, pos_img, neg_img), _ in triplet_train_loader:
        batch_count += 1
        anchor_img = anchor_img.to(device)
        pos_img = pos_img.to(device)
        neg_img = neg_img.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        anchor_emb = model(anchor_img)
        pos_emb = model(pos_img)
        neg_emb = model(neg_img)
        
        # Calculate loss
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Print batch progress every 10 batches
        if batch_count % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    avg_loss = train_loss / len(triplet_train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    
    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'best_similarity_model.pth')
    
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Early stopping check - if we're getting close to our time limit
    if elapsed_time > 5400:  # 90 minutes (leaving 30 minutes for inference)
        print("Time limit approaching, stopping training early")
        break

print("Training completed!")

# Load the best model for inference
model.load_state_dict(torch.load('best_similarity_model.pth'))
model.eval()

print("Extracting gallery embeddings...")
# Extract embeddings for gallery images
gallery_embeddings = []
gallery_paths = []

with torch.no_grad():
    for images, _, paths in gallery_loader:
        images = images.to(device)
        outputs = model(images)
        gallery_embeddings.append(outputs.cpu().numpy())
        gallery_paths.extend(paths)

gallery_embeddings = np.vstack(gallery_embeddings)
print(f"Gallery embeddings shape: {gallery_embeddings.shape}")

print("Extracting query embeddings...")
# Extract embeddings for query images
query_embeddings = []
query_paths = []

with torch.no_grad():
    for images, _, paths in query_loader:
        images = images.to(device)
        outputs = model(images)
        query_embeddings.append(outputs.cpu().numpy())
        query_paths.extend(paths)

query_embeddings = np.vstack(query_embeddings)
print(f"Query embeddings shape: {query_embeddings.shape}")

print("Finding nearest neighbors...")
# Find top-10 similar images for each query
k = min(10, len(gallery_embeddings))  # Top-k, ensuring we don't exceed gallery size
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean').fit(gallery_embeddings)
distances, indices = nbrs.kneighbors(query_embeddings)

# Prepare submission
results = {}
for i, query_path in enumerate(query_paths):
    query_name = os.path.basename(query_path)
    similar_images = [os.path.basename(gallery_paths[idx]) for idx in indices[i]]
    results[query_name] = similar_images

# Write results to file
import json
with open('submission.json', 'w') as f:
    json.dump(results, f)

print("Submission file created.")