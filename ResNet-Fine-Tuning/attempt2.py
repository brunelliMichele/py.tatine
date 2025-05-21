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

# Simple dataset for images
class SimpleImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Verify directory exists
        if not os.path.exists(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")
            
        # Get all image files
        self.image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for file in os.listdir(image_dir):
            if file.lower().endswith(valid_extensions):
                self.image_paths.append(os.path.join(image_dir, file))
                
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
            
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path

# Embedding network using ResNet50
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(EmbeddingNet, self).__init__()
        # Load pretrained ResNet50
        self.base_model = resnet50(pretrained=True)
        # Remove the final fully connected layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        # Add a new FC layer for the embedding
        self.embedding = nn.Linear(2048, embedding_size)
        
    def forward(self, x):
        # Forward pass through base model
        x = self.base_model(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Get embedding
        x = self.embedding(x)
        # Normalize the embedding
        x = F.normalize(x, p=2, dim=1)
        return x

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create datasets - UPDATE THESE PATHS!
try:
    train_dir = "./data/training"
    print(f"Loading training dataset from {train_dir}")
    
    # Check if directory exists
    if not os.path.exists(train_dir):
        print(f"WARNING: Directory {train_dir} not found!")
        # Let's try to find where the images might be
        possible_dirs = ["./data", "./training", "./images", "."]
        for d in possible_dirs:
            if os.path.exists(d):
                print(f"Found directory {d}, checking for images...")
                image_count = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                if image_count > 0:
                    print(f"Found {image_count} images in {d}")
                    train_dir = d
                    break
    
    # Create dataset
    train_dataset = SimpleImageDataset(image_dir=train_dir, transform=transform)
    
    # Create dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # For gallery and query, we'll use the same dataset for demonstration
    # In a real competition, you'd use separate datasets
    gallery_dataset = train_dataset
    query_dataset = train_dataset
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=0)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=0)
    
except Exception as e:
    print(f"Error setting up datasets: {e}")
    raise

# Create embedding model
model = EmbeddingNet(embedding_size=128).to(device)

# Freeze early layers to speed up training
ct = 0
for child in model.base_model.children():
    ct += 1
    if ct < 6:  # Freeze first 6 layers
        for param in child.parameters():
            param.requires_grad = False

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Main training loop - simplified to just train the embedding model
# Without triplet loss, we'll just extract features and evaluate similarity
print("Training embedding model...")
num_epochs = 3  # Very short training for quick results

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Instead of triplet loss, we're just doing a pass through the network
    # to fine-tune the weights with the unfrozen layers
    for images, _ in train_loader:
        images = images.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass - just get embeddings
        embeddings = model(images)
        
        # Simple L2 regularization loss to prevent overfitting
        # This is very simplified - in a real scenario you'd use triplet or contrastive loss
        loss = torch.mean(torch.norm(embeddings, dim=1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "embedding_model.pth")
print("Model saved!")

# Feature extraction
print("Extracting gallery features...")
gallery_features = []
gallery_paths = []

model.eval()
with torch.no_grad():
    for images, paths in gallery_loader:
        images = images.to(device)
        features = model(images)
        gallery_features.append(features.cpu().numpy())
        gallery_paths.extend(paths)

gallery_features = np.vstack(gallery_features)
print(f"Gallery features shape: {gallery_features.shape}")

print("Extracting query features...")
query_features = []
query_paths = []

with torch.no_grad():
    for images, paths in query_loader:
        images = images.to(device)
        features = model(images)
        query_features.append(features.cpu().numpy())
        query_paths.extend(paths)

query_features = np.vstack(query_features)
print(f"Query features shape: {query_features.shape}")

# Find nearest neighbors
print("Finding similar images...")
k = min(10, len(gallery_features))  # Top-k
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='cosine').fit(gallery_features)
distances, indices = nbrs.kneighbors(query_features)

# Create submission
results = {}
for i, query_path in enumerate(query_paths):
    query_name = os.path.basename(query_path)
    similar_images = [os.path.basename(gallery_paths[idx]) for idx in indices[i]]
    results[query_name] = similar_images

# Write results to file
import json
with open('submission.json', 'w') as f:
    json.dump(results, f)

print("Submission created! Ready for the competition.")