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

# Improved triplet dataset for hard mining
class ImprovedTripletDataset(Dataset):
    def __init__(self, dataset, transform=None, triplets_per_image=3):
        self.dataset = dataset
        self.transform = transform
        self.labels = [label for _, label in dataset]
        self.triplets_per_image = triplets_per_image
        
        # Crea mapping label -> indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        
        # Pre-genera triplets per efficienza
        self.triplets = self._generate_triplets()
    
    def _generate_triplets(self):
        triplets = []
        all_labels = list(self.label_to_indices.keys())
        
        for idx in range(len(self.dataset)):
            anchor_label = self.labels[idx]
            positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
            negative_labels = [l for l in all_labels if l != anchor_label]
            
            if not positive_indices or not negative_labels:
                continue
            
            # Genera multipli triplet per ogni anchor
            for _ in range(self.triplets_per_image):
                pos_idx = random.choice(positive_indices)
                neg_label = random.choice(negative_labels)
                neg_idx = random.choice(self.label_to_indices[neg_label])
                triplets.append((idx, pos_idx, neg_idx))
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, pos_idx, neg_idx = self.triplets[idx]
        anchor_img, anchor_label = self.dataset[anchor_idx]
        pos_img, pos_label = self.dataset[pos_idx]
        neg_img, neg_label = self.dataset[neg_idx]
        
        # Restituisci come batch concatenato per hard mining
        return torch.cat([anchor_img.unsqueeze(0), pos_img.unsqueeze(0), neg_img.unsqueeze(0)], dim=0), \
               (anchor_label, pos_label, neg_label)

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

# Corrected Triplet Loss with proper forward signature
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, hard_mining=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        
    def forward(self, anchor, positive=None, negative=None):
        if self.hard_mining and positive is None and negative is None:
            # Hard mining mode: anchor contiene tutti gli embeddings concatenati
            return self.hard_triplet_loss(anchor)
        else:
            # Standard triplet loss con cosine distance
            dist_pos = 1 - F.cosine_similarity(anchor, positive, dim=1)
            dist_neg = 1 - F.cosine_similarity(anchor, negative, dim=1)
            losses = torch.relu(dist_pos - dist_neg + self.margin)
            return torch.mean(losses)
    
    def hard_triplet_loss(self, embeddings):
        """
        Hard negative mining triplet loss
        embeddings: tensor di shape [batch_size * 3, embedding_dim]
        """
        batch_size = embeddings.size(0) // 3
        anchor = embeddings[:batch_size]
        positive = embeddings[batch_size:2*batch_size]
        negative = embeddings[2*batch_size:]
        
        # Distanze positive (anchor-positive) con cosine
        pos_dist = 1 - F.cosine_similarity(anchor, positive, dim=1)
        
        # Per hard negative mining, calcola distanze tra tutti gli anchor e tutti i negative
        # Normalizza gli embeddings per cosine similarity
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)
        
        # Matrice delle similarità cosine
        cosine_sim = torch.mm(anchor_norm, negative_norm.t())
        # Converti in distanze cosine
        cosine_dist = 1 - cosine_sim
        
        # Hard negative mining: per ogni anchor, prendi il negativo più vicino (distanza minima)
        hard_neg_dist, _ = torch.min(cosine_dist, dim=1)
        
        # Triplet loss
        losses = torch.relu(pos_dist - hard_neg_dist + self.margin)
        
        return torch.mean(losses)

# Training function with hard mining
def train_with_hard_mining(model, triplet_train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    batch_count = 0
    
    for batch_data, batch_labels in triplet_train_loader:
        batch_count += 1
        # batch_data ha shape [batch_size, 3, channels, height, width]
        batch_size = batch_data.size(0)
        
        # Reshape per processare tutti insieme
        all_images = batch_data.view(-1, *batch_data.shape[2:]).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - ottieni embeddings per tutti
        all_embeddings = model(all_images)
        
        # Calcola loss con hard mining (passa solo embeddings)
        loss = criterion(all_embeddings)  # Solo un argomento per hard mining
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_count % 10 == 0:
            print(f"Batch {batch_count}, Loss: {loss.item():.4f}")
    
    return train_loss / len(triplet_train_loader)

# Function to find similar images using cosine similarity
def find_similar_images_cosine(query_embeddings, gallery_embeddings, gallery_paths, k=10):
    """
    Trova immagini simili usando cosine similarity invece di euclidean distance
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Calcola cosine similarity (più alto = più simile)
    similarities = cosine_similarity(query_embeddings, gallery_embeddings)
    
    # Ottieni i top-k indici per ogni query (ordinati dal più simile)
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    
    return top_k_indices

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
    train_dataset = torchvision.datasets.ImageFolder(root="../data/training", transform=transform)
    print(f"Found {len(train_dataset)} training images")
    
    # Create triplet dataset with hard mining support
    print("Creating improved triplet dataset...")
    triplet_dataset = ImprovedTripletDataset(train_dataset, triplets_per_image=3)
    print(f"Created triplet dataset with {len(triplet_dataset)} triplets")
    
    # Create data loaders
    triplet_train_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True, num_workers=0)  # Reduced batch size for hard mining
    
    print("Loading gallery and query datasets...")
    # Check if directories exist - use training as fallback if not
    gallery_dir = "../data/test/gallery" if os.path.exists("../data/test/gallery") else "data/training"
    query_dir = "../data/test/query" if os.path.exists("../data/test/query") else "data/training"
    
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
criterion = TripletLoss(margin=0.2, hard_mining=True)  # Reduced margin for cosine distance

# Training
num_epochs = 5
best_loss = float('inf')

print("Starting training with hard mining and cosine distance...")
start_time = time.time()
model_save_path = os.path.abspath('best_similarity_model.pth')

for epoch in range(num_epochs):
    avg_loss = train_with_hard_mining(model, triplet_train_loader, optimizer, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    
    # Save the best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Try alternative save location if first attempt fails
            alt_save_path = os.path.abspath(os.path.join(os.getcwd(), 'temp_best_similarity_model.pth'))
            try:
                torch.save(model.state_dict(), alt_save_path)
                print(f"Model saved to alternative location: {alt_save_path}")
            except Exception as alt_e:
                print(f"Failed to save model to alternative location: {alt_e}")
        
    scheduler.step()
    
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Early stopping check - if we're getting close to our time limit
    if elapsed_time > 5400:  # 90 minutes (leaving 30 minutes for inference)
        print("Time limit approaching, stopping training early")
        break

print("Training completed!")

# Load the best model for inference
try:
    model.load_state_dict(torch.load(model_save_path))
    print("Model loaded from primary location")
except Exception as e:
    print(f"Error loading model from primary location: {e}")
    # Try alternative location
    alt_save_path = os.path.abspath(os.path.join(os.getcwd(), 'temp_best_similarity_model.pth'))
    try:
        model.load_state_dict(torch.load(alt_save_path))
        print("Model loaded from alternative location")
    except Exception as alt_e:
        print(f"Failed to load model from any location: {alt_e}")
        print("Continuing with current model weights")
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

print("Finding nearest neighbors with cosine similarity...")
k = min(10, len(gallery_embeddings))
top_k_indices = find_similar_images_cosine(query_embeddings, gallery_embeddings, gallery_paths, k)

# Prepare submission
results = {}
for i, query_path in enumerate(query_paths):
    query_name = os.path.basename(query_path)
    similar_images = [os.path.basename(gallery_paths[idx]) for idx in top_k_indices[i]]
    results[query_name] = similar_images

# Write results to file
import json
with open('submission.json', 'w') as f:
    json.dump(results, f)

print("Submission file created with improved similarity matching!")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")