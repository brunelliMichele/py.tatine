import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import faiss
import numpy as np
from PIL import Image

# Configurazione base
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = 'competition/data/training'
QUERY_DIR = 'competition/data/test/query'
GALLERY_DIR = 'competition/data/test/gallery'
OUTPUT_JSON = 'retrieval_results.json'
BATCH_SIZE = 32
EMBEDDING_SIZE = 512
TOP_K = 10

# Trasformazioni
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Caricamento dati training
train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modello base
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, EMBEDDING_SIZE)
model = model.to(DEVICE)

# Ottimizzatore e loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train():
    model.train()
    for epoch in range(5):  # puoi aumentare
        epoch_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), 'best_model.pth')

# Estrazione feature
def extract_features(directory):
    model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for img_name in tqdm(os.listdir(directory), desc=f"Extracting from {directory}"): #FORSE OS.WALK? 
            img_path = os.path.join(directory, img_name)
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            feat = model(tensor).cpu().numpy().flatten()
            features.append(feat)
            paths.append(img_path)
    return np.array(features).astype('float32'), paths

# Retrieval
def retrieve():
    gallery_feats, gallery_paths = extract_features(GALLERY_DIR)
    query_feats, query_paths = extract_features(QUERY_DIR)

    index = faiss.IndexFlatL2(EMBEDDING_SIZE)
    index.add(gallery_feats)
    D, I = index.search(query_feats, TOP_K)

    results = []
    for idx, query_path in enumerate(query_paths):
        result = {
            "filename": query_path,
            "gallery_images": [gallery_paths[i] for i in I[idx]]
        }
        results.append(result)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    train()
    retrieve()
