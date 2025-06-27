import os
import sys
# In alto, sotto import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from submit import submit
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
from torchvision.models import resnet18, ResNet18_Weights
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configurazione base
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUERY_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "..", "data", "test", "gallery")
TRAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "training"))
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
print("Classi trovate:", train_dataset.class_to_idx)
print("Numero classi:", len(train_dataset.classes))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modello base
model = resnet18(weights=ResNet18_Weights.DEFAULT)
NUM_CLASSES = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
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
            print("Labels dtype:", labels.dtype)
            print("Labels min:", labels.min().item(), "max:", labels.max().item())
            print("Outputs shape:", outputs.shape)
            loss = criterion(outputs, labels)
            print("Labels shape:", labels.shape)
            print("Labels dtype:", labels.dtype)
            print("Labels min/max:", labels.min().item(), labels.max().item())
            print("Outputs shape:", outputs.shape)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), 'best_model.pth')

# Estrazione feature
feature_model = resnet18(weights=ResNet18_Weights.DEFAULT)
feature_model.fc = nn.Identity()
feature_model.load_state_dict(torch.load('best_model.pth'), strict=False)
feature_model = feature_model.to(DEVICE)
def extract_features(directory):
    feature_model.eval()
    features = []
    paths = []
    with torch.no_grad():
        for img_name in tqdm(os.listdir(directory), desc=f"Extracting from {directory}"):
            img_path = os.path.join(directory, img_name)
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            feat = feature_model(tensor).cpu().numpy().flatten()
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

    results = {}
    for idx, query_path in enumerate(query_paths):
        query_fname = os.path.basename(query_path)
        gallery_fnames = [os.path.basename(gallery_paths[i]) for i in I[idx]]
        results[query_fname] = gallery_fnames

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_JSON}")

    submit(results, "Py.tatine")

if __name__ == '__main__':
    train()
    retrieve()
