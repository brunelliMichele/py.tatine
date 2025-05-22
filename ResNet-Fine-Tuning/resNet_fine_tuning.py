import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm

# 📁 Percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "..", "data", "training")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "..", "resnet50_finetuned.pth")

# ⚙️ Parametri
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 0

# 🔄 Trasformazioni
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📥 Dataset e DataLoader
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)

# 🧠 Modello ResNet50 pre-addestrato
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Congela tutti i layer tranne l'ultimo
for param in model.parameters():
    param.requires_grad = False

# Sostituisce il classificatore finale
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# ⚙️ Loss e ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 🔁 Loop di training
print("🚀 Inizio fine-tuning (solo fc)...")
model.train()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"📉 Loss epoca {epoch+1}: {epoch_loss:.4f}")

# 💾 Salvataggio modello
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Fine-tuning completato. Modello salvato in {MODEL_SAVE_PATH}")