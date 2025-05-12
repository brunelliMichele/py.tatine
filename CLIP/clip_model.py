import torch
import clip
from PIL import Image
import os
import matplotlib.pyplot as plt

# Carica il modello CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory contenente le immagini da confrontare
image_folder = "./data/test/gallery"
filenames = os.listdir(image_folder)

# Carica e pre-processa tutte le immagini nel set
images = []
original_images = []
for fname in filenames:
    img_path = os.path.join(image_folder, fname)
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        images.append(image)
        original_images.append(fname)
    except:
        continue

images_tensor = torch.cat(images).to(device)

# Codifica tutte le immagini nel set
with torch.no_grad():
    image_features = model.encode_image(images_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)

# Carica e codifica l'immagine query
query_image_path = "./data/test/query/4597118805213184.jpg"
query_image = preprocess(Image.open(query_image_path).convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    query_feature = model.encode_image(query_image)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)

# Calcola la similarità coseno tra la query e le immagini
similarities = (query_feature @ image_features.T).squeeze(0)  # shape: [N]

# Ordina in base alla similarità
top_k = 5
best_indices = similarities.topk(top_k).indices

# Mostra i risultati
print(f"\nTop {top_k} immagini più simili a '{query_image_path}':")
for idx in best_indices:
    print(f"  - {original_images[idx]} (similarità: {similarities[idx].item():.4f})")

# Numero di risultati da visualizzare
top_k = min(5, len(best_indices)) # evita gli errori se ci sono meno di 5 immagini

# Crea una figura con una colonna in più per l'immagine query
plt.figure(figsize=(16, 5))

# Mostra l'immagine di query (posizione 0)
query_img = Image.open(query_image_path).convert("RGB")
plt.subplot(1, top_k + 1, 1)
plt.imshow(query_img)
plt.title("Query", fontsize=14, color="blue")
plt.axis("off")

# Mostra le immagini più simili
for i, idx in enumerate(best_indices[:top_k]):
    img_path = os.path.join(image_folder, original_images[idx])
    img = Image.open(img_path).convert("RGB")

    plt.subplot(1, top_k + 1, i + 2)
    plt.imshow(img)
    plt.title(f"Sim: {similarities[idx]:.2f}")
    plt.axis("off")

plt.suptitle(f"Top {top_k} immagini simili a '{query_image_path}'", fontsize=16)
plt.tight_layout()
plt.show()