from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import os
from typing import List
import shutil

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device = device)

# Cartella delle immagini
gallery_folder = "../data/test/gallery"

# Carica e codifica tutte le immagini della galleria
gallery_images = []
gallery_filenames = []
for fname in os.listdir(gallery_folder):
    img_path = os.path.join(gallery_folder, fname)
    try:
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        gallery_images.append(image)
        gallery_filenames.append(fname)
    except:
        continue

gallery_tensor = torch.cat(gallery_images).to(device)
with torch.no_grad():
    gallery_features = model.encode_image(gallery_tensor)
    gallery_features /= gallery_features.norm(dim = 1, keepdim = True)

@app.post("/search")
async def search_similar_images(query_image: UploadFile = File(...), top_k: int = 5):
    temp_path = f"./tmp_{query_image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(query_image.file, f)
    
    # preprocessa l'immagine query
    try:
        image = preprocess(Image.open(temp_path).convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code = 400)
    
    with torch.no_grad():
        query_feature = model.encode_image(image)
        query_feature /= query_feature.norm(dim = 1, keepdim = True)
        similarities = (query_feature @ gallery_features.T).squeeze(0)

    best_indicies = similarities.topk(top_k).indices

    results = [
        {
            "filename": gallery_filenames[idx],
            "similarity": round(similarities[idx].item(), 4)
        }
        for idx in best_indicies
    ]
    
    # Rimozione file tmp
    os.remove(temp_path)

    return {"results": results}