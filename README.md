# Py.tatine's Image Retrieval with Fine-Tuned Models

This repository contains multiple implementations and experimental pipelines for **image retrieval**, specifically developed for a **machine learning competition on image similarity search**.

The goal of the project is to retrieve the **top-10 most visually similar images** from a gallery given a query image. Different deep learning strategies for **fine-tuning convolutional neural networks (CNNs)** have been explored to improve the quality of learned image embeddings.

## Getting Started

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/brunelliMichele/py.tatine.git
    ```

2. **Move to the project folder:**
    ```sh
    cd py.tatine
    ```

3. **Install the required libraries and dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Strucutres

### Project structure

The repository is organized as follows:

```
.
├── data/
│   └── ...
├── desktop.ini
├── models
│   ├── CLIP
│   │   ├── clip_competition.py
│   │   ├── clip_fine_tuning.py
│   │   └── clip2.py
│   ├── DinoV2
│   │   └── main_dino_retrieval.py
│   ├── image-retrieval-with-efficientnet
│   │   ├── EfficientNetB0.py
│   │   ├── EfficientNetB4.py
│   │   └── merged_efficientNet.py
│   ├── manual_submission.py
│   ├── ResNet
│   │   ├── main_resnet50_retrieval.py
│   │   └── ResNet-Fine-Tuning
│   │       ├── L2-with-CrossEntropy.py
│   │       └── resNet_fine_tuning.py
│   ├── submission.json
│   ├── submit.py
│   └── VGG16
│       ├── main_vgg16_retrieval.py
│       └── VGG16_fine_tuned
│           └── vgg16_fine_tuning.py
├── README.md
├── report/
│   └── ...
├── requirements.txt
├── results
│   ├── CLIP
│   │   ├── fine-tuning/
│           └── ...
│   │   └── RN50x64/
│           └── ...
│   ├── DINO/
│       └── ...
│   ├── EfficientNet
│   │   ├── B0/
│           └── ...
│   │   ├── B4/
│           └── ...
│   │   └── merged/
│           └── ...
│   ├── ResNet
│   │   ├── L2_CrossEntropy/
│           └── ...
│   │   └── RN50/
│           └── ...
│   └── VGG16/
        └── ...
```


### Dataset structure

The dataset is structured as follows:

```
data/
├── training/           # Labeled training images in class folders
    └── ...           
└── test/
    ├── query/          # Unlabeled query images
        └── ... 
    └── gallery/        # Unlabeled gallery images
        └── ... 
```

## Models Implemented -- (TO CHECK)

| Model        | Type | Variants                  | Fine-tuning | Pooling      | Script(s)                                       |
|--------------|------|---------------------------|-------------|--------------|-------------------------------------------------|
| CLIP         | ViT  | ViT-B/32, RN50x64, others | Yes / No    | Internal     | `clip_competition.py`, `clip_fine_tuning.py`, `clip2.py` |
| DINOv2       | ViT  | facebook/dinov2-base       | No          | GAP          | `main_dino_retrieval.py`                        |
| EfficientNet | CNN  | B0, B4, merged             | No          | GeM / GAP    | `EfficientNetB0.py`, `EfficientNetB4.py`, `merged_efficientNet.py` |
| ResNet       | CNN  | ResNet50, fine-tuned       | Yes         | GAP / L2     | `main_resnet50_retrieval.py`, `resNet_fine_tuning.py`, `L2-with-CrossEntropy.py` |
| VGG16        | CNN  | VGG16                      | Yes         | GAP          | `main_vgg16_retrieval.py`, `vgg16_fine_tuning.py` |

## Report 

A full technical report is provided under `report/todo.pdf`, detailing all experimental settings, implementation choices, and results.

## Authors
This project was created by **Py.Tatine group** of the **Introduction to Machine Learning course** in the Master degree in **Data Science** at the University of Trento (Academic Year 2024–2025).

 - Brunelli Michele - [@brunelliMichele](https://github.com/brunelliMichele)
 - Danesi Francesco - [@FrancescoDanesi126](https://github.com/Francescodanesi126)
 - Ferrari Anna - [@annaferrari02](https://github.com/annaferrari02)
