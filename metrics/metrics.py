import os

def build_filename_to_class_mapping(dataset_dir):
    """
    Costruisce una mappa: nome_file.jpg → nome_classe (cartella)
    Scorre tutte le sottocartelle e associa il nome del file alla sua classe.
    """
    mapping = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                class_name = os.path.basename(root)
                mapping[f] = class_name
    return mapping

def top_k_accuracy(res, filename_to_class, k=10):
    """
    Top-k accuracy: almeno 1 immagine rilevata ha la stessa classe della query.
    """
    correct = 0
    total = 0
    for qfile, retrieved_files in res.items():
        q_class = filename_to_class.get(qfile)
        if q_class is None:
            continue  # file non trovato nella mappa
        retrieved_classes = [filename_to_class.get(f) for f in retrieved_files[:k]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"[METRIC] Top-{k} Accuracy: {acc:.4f}")
    return acc

def precision_at_k(res, filename_to_class, k=10):
    """
    Precision@k: media delle proporzioni di immagini rilevate che hanno la stessa classe della query.
    """
    total_precision = 0
    total_queries = 0
    for qfile, retrieved_files in res.items():
        q_class = filename_to_class.get(qfile)
        if q_class is None:
            continue
        retrieved_classes = [filename_to_class.get(f) for f in retrieved_files[:k]]
        correct = sum(1 for c in retrieved_classes if c == q_class)
        total_precision += correct / k
        total_queries += 1
    avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
    print(f"[METRIC] Precision@{k}: {avg_precision:.4f}")
    return avg_precision

dataset_dir = "INSERT DATASET PATH"
