def compute_topk_accuracy(results, ground_truth, k):

    correct = 0
    total = 0

    for query_id, predicted_gallery_ids in results.items():
        gt_ids = ground_truth.get(query_id, [])
        topk_preds = predicted_gallery_ids[:k]

        if any(gt in topk_preds for gt in gt_ids):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0