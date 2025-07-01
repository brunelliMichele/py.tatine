from submit import submit
import json

with open("results/CLIP/fine-tuning/submission.json", "r") as f:
    results = json.load(f)

submit(results, "Py.tatine")