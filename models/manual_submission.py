from submit import submit
import json

with open("results/EfficientNet/merged/submission.json", "r") as f:
    results = json.load(f)

submit(results, "Py.tatine")