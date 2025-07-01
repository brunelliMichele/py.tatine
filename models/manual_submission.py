from submit import submit
import json

with open("results/VGG16/submission.json", "r") as f:
    results = json.load(f)

submit(results, "Py.tatine")