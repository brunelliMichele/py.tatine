from submit import submit
import json

with open("results/ResNet/L2_CrossEntropy/submission.json", "r") as f:
    results = json.load(f)

submit(results, "Py.tatine")