import json
import os
from visualizations import ClassAccuracyAnalyzer

json_path = "model_output/per_class_accuracy_nh_b_f1_c3.json"  
with open(json_path, "r") as f:
    data = json.load(f)

class_names = data["class_names"]
accuracies = data["accuracies"]
sample_counts = data["sample_counts"]

plotter = ClassAccuracyAnalyzer(".")
plotter._plot_class_accuracy(class_names, accuracies, sample_counts)
print(f"Plot saved.")
