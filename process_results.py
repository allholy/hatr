import os
import re
from collections import defaultdict
import numpy as np

# Define root folder
root_dir = "model_output"

# Data structure to store metrics
metrics_data = defaultdict(lambda: defaultdict(list))

# Regex to extract metrics from results.txt
pattern = re.compile(r'(cls_acc|top_acc|cls_acc_2|top_acc_2):\s*([\d.]+)%')

# Walk through folder
for hierarchy in os.listdir(root_dir):
    hierarchy_path = os.path.join(root_dir, hierarchy)
    if not os.path.isdir(hierarchy_path):
        continue

    for mode in os.listdir(hierarchy_path):
        mode_path = os.path.join(hierarchy_path, mode)
        if not os.path.isdir(mode_path):
            continue

        for fold in os.listdir(mode_path):
            fold_path = os.path.join(mode_path, fold)
            results_file = os.path.join(fold_path, 'results.txt')
            if not os.path.isfile(results_file):
                continue

            with open(results_file, 'r') as f:
                content = f.read()
                for match in pattern.finditer(content):
                    metric_name = match.group(1)
                    value = float(match.group(2))
                    metrics_data[f"{hierarchy}/{mode}"][metric_name].append(value)

# Write the summary to a file
with open(f"{root_dir}/summary_metrics.txt", "w") as out_file:
    for folder_key, metrics in sorted(metrics_data.items()):
        out_file.write(f"{folder_key}:\n")
        for metric, values in sorted(metrics.items()):
            mean = np.mean(values)
            var = np.var(values)
            out_file.write(f"  {metric}: {mean:.2f}% +/- {var:.2f}%\n")
        out_file.write("\n")

print("Summary written to summary_metrics.txt")
