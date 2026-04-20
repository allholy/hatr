import os
import json
import csv
import numpy as np
import torch
from models import BaseClassifier
import datetime


# --- Configuration
# AUDIO_FOLDER = "/media/anastp/DATA/datasets/FSD50K/FSD50K_processed/features/CLAP_audio_embeddings"
# TEXT_FOLDER = "/media/anastp/DATA/datasets/FSD50K/FSD50K_processed/features/CLAP_text_embeddings_description"
AUDIO_FOLDER = "/media/anastp/DATA/datasets/PSE/pse_sonic_data/embeddings/clap"
TEXT_FOLDER = "/media/anastp/DATA/datasets/PSE/pse_sonic_data/embeddings/clap_text"
MODE = "both"  # "audio", "text", or "both"
MODEL_WEIGHTS = f"model_output/t-contr_ce_penalty/{MODE}/fold_0/best_model.pth"
output_mode = "multimodal" if MODE == "both" else MODE
OUTPUT_CSV = f"../predictions/pse_bst_predictions_{output_mode}.csv"
# ------------------

# Load class names
with open("class_dict.json", "r") as f:
    class_dict = json.load(f)
index_to_name = {v: k for k, v in class_dict.items()}  # convert index back to name

# Load the model
model = BaseClassifier(
    hidden_size=128,
    num_classes=len(class_dict),
    emb_size_audio=512,
    emb_size_text=512,
    mode=MODE
)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location="cpu"))
model.eval()  # set model to evaluation mode

# Predict the class
def predict_class(audio_emb, text_emb):
    audio_tensor = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0) if audio_emb is not None else None
    text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0) if text_emb is not None else None

    with torch.no_grad():  # no gradients needed for prediction
        _, logits = model(audio_emb=audio_tensor, text_emb=text_tensor)
        pred_class_idx = torch.argmax(logits, dim=1).item()

    return pred_class_idx, index_to_name[pred_class_idx]

# Get list of files to process
def list_files(folder):
    """Recursively list all .npy files from a folder and return relative paths."""
    all_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".npy"):
                full_path = os.path.join(root, file)
                all_files.append(os.path.relpath(full_path, folder))
    return all_files

def get_file_list(mode):
    if mode == "audio":
        return sorted(list_files(AUDIO_FOLDER))
    elif mode == "text":
        return sorted(list_files(TEXT_FOLDER))
    else:  # both
        audio_files = set(list_files(AUDIO_FOLDER))
        text_files = set(list_files(TEXT_FOLDER))
        return sorted(audio_files & text_files)

files = get_file_list(MODE)
print(f"Processing {len(files)} files using mode '{MODE}'")

# Run prediction and save to CSV
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["sound_id", "class"])

    for filename in files:
        audio_emb = None
        text_emb = None
        sound_id = os.path.splitext(os.path.basename(filename))[0]
        try:
            if MODE in ("audio", "both"):
                audio_emb = np.load(os.path.join(AUDIO_FOLDER, filename), allow_pickle=True)
            if MODE in ("text", "both"):
                text_emb = np.load(os.path.join(TEXT_FOLDER, filename), allow_pickle=True)
        except Exception as e:
            print(f"[Error] Could not load '{filename}': {e}")
            continue

        pred_idx, pred_name = predict_class(audio_emb, text_emb)
        # print(f"{filename}: Predictoin -> {pred_name}")
        writer.writerow([sound_id, pred_name])

print(f"Predictions saved to: {OUTPUT_CSV}")

# Save metadata alongside predictions
metadata = {
    "timestamp": datetime.datetime.now().isoformat(),
    "model_config": {
        "model_name": model.__class__.__name__,
        "mode": MODE,
        "hidden_size": model.hidden_size,
        "num_classes": model.num_classes,
        "emb_size_audio": model.emb_size_audio,
        "emb_size_text": model.emb_size_text,
    },
    "model_weights": MODEL_WEIGHTS,
    "audio_folder": AUDIO_FOLDER if MODE in ("audio", "both") else None,
    "text_folder": TEXT_FOLDER if MODE in ("text", "both") else None,
}

metadata_path = OUTPUT_CSV.replace(".csv", "_metadata.json")
with open(metadata_path, "w") as metafile:
    json.dump(metadata, metafile, indent=4)

print(f"Metadata saved to: {metadata_path}")
