import os 
import numpy as np
import torch
from models import BaseClassifier

# Configuration
AUDIO_FOLDER = "/media/anastp/DATA/datasets/BSD10k-v1.1/features/CLAP_audio_embeddings"
TEXT_FOLDER = "/media/anastp/DATA/datasets/BSD10k-v1.1/features/CLAP_text_embeddings"
MODE = "both"  # "audio", "text", or "both"
MODEL_WEIGHTS = f"model_output/t-contr_ce_penalty/{MODE}/fold_0/best_model.pth"
OUTPUT_FOLDER= "/media/anastp/DATA/datasets/BSD10k-v1.1/features/bst_multimodal_embeddings"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BaseClassifier(
    hidden_size=128,
    num_classes=23,
    emb_size_audio=512,
    emb_size_text=512,
    mode=MODE
)
model.load_state_dict(torch.load(MODEL_WEIGHTS))
model.to(device).eval()

def list_files(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".npy")])

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
for filename in files:
    audio_tensor = text_tensor = None
    
    if MODE in ("audio", "both"):
        audio_emb = np.load(os.path.join(AUDIO_FOLDER, filename)).astype(np.float32)
        audio_tensor = torch.tensor(audio_emb).unsqueeze(0).to(device)
        
    if MODE in ("text", "both"):
        text_emb = np.load(os.path.join(TEXT_FOLDER, filename)).astype(np.float32)
        text_tensor = torch.tensor(text_emb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        z, logits = model(audio_emb=audio_tensor, text_emb=text_tensor)
        
    embedding_np = z.cpu().numpy()[0]
    
    np.save(os.path.join(OUTPUT_FOLDER, filename), embedding_np)

print("Extracted embeddings in shape:", embedding_np.shape)

