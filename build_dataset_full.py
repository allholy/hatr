import os
import pandas as pd
import json
from utils import get_subconfig

"""
The dataset is built from existing embedding (.npy) files identified by sound_id.
A sample is included only if both audio and text embeddings exist and a matching metadata 
entry is found.
"""

# Filepaths
metadata_csv = get_subconfig("metadata_csv")
audio_emb_folder = get_subconfig("audio_emb_folder")
text_emb_folder = get_subconfig("text_emb_folder")

output_path = get_subconfig("output_path")
os.makedirs(output_path, exist_ok=True)
processed_dataset_csv = os.path.join(output_path, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(output_path, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(output_path, get_subconfig("top_class_dict_json"))
top_class_subclass_dict_json = os.path.join(output_path, get_subconfig("top_class_subclass_dict_json"))

# Make files
df = pd.read_csv(metadata_csv)

class_dict = dict(zip(df['class'], df['class_idx']))
df['class_top'] = df['class'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else None)
class_top_dict = {class_top: idx for idx, class_top in enumerate(df['class_top'].unique())}
class_top_subclass_dict = {
    top_class: {subclass: idx for idx, subclass in enumerate(df[df['class_top'] == top_class]['class'].unique())}
    for top_class in df['class_top'].unique()
}

with open(class_dict_json, 'w') as f:
    json.dump(class_dict, f, indent=4)
print(f"Saved class dictionary to {class_dict_json}")

with open(top_class_dict_json, 'w') as f:
    json.dump(class_top_dict, f, indent=4)
print(f"Saved top class dictionary to {top_class_dict_json}")

with open(top_class_subclass_dict_json, 'w') as f:
    json.dump(class_top_subclass_dict, f, indent=4)
print(f"Saved top class subclass dictionary to {top_class_subclass_dict_json}")

records = []

for file in os.listdir(audio_emb_folder):
    if not file.endswith(".npy"):
        continue

    sound_id = os.path.splitext(file)[0]
    try:
        sound_id_int = int(sound_id)
    except ValueError:
        print(f"Skipping invalid file name: {file}")
        continue

    match = df[df['sound_id'] == sound_id_int]
    if match.empty:
        print(f"Warning: No match for sound_id {sound_id} in metadata.")
        continue

    text_file = f"{sound_id}.npy"
    text_filepath = os.path.join(text_emb_folder, text_file)
    if not os.path.isfile(text_filepath):
        print(f"Missing spectrogram for sound_id {sound_id}")
        continue

    class_top = match['class_top'].values[0]
    class_top_idx = class_top_dict.get(class_top, -1)
    class_name = match['class'].values[0]
    class_idx = int(match['class_idx'].values[0])
    relative_class_idx = class_top_subclass_dict[class_top].get(class_name, -1)

    audio_emb_filepath = os.path.abspath(os.path.join(audio_emb_folder, file))
    text_emb_filepath = os.path.abspath(text_filepath)

    records.append({
        "index": sound_id,
        "audio_emb_filepath": audio_emb_filepath,
        "text_emb_filepath": text_emb_filepath,
        "top_class": class_top,
        "top_class_idx": class_top_idx,
        "class": class_name,
        "class_idx": class_idx,
    })

db_df = pd.DataFrame(records)
db_df.to_csv(processed_dataset_csv, index=False)
print(f"Saved embedding dataframe to {processed_dataset_csv}")
