import json
import pandas as pd
import plotly.express as px
import os 

base_coordinates = "model_output/t-contr_ce_penalty/both/fold_4/latent_visualization"
metadata = "/media/anastp/DATA/datasets/BSD10k/BSD10k-v1.1/metadata/BSD10k_metadata.csv"
color_dict_path = "data/class_color_dict.json"

with open(os.path.join(base_coordinates, "pca_data.json")) as f:
    tsne_data = json.load(f)

latent_df = pd.DataFrame({
    "x": tsne_data["x"],
    "y": tsne_data["y"],
    "class": tsne_data["class"],
    # "top_class": tsne_data["top_class"],
    "sound_id": tsne_data["sound_id"]
})

meta_df = pd.read_csv(metadata, usecols=["sound_id", "tags", "uploader", "title"], dtype={"sound_id": str} )
df = latent_df.merge(meta_df, on="sound_id", how="left")
# print(df.columns)
# print(df[['class', 'sound_id']].head(3))

with open(color_dict_path) as f:
    class_color_dict = json.load(f)

def plot_by_class_interactive(df, class_color_dict, output_dir, filename="latent_by_class.html"):
    ordered_classes = list(class_color_dict.keys())
    df['class'] = pd.Categorical(df['class'], categories=ordered_classes, ordered=True)
    color_map = {cls: class_color_dict[cls] for cls in ordered_classes}

    # Create a new column with clickable links
    if 'sound_id' in df.columns:
        df['sound_link'] = df['sound_id'].apply(lambda sid: f'<a href="https://freesound.org/s/{sid}" target="_blank">{sid}</a>')

    # Use 'sound_link' instead of 'sound_id' in hover
    hover_cols = [c for c in ["sound_link", "tags", "uploader", "title"] if c in df.columns]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        hover_data=hover_cols,
        color_discrete_map=color_map
    )

    fig.update_traces(hovertemplate=None)  # Enable full HTML rendering
    fig.update_layout(
        title="Latent space visualization",
        width=900,
        height=700,
        legend_title="Original class"
    )

    for i, cls in enumerate(ordered_classes):
        for trace in fig.data:
            if trace.name == cls:
                trace.legendrank = i

    output_path = os.path.join(output_dir, filename)
    fig.write_html(output_path)
    print(f"Interactive plot saved as html to: {output_path}")



plot_by_class_interactive(df, class_color_dict, base_coordinates)
# file:///media/anastp/DATA/doctora-code/classifiers/multimodal_hierarchical/model_output/t-contr_ce_penalty/both/fold_0/latent_visualization/latent_by_class.html
