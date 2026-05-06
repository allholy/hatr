import json
import pandas as pd
import plotly.express as px
import os 
from utils import get_subconfig

base_coordinates = "model_output/t-contr_ce_penalty/both/fold_4/latent_visualization"

active_dataset_name = get_subconfig("active_dataset")
datasets_cfg = get_subconfig("datasets")
metadata = datasets_cfg[active_dataset_name]["metadata_csv"]
color_dict_path = get_subconfig("color_dict_path")

with open(os.path.join(base_coordinates, "pca_data.json")) as f:
    tsne_data = json.load(f)

latent_df = pd.DataFrame({
    "x": tsne_data["x"],
    "y": tsne_data["y"],
    "class": tsne_data["class"],
    "sound_id": tsne_data["sound_id"]
})

meta_df = pd.read_csv(metadata, usecols=["sound_id", "tags", "uploader", "title"], dtype={"sound_id": str})
df = latent_df.merge(meta_df, on="sound_id", how="left")

with open(color_dict_path) as f:
    class_color_dict = json.load(f)


def plot_by_class_interactive(df, class_color_dict, output_dir, filename="latent_by_class.html"):
    ordered_classes = list(class_color_dict.keys())
    df['class'] = pd.Categorical(df['class'], categories=ordered_classes, ordered=True)
    color_map = {cls: class_color_dict[cls] for cls in ordered_classes}

    if 'sound_id' in df.columns:
        df['sound_link'] = df['sound_id'].apply(lambda sid: f'<a href="https://freesound.org/s/{sid}" target="_blank">{sid}</a>')


    df["search_uploader"] = df["uploader"].fillna("").str.lower()
    df["search_id"] = df["sound_id"].fillna("").str.lower()
    df["search_tags"] = df["tags"].fillna("").str.lower()
    df["search_all"] = (
        df["search_uploader"] + " " +
        df["search_id"] + " " +
        df["search_tags"] + " " +
        df["sound_id"].fillna("").str.lower()
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        # hover_name="sound_id",
        hover_data={
            "sound_id": True,
            "tags": True,
            "uploader": True,
            "title": True,
            "x": False,
            "y": False
        },
        color_discrete_map=color_map,
        custom_data=["search_all", "search_uploader", "search_id", "search_tags"],
        render_mode="webgl"
    )

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

    html_str = fig.to_html(include_plotlyjs="cdn")

    search_js = """
    <div style="position:absolute; z-index:1000; margin:10px; background:white; padding:8px; border:1px solid #ccc;">
        <input type="text" id="searchBox" placeholder="Search..." 
        style="padding:6px; width:200px;">

        <select id="searchField" style="padding:6px;">
            <option value="0">All</option>
            <option value="1">Uploader</option>
            <option value="2">Sound ID</option>
            <option value="3">Tags</option>
        </select>
    </div>

    <div id="infoBox" style="
        position:absolute;
        right:20px;
        top:20px;
        width:320px;
        max-height:400px;
        overflow:auto;
        background:white;
        border:1px solid #ccc;
        padding:10px;
        z-index:1000;
        font-size:12px;
    ">
        Click a point to see metadata
    </div>

    <script>
    const plot = document.querySelectorAll('.plotly-graph-div')[0];

    let timeout = null;

    document.getElementById('searchBox').addEventListener('input', function(e) {
        clearTimeout(timeout);

        timeout = setTimeout(() => {
            const query = e.target.value.toLowerCase();
            const fieldIndex = parseInt(document.getElementById('searchField').value);

            plot.data.forEach((trace, i) => {
                const newOpacity = trace.customdata.map(row => {
                    const text = row[fieldIndex];
                    return text.includes(query) ? 1 : 0.05;
                });

                Plotly.restyle(plot, {'marker.opacity': [newOpacity]}, [i]);
            });
        }, 200);
    });

    plot.on('plotly_click', function(data) {
        const p = data.points[0];

        const uploader = p.customdata[1];
        const sound_id = p.customdata[2];
        const tags = p.customdata[3];
        const title = p.customdata[0];

        document.getElementById("infoBox").innerHTML = `
            <b>Sound ID:</b> ${sound_id}<br><br>
            <b>Uploader:</b> ${uploader}<br><br>
            <b>Title:</b> ${title}<br><br>
            <b>Tags:</b><br>${tags}<br><br>
            <a href="https://freesound.org/s/${sound_id}" target="_blank">Open on Freesound</a>
        `;
    });
    </script>
    """

    html_str = html_str.replace("</body>", search_js + "</body>")

    with open(output_path, "w") as f:
        f.write(html_str)

    print(f"Interactive plot saved as html to: {output_path}")


plot_by_class_interactive(df, class_color_dict, base_coordinates)