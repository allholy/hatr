from collections import defaultdict
import joblib
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import torch
import pandas as pd 
from tqdm import tqdm

from utils import get_subconfig

pastel_cmap = LinearSegmentedColormap.from_list("pastel_blue_pink", [(0,"#ffffff"), (0.4,"#c4e7f9"), (1,"#fe9492")])
plt.colormaps.register(name="pastel_blue_pink", cmap=pastel_cmap)

class BaseVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Rename using CSV mapping
        class_names_path = get_subconfig("class_names")  # class names path
        self.class_names = self._rename_class_names_from_csv(class_names_path)

    def _rename_class_names_from_csv(self, csv_path):
        """Replace class names using CSV mapping."""
        df = pd.read_csv(csv_path)
        name_mapping = dict(zip(df["class_key"], df["class_key_long"]))
        return name_mapping

class LatentSpaceVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.pca_path = os.path.join(output_dir, "pca_model.pkl")
        
    def load_color_dicts(self, class_color_path):
        with open(class_color_path, "r") as f:
            self.class_color_dict = json.load(f)
            
    def compute_tsne(self, latents):
        latents = np.array(latents)
        tsne = TSNE(n_components=2, random_state=1821, perplexity=min(30, len(latents) // 5))
        latents_2d = tsne.fit_transform(latents)
        return latents_2d
    
    def compute_pca(self, latents, save=True):
        latents = np.array(latents)
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        if save:
            joblib.dump(pca, self.pca_path)
        return latents_2d

    def visualize(self, latents, labels, top_labels, class_dict, top_class_dict, sound_ids, pca=None):
        inv_class_dict = {v: k for k, v in class_dict.items()}
        inv_top_class_dict = {v: k for k, v in top_class_dict.items()}
        
        labels = np.array(labels)
        top_labels = np.array(top_labels)

        latents_tsne = self.compute_tsne(latents)
        if pca is None:
            latents_pca = self.compute_pca(latents)
        else:  # dont recompute pca if we test with other data
            latents_2d = np.array(latents)
            latents_pca = pca.transform(latents_2d)

        self._plot_by_class(latents_tsne, labels, inv_class_dict, filename="latent_by_class.pdf")
        self._plot_by_top_class(latents_tsne, top_labels, inv_top_class_dict, filename="latent_by_top_class.pdf")
        self._save_tsne_data(latents_tsne, labels, top_labels, sound_ids, inv_class_dict, inv_top_class_dict)
        self._plot_by_class(latents_pca, labels, inv_class_dict, filename="latent_by_class_pca.pdf")
        self._plot_by_top_class(latents_pca, top_labels, inv_top_class_dict, filename="latent_by_top_class_pca.pdf")
        self._save_pca_data(latents_pca, labels, top_labels, sound_ids, inv_class_dict, inv_top_class_dict)

        return latents_tsne 
        
    def _plot_by_class(self, latents_2d, labels, inv_class_dict, filename="latent_by_class.png"):
        plt.figure(figsize=(12, 10))
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            original_name = inv_class_dict.get(label, f"Class {label}")
            class_name = self.class_names.get(original_name, original_name) if hasattr(self, "class_names") else original_name

            # Find color using the original name, not the mapped one
            color = self.class_color_dict.get(original_name, "#808080")  
            idxs = labels == label
            plt.scatter(
                latents_2d[idxs, 0],
                latents_2d[idxs, 1],
                color=color,
                label=class_name,
                alpha=0.9,
                edgecolors='w',
                linewidth=0.5,
                s=50
            )
        plt.title(f"{filename[:-4].replace('_', ' ').capitalize()}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def _plot_by_top_class(self, latents_2d, top_labels, inv_top_class_dict, filename="latent_by_top_class.png"):
        plt.figure(figsize=(12, 10))
        unique_top_labels = sorted(set(top_labels))
        for top_label in unique_top_labels:
            top_class_name = inv_top_class_dict.get(top_label, f"Top Class {top_label}")
            color = self.class_color_dict.get(top_class_name, "#606060") 
            idxs = top_labels == top_label
            plt.scatter(
                latents_2d[idxs, 0],
                latents_2d[idxs, 1],
                color=color,
                label=top_class_name,
                alpha=0.9,
                edgecolors='w',
                linewidth=0.5,
                s=50
            )
        plt.title(f"{filename[:-4].replace('_', ' ').capitalize()}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def _save_tsne_data(self, latents_2d, labels, top_labels, sound_ids, inv_class_dict, inv_top_class_dict):
        tsne_data = {
            'x': latents_2d[:, 0].tolist(),
            'y': latents_2d[:, 1].tolist(),
            'class': [inv_class_dict.get(l, f"Class {l}") for l in labels.tolist()],
            'top_class': [inv_top_class_dict.get(l, f"Top Class {l}") for l in top_labels.tolist()],
            'sound_id': [str(sid) for sid in sound_ids]
        }

        with open(os.path.join(self.output_dir, "tsne_data.json"), 'w') as f:
            json.dump(tsne_data, f)

    def _save_pca_data(self, latents_2d, labels, top_labels, sound_ids, inv_class_dict, inv_top_class_dict):
        pca_data = {
            'x': latents_2d[:, 0].tolist(),
            'y': latents_2d[:, 1].tolist(),
            'class': [inv_class_dict.get(l, f"Class {l}") for l in labels.tolist()],
            'top_class': [inv_top_class_dict.get(l, f"Top Class {l}") for l in top_labels.tolist()],
            'sound_id': [str(sid) for sid in sound_ids]
        }

        with open(os.path.join(self.output_dir, "pca_data.json"), 'w') as f:
            json.dump(pca_data, f)

    def visualize_ids(self, latents_2d, labels, top_labels, class_dict, top_class_dict, sound_ids):
        inv_class_dict = {v: k for k, v in class_dict.items()}
        inv_top_class_dict = {v: k for k, v in top_class_dict.items()}

        labels = np.array(labels)
        top_labels = np.array(top_labels)
        sound_ids = np.array(sound_ids)
        
        self._plot_ids_by_class(latents_2d, labels, sound_ids, inv_class_dict)
        self._plot_ids_by_top_class(latents_2d, top_labels, sound_ids, inv_top_class_dict)

    def _plot_ids_by_class(self, latents_2d, labels, sound_ids, inv_class_dict):
        plt.figure(figsize=(40, 35))
        
        unique_labels = sorted(set(labels))
        legend_handles = []
        
        for label in unique_labels:
            class_name = inv_class_dict.get(label, f"Class {label}")
            color = self.class_color_dict.get(class_name, "#808080")
            idxs = labels == label
            
            if np.any(idxs):
                scatter = plt.scatter(
                    latents_2d[idxs, 0],
                    latents_2d[idxs, 1],
                    color=color,
                    alpha=0.7,  
                    edgecolors='none',
                    s=30
                )
                legend_handles.append((scatter, class_name))
                
                for i, (x, y, sid) in enumerate(zip(latents_2d[idxs, 0], latents_2d[idxs, 1], sound_ids[idxs])):
                    plt.text(x, y, str(sid), color='black', fontsize=6, ha='center', va='center')
        
        plt.title("t-SNE visualization of latent space with IDs (By Class)")
        
        if legend_handles:
            plt.subplots_adjust(right=0.75)
            plt.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles], 
                      loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.savefig(os.path.join(self.output_dir, "latent_ids_by_class.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_ids_by_top_class(self, latents_2d, top_labels, sound_ids, inv_top_class_dict):
        plt.figure(figsize=(40, 35))
        
        unique_top_labels = sorted(set(top_labels))
        legend_handles = []
        
        for top_label in unique_top_labels:
            top_class_name = inv_top_class_dict.get(top_label, f"Top class {top_label}")
            color = self.class_color_dict.get(top_class_name, "#606060")
            idxs = top_labels == top_label
            
            if np.any(idxs):
                scatter = plt.scatter(
                    latents_2d[idxs, 0],
                    latents_2d[idxs, 1],
                    color=color,
                    alpha=0.7,  
                    edgecolors='none',
                    s=30
                )
                legend_handles.append((scatter, top_class_name))
                
                for i, (x, y, sid) in enumerate(zip(latents_2d[idxs, 0], latents_2d[idxs, 1], sound_ids[idxs])):
                    plt.text(x, y, str(sid), color='black', fontsize=6, ha='center', va='center')
        
        plt.title("t-SNE Visualization of Latent Space with IDs (By Top-Level Class)")
        
        if legend_handles:
            plt.subplots_adjust(right=0.75)
            plt.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles], 
                      loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.savefig(os.path.join(self.output_dir, "latent_ids_by_top_class.png"), dpi=300, bbox_inches='tight')
        plt.close()


class ConfusionMatrixVisualizer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)
    
    def compute_and_visualize(self, predictions, true_labels, true_top_labels, top_class_predictions, class_dict, top_class_dict):
        inv_class_dict = {v: k for k, v in class_dict.items()}
        inv_top_class_dict = {v: k for k, v in top_class_dict.items()}
        
        self._visualize_class_confusion(predictions, true_labels, inv_class_dict)
        self._visualize_top_class_confusion(top_class_predictions, true_top_labels, inv_top_class_dict)
    
    def _visualize_class_confusion(self, predictions, true_labels, inv_class_dict):
        cm_classes_abs = confusion_matrix(true_labels, predictions)
        cm_classes = cm_classes_abs.astype('float') / cm_classes_abs.sum(axis=1)[:, np.newaxis] * 100
        cm_classes = np.nan_to_num(cm_classes)
        if hasattr(self, "class_names") and self.class_names is not None:
            class_labels = [self.class_names.get(inv_class_dict[i], f"Class {i}") for i in range(len(inv_class_dict))]
        else:
            class_labels = [inv_class_dict.get(i, f"Class {i}") for i in range(len(inv_class_dict))]

        def plot_heatmap(matrix, annot_matrix, filename):
            plt.figure(figsize=(16, 14))
            sns.heatmap(
                matrix,
                annot=annot_matrix,
                fmt="",
                cmap="pastel_blue_pink", 
                xticklabels=class_labels,
                yticklabels=class_labels,
                annot_kws={'size': 13},
            )

            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(True) 
                spine.set_color("lightgray")
                spine.set_linewidth(0.5)

            plt.xlabel("Predicted class", fontsize=16)
            plt.ylabel("Ground-truth class", fontsize=16)
            plt.xticks(rotation=45, fontsize=14)
            plt.yticks(rotation=0, fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=400, bbox_inches='tight')
            plt.close()

        # Prepare annotation matrices (replace zeros with empty string)
        annot_percent = np.where(cm_classes == 0, "", np.vectorize(lambda x: f"{x:.1f}".rstrip("0").rstrip("."))(cm_classes))
        annot_abs = np.where(cm_classes_abs == 0, "", cm_classes_abs)

        # Plot percentage and absolute counts heatmaps
        plot_heatmap(cm_classes, annot_percent, "confusion_matrix_classes_percent.png")
        plot_heatmap(cm_classes_abs, annot_abs, "confusion_matrix_classes_absolute.png")
            
    def _visualize_top_class_confusion(self, top_class_predictions, true_top_labels, inv_top_class_dict):
        cm_top_classes_abs = confusion_matrix(true_top_labels, top_class_predictions)
        cm_top_classes = cm_top_classes_abs.astype('float') / cm_top_classes_abs.sum(axis=1)[:, np.newaxis] * 100
        cm_top_classes = np.nan_to_num(cm_top_classes)
        if hasattr(self, "class_names") and self.class_names is not None:
            top_class_labels = [self.class_names.get(inv_top_class_dict[i], f"Top Class {i}") for i in range(len(inv_top_class_dict))]
        else:
            top_class_labels = [inv_top_class_dict.get(i, f"Top Class {i}") for i in range(len(inv_top_class_dict))]        

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm_top_classes, 
            annot=True, 
            fmt=".1f", 
            cmap="Blues",
            xticklabels=top_class_labels,
            yticklabels=top_class_labels
        )
        # plt.title("Confusion Matrix - Top Classes (Percentage by Row)")
        plt.xlabel("Predicted class", fontsize=16)
        plt.ylabel("Ground-truth class", fontsize=16)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix_top_classes_percent.png"), dpi=400)
        plt.close()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm_top_classes_abs, 
            annot=True, 
            fmt="d",  
            cmap="Blues",
            xticklabels=top_class_labels,
            yticklabels=top_class_labels
        )
        # plt.title("Confusion Matrix - Top Classes (Absolute Counts)")
        plt.xlabel("Predicted Top Class")
        plt.ylabel("True Top Class")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix_top_classes_absolute.png"), dpi=300)
        plt.close()


class ClassAccuracyAnalyzer(BaseVisualizer):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def analyze_accuracy_per_class(self, predictions, true_labels, class_dict):
        inv_class_dict = {v: k for k, v in class_dict.items()}
        
        class_counts = {}
        class_correct = {}
        
        for true_label, pred_label in zip(true_labels, predictions):
            true_class = int(true_label)
            pred_class = int(pred_label)
            
            if true_class not in class_counts:
                class_counts[true_class] = 0
                class_correct[true_class] = 0
                
            class_counts[true_class] += 1
            if true_class == pred_class:
                class_correct[true_class] += 1
        
        class_accuracy = {}
        for class_idx in class_counts:
            class_accuracy[class_idx] = (class_correct[class_idx] / class_counts[class_idx]) * 100
        sorted_classes = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)
        
        class_names = []
        accuracies = []
        sample_counts = []
        
        for class_idx, acc in sorted_classes:
            class_name = inv_class_dict.get(class_idx, f"Class {class_idx}")
            class_names.append(class_name)
            accuracies.append(acc)
            sample_counts.append(class_counts[class_idx])
        
        self._save_accuracy_data(class_names, accuracies, sample_counts)
        self._plot_class_accuracy(class_names, accuracies, sample_counts)
    
    def _save_accuracy_data(self, class_names, accuracies, sample_counts):
        class_data = {
            "class_names": class_names,
            "accuracies": accuracies,
            "sample_counts": sample_counts
        }
        
        with open(os.path.join(self.output_dir, "per_class_accuracy.json"), 'w') as f:
            json.dump(class_data, f, indent=2)
    
    def _plot_class_accuracy(self, class_names, accuracies, sample_counts):
        max_classes_to_show = 50
        if len(class_names) > max_classes_to_show:
            class_names = class_names[:max_classes_to_show]
            accuracies = accuracies[:max_classes_to_show]
            sample_counts = sample_counts[:max_classes_to_show]
        
        plt.figure(figsize=(15, 6))
        
        vmin = min(sample_counts)
        vmax = max(sample_counts)
        cmap = plt.get_cmap("pastel_blue_pink")
        shift_value = 0.1
        shifted = shift_value + (1-shift_value)*(np.array(sample_counts) - vmin) / (vmax - vmin)
        mapped_colors = plt.get_cmap("pastel_blue_pink")(shifted)

        bars = plt.bar(range(len(class_names)), accuracies, color=mapped_colors)

        if hasattr(self, "class_names") and self.class_names is not None:
            display_names = [self.class_names.get(cn, cn) for cn in class_names]
        else:
            display_names = class_names

        plt.xlabel("Class", fontsize=15)
        plt.ylabel("Accuracy (%)", fontsize=15)
        plt.xticks(range(len(display_names)), display_names, rotation=45, fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        # plt.title('Per-Class Accuracy')
        plt.ylim(0, 110)  
        
        for i, (bar, count) in enumerate(zip(bars, sample_counts)):
            plt.text(i, bar.get_height() + 2, f"n={count}", ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            # spine.set_visible(False)
            spine.set_color("lightgray")
            spine.set_linewidth(0.5)

        sm = plt.cm.ScalarMappable(cmap=ListedColormap(cmap(np.linspace(shift_value,1,256))), norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.9, aspect=12)
        cbar.outline.set_visible(False)
        cbar.set_label('Sample count', fontsize=14)  

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "per_class_accuracy.png"), dpi=400, bbox_inches='tight')
        plt.close()


class AttentionVisualizer:
    def __init__(self, model, subset=None, save_path="./attention", device="cpu"):
        self.model = model
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.device = device
        self.subset = subset

    def single_sample_attention(self, sample):
        self.model.eval()
        with torch.no_grad():
            audio = sample['audio_embedding'].unsqueeze(0).to(self.device)
            text = sample['text_embedding'].unsqueeze(0).to(self.device)
            label = sample['class']
            sound_id = sample['sound_id']

            _, _, attn = self.model(audio, text)
            attn = attn.squeeze().cpu().numpy()  # shape [2]

            # Single horizontal stacked bar
            plt.figure(figsize=(6, 1.5))
            plt.barh([0], attn[0], color='skyblue', label='Audio', left=0)
            plt.barh([0], attn[1], color='salmon', label='Text', left=attn[0])
            plt.xlim(0, 1)
            plt.yticks([])  # hide y-axis
            plt.xlabel("Attention distribution")
            plt.title(f"Attention for sound {sound_id} in {label}")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"attention_sample_{sound_id}_{label}.png"))
            plt.close()

    def avg_attention_by_class(self):
        self.model.eval()
        
        attn_by_class = defaultdict(lambda: {"audio": [], "text": []})
        
        with torch.no_grad():
            for sample in self.subset:
                audio = sample['audio_embedding'].unsqueeze(0).to(self.device)
                text = sample['text_embedding'].unsqueeze(0).to(self.device)
                label = sample['class']                
                
                _, _, attn = self.model(audio, text)
                attn = attn.squeeze().cpu().numpy()
                
                attn_by_class[label]["audio"].append(attn[0])
                attn_by_class[label]["text"].append(attn[1])
        
        # Compute average attention per class
        classes = list(attn_by_class.keys())
        avg_audio = [np.mean(attn_by_class[c]["audio"]) for c in classes]
        avg_text = [np.mean(attn_by_class[c]["text"]) for c in classes]
        
        attn_matrix = np.array([avg_audio, avg_text])
        
        plt.figure(figsize=(12, 3))
        sns.heatmap(attn_matrix, annot=True, cmap="YlOrRd", yticklabels=['Audio','Text'], xticklabels=classes)
        plt.title("Average attention by class")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "attention_by_class.png"))
        plt.close()     


class ActivationVisualizer:
    def __init__(self, model, save_path, subset, features_to_plot=None, class_dict=None, top_class_dict=None):
        self.model = model.eval()
        self.save_path = save_path
        self.subset = subset
        self.features_to_plot = features_to_plot
        self.class_dict = class_dict or {}
        self.top_class_dict = top_class_dict or {}

        os.makedirs(save_path, exist_ok=True)

    def get_class_name(self, class_idx):
        return next((k for k, v in self.class_dict.items() if v == class_idx), f"Class {class_idx}")

    def get_top_class_name(self, top_class_idx):
        return next((k for k, v in self.top_class_dict.items() if v == top_class_idx), f"TopClass {top_class_idx}")

    def get_saliency(self, sample, target_class_idx=None):
        device = next(self.model.parameters()).device
        inputs = {}
        grads = {}

        name_map = [("audio_embedding", "audio_emb"),("text_embedding", "text_emb")]

        for dataset_key, model_key in name_map:
            if dataset_key in sample and dataset_key in self.features_to_plot:
                x = sample[dataset_key].unsqueeze(0).to(device).requires_grad_(True)
                inputs[model_key] = x
                grads[dataset_key] = {'input': x.detach().cpu().numpy()}
            else:
                inputs[model_key] = None
                grads[dataset_key] = {'input': None}

        _, output, _ = self.model(**inputs)
        pred_class_idx = torch.argmax(output, dim=1).item()
        target_class = target_class_idx if target_class_idx is not None else pred_class_idx
        output[0, target_class].backward()

        for dataset_key, model_key in name_map:
            if inputs[model_key] is not None:
                grads[dataset_key]['gradient'] = inputs[model_key].grad.detach().cpu().numpy()

        return grads, pred_class_idx

    def compute_saliency(self, grads):
        maps = {}
        for k, v in grads.items():
            sal = np.abs(v['gradient'] * v['input'])
            if k == 'spectrograms':
                sal = gaussian_filter(sal.squeeze(0), sigma=2)
            else:
                sal = gaussian_filter(sal.squeeze(0).reshape(-1), sigma=2).reshape(1, -1)
            maps[k] = sal / (np.max(sal) + 1e-9)
        return maps

    def plot_features(self, name, feature, saliency, title, file_name):
        if name == 'spectrograms':
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            axs[0].imshow(feature.squeeze(0), aspect='auto', cmap='viridis', origin='lower')
            axs[1].imshow(saliency, aspect='auto', cmap='hot', origin='lower')
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
            axs[0].imshow(feature.reshape(1, -1), aspect='auto', cmap='Blues')
            axs[1].imshow(saliency.reshape(1, -1), aspect='auto', cmap='hot')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_anchor('C')  

        fig.suptitle(title, fontsize=14)
        plt.savefig(os.path.join(self.save_path, file_name), bbox_inches='tight', dpi=300)
        plt.close()

    def single_sample_activation(self, sample, target_class_idx=None):
        sound_id = sample.get('sound_id', None)
        class_idx = sample.get('class_idx', None)

        class_name = self.get_class_name(class_idx) if class_idx is not None else "Unknown"

        grads, pred_class_idx = self.get_saliency(sample, target_class_idx)
        pred_class_name = self.get_class_name(pred_class_idx)

        maps = self.compute_saliency(grads)

        for feat in maps:
            title = f"Sample - {feat} | GT: {class_name}, Pred: {pred_class_name}, ID: {sound_id}"
            fname = f"sample_{feat}_{class_name}_{pred_class_name}.png"
            self.plot_features(feat, grads[feat]['input'], maps[feat], title, fname)

    def avg_class_activation(self, class_idx=None):
        samples = [s for s in self.subset if s['class_idx'].item() == class_idx] if class_idx is not None else self.subset
        grouped = defaultdict(list)
        for s in samples:
            grouped[s['class_idx'].item()].append(s)

        for cls, group in grouped.items():
            grads_accum = {}
            for f in self.features_to_plot:
                x = group[0][f[:-1] if f.endswith('s') else f]
                grads_accum[f] = {'input': np.zeros_like(x.unsqueeze(0).numpy()), 'gradient': np.zeros_like(x.unsqueeze(0).numpy())}

            for sample in tqdm(group, desc=f"Class {cls}"):
                g, _ = self.get_saliency(sample, cls)
                for f in g:
                    grads_accum[f]['input'] += g[f]['input']
                    grads_accum[f]['gradient'] += g[f]['gradient']

            for f in grads_accum:
                grads_accum[f]['input'] /= len(group)
                grads_accum[f]['gradient'] /= len(group)

            # Compute mean saliency for all features
            feature_avgs = {}
            for feat, vals in grads_accum.items():
                if 'gradient' in vals and 'input' in vals and vals['gradient'] is not None:
                    feature_avgs[feat] = np.abs(vals['gradient'] * vals['input']).mean()
            avg_file = os.path.join(self.save_path, "all_classes_avg_saliency.csv")
            mode = 'a' if os.path.exists(avg_file) else 'w'
            with open(avg_file, mode) as f:
                if mode == 'w':
                    f.write("class," + ",".join(feature_avgs.keys()) + "\n")
                class_name = self.get_class_name(cls)
                f.write(f"{class_name}," + ",".join(f"{v:.4f}" for v in feature_avgs.values()) + "\n")

            # Salience maps 
            maps = self.compute_saliency(grads_accum)
            for feat in maps:
                class_name = self.get_class_name(cls)
                title = f"Avg - {feat} | Class {class_name}"
                fname = f"avg_class_{feat}_{class_name}.png"
                self.plot_features(feat, grads_accum[feat]['input'], maps[feat], title, fname)

    def avg_top_class_activation(self, top_class_idx=None):
        samples = [s for s in self.subset if s['top_class_idx'].item() == top_class_idx] if top_class_idx is not None else self.subset
        grouped = defaultdict(list)
        for s in samples:
            grouped[s['top_class_idx'].item()].append(s)

        for cls, group in grouped.items():
            grads_accum = {}
            for f in self.features_to_plot:
                x = group[0][f[:-1] if f.endswith('s') else f]
                grads_accum[f] = {'input': np.zeros_like(x.unsqueeze(0).numpy()), 'gradient': np.zeros_like(x.unsqueeze(0).numpy())}

            for sample in tqdm(group, desc=f"Top Class {cls}"):
                g, _ = self.get_saliency(sample, cls)
                for f in g:
                    grads_accum[f]['input'] += g[f]['input']
                    grads_accum[f]['gradient'] += g[f]['gradient']

            for f in grads_accum:
                grads_accum[f]['input'] /= len(group)
                grads_accum[f]['gradient'] /= len(group)

            maps = self.compute_saliency(grads_accum)
            for feat in maps:
                top_class_name = self.get_top_class_name(cls)
                title = f"Avg - {feat} | Top Class {top_class_name}"
                fname = f"avg_top_class_{feat}_{top_class_name}.png"
                self.plot_features(feat, grads_accum[feat]['input'], maps[feat], title, fname)
