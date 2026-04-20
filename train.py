from collections import defaultdict
import collections.abc
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils import get_subconfig
from models import BaseClassifier
from dataset_utils import HATRDataset
from losses import CrossEntropyLoss, TopClassLoss, HierarchicalCrossEntropyLoss, CosineTripletLoss_Top, HierarchicalContrastiveLoss
from visualizations import LatentSpaceVisualizer, ConfusionMatrixVisualizer, ClassAccuracyAnalyzer, AttentionVisualizer, ActivationVisualizer

from datetime import datetime
current_time = datetime.now().strftime("%y%m%d-%H:%M")


# Paths
dataset_path = get_subconfig("metadata_csv")
color_dict_path = get_subconfig("color_dict_path")
top_color_dict_path = get_subconfig("top_color_dict_path")

data_dir = get_subconfig("output_path")
prepared_dataset_path = os.path.join(data_dir, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))
subclass_json = os.path.join(data_dir, get_subconfig("top_class_subclass_dict_json"))


def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)

def make_serializable(obj, decimals=6):
    """Recursively convert tensors, numpy arrays, and numbers to JSON-serializable types with rounding."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        return make_serializable(obj, decimals)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return round(float(obj), decimals)
        else:
            return [make_serializable(x, decimals) for x in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return {k: make_serializable(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return [make_serializable(x, decimals) for x in obj]
    else:
        return obj
    
def train_model(model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, classification_weight=1.0, contrastive_weight=0.0, top_class_weight=0.0,
                hierarchical_ce_weight=0.0, hierarchical_triplet_weight=0.0,
                classification_criterion=None, contrastive_criterion=None, top_class_criterion=None, hierarchical_ce_criterion=None, hierarchical_triplet_criterion=None,
                output_dir='model_output', scheduler_type='plateau', patience=10, early_stopping_factor=5):
    
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, verbose=True)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None
    
    best_accuracy = 0.0
    epochs_without_improvement = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        losses = defaultdict(float)

        attn_audio_epoch = []
        attn_text_epoch = []

        for data in train_loader:
            class_labels = data['class_idx'].to(device)
            top_class_labels = data['top_class_idx'].to(device)
            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)
            
            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)

            optimizer.zero_grad()
            
            z, class_logit, attn_scores = model(audio_emb, text_emb)
            
            # collect batch attention once per batch
            if attn_scores is not None:
                attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
                attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

            total_loss = 0.0
            
            if classification_criterion:
                cls_loss = classification_criterion(class_logit, class_labels)
                losses['cls'] += cls_loss.item()
                total_loss += classification_weight * cls_loss

            if top_class_criterion and top_class_weight > 0:
                top_class_loss = top_class_criterion(class_logit, class_labels)
                losses['top'] += top_class_loss.item()
                total_loss += top_class_weight * top_class_loss

            if contrastive_criterion and contrastive_weight > 0 and z is not None:
                contra_loss = contrastive_criterion(z, class_labels)
                losses['contra'] += contra_loss.item()
                total_loss += contrastive_weight * contra_loss

            if hierarchical_ce_criterion and hierarchical_ce_weight > 0:
                top_class_logit = torch.zeros(class_logit.size(0), len(top_class_dict), device=device)
                for i, class_idx in enumerate(class_labels):
                    if class_idx.item() in class_to_top_class:
                        top_idx = class_to_top_class[class_idx.item()]
                        top_class_logit[i, top_idx] = class_logit[i, class_idx]
                
                hier_ce_loss = hierarchical_ce_criterion(class_logit, top_class_logit, class_labels)
                losses['hier_ce'] += hier_ce_loss.item()
                total_loss += hierarchical_ce_weight * hier_ce_loss

            if hierarchical_triplet_criterion and hierarchical_triplet_weight > 0 and z is not None:
                batch_size = z.size(0)
                if batch_size >= 3:
                    anchor_idx = torch.arange(0, batch_size, 3).long()
                    positive_idx = torch.arange(1, batch_size, 3).long()
                    negative_idx = torch.arange(2, batch_size, 3).long()
                    
                    min_len = min(len(anchor_idx), len(positive_idx), len(negative_idx))
                    anchor_idx = anchor_idx[:min_len]
                    positive_idx = positive_idx[:min_len]
                    negative_idx = negative_idx[:min_len]
                    
                    if len(anchor_idx) > 0:
                        anchor = z[anchor_idx]
                        positive = z[positive_idx]
                        negative = z[negative_idx]
                        label_anchor = class_labels[anchor_idx]
                        label_negative = class_labels[negative_idx]
                        
                        hier_triplet_loss = hierarchical_triplet_criterion(anchor, positive, negative, label_anchor, label_negative)
                        losses['hier_triplet'] += hier_triplet_loss.item()
                        total_loss += hierarchical_triplet_weight * hier_triplet_loss

            total_loss.backward()
            optimizer.step()
            losses['total'] += total_loss.item()

        # per-epoch attention summary
        if attn_audio_epoch:
            attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
            attn_text_epoch = torch.cat(attn_text_epoch, dim=0)
            history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
            history["attention_text"].append(attn_text_epoch.mean(0).numpy())

        num_batches = len(train_loader)
        for k in losses:
            history[f'train_{k}_loss'].append(losses[k] / num_batches)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                labels = data['class_idx'].to(device)
                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)
                
                if audio_emb is not None:
                    audio_emb = audio_emb.to(device)
                if text_emb is not None:
                    text_emb = text_emb.to(device)

                _, class_logit, _ = model(audio_emb, text_emb)
                    
                _, predicted = torch.max(class_logit.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        history['val_accuracy'].append(val_accuracy)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Val Acc: {val_accuracy:.2f}%")
        for k in losses:
            if losses[k] > 0:
                print(f"  {k.capitalize()} Loss: {losses[k] / num_batches:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))  # model parameters
            torch.save(model, os.path.join(output_dir, "best_model.pt"))  # whole model 
            print(f"New best model saved: {best_accuracy:.2f}%")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience * early_stopping_factor:
                print("Early stopping triggered.")
                break

    return best_accuracy, history, model


def test_model(model, model_path, data_loader, device, class_to_top_class, output_dir, model_name, fold_id,
               class_dict=None, top_class_dict=None):

    # -------------------- Setup --------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # -------------------- Helpers --------------------
    def collect_outputs():
        all_latents, all_labels, all_top_labels = [], [], []
        all_predictions, all_top_class_predictions = [], []
        all_class_probs, all_sound_ids = [], []
        misclassified = {"sound_id": [], "true": [], "pred": [], "conf": []}

        correct = top1_top = top2_correct = top2_top = total = 0

        with torch.no_grad():
            for data in data_loader:
                labels = data['class_idx'].to(device)
                top_labels = data['top_class_idx'].to(device)
                sound_ids = data['sound_id']

                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)
                if audio_emb is not None: audio_emb = audio_emb.to(device)
                if text_emb is not None: text_emb = text_emb.to(device)

                z, class_logits, _ = model(audio_emb, text_emb)
                probs = torch.softmax(class_logits, dim=1)

                top2_vals, top2_preds = torch.topk(probs, 2, dim=1)
                top1, top2 = top2_preds[:, 0], top2_preds[:, 1]
                max_probs = top2_vals[:, 0]

                all_class_probs.extend(probs.cpu().numpy())
                total += labels.size(0)
                correct += (top1 == labels).sum().item()
                top2_correct += ((top1 == labels) | (top2 == labels)).sum().item()

                # Top-class accuracy
                for i in range(labels.size(0)):
                    true_cls = labels[i].item()
                    pred1, pred2 = top1[i].item(), top2[i].item()
                    if true_cls in class_to_top_class and pred1 in class_to_top_class:
                        true_top, pred_top1 = class_to_top_class[true_cls], class_to_top_class[pred1]
                        if true_top == pred_top1:
                            top1_top += 1
                        if pred2 in class_to_top_class:
                            pred_top2 = class_to_top_class[pred2]
                            if true_top in (pred_top1, pred_top2):
                                top2_top += 1

                # Misclassifications
                for i in range(labels.size(0)):
                    if top1[i] != labels[i]:
                        misclassified["sound_id"].append(int(sound_ids[i]))
                        misclassified["true"].append(int(labels[i]))
                        misclassified["pred"].append(int(top1[i]))
                        misclassified["conf"].append(float(max_probs[i]))

                # Collect all data
                if z is not None: 
                    all_latents.append(z.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_top_labels.append(top_labels.cpu().numpy())
                all_predictions.append(top1.cpu().numpy())
                all_sound_ids.extend(sound_ids.cpu().numpy())

                pred_top_classes = [class_to_top_class.get(p, -1) for p in top1.cpu().numpy()]
                all_top_class_predictions.append(np.array(pred_top_classes))

        return (all_latents, all_labels, all_top_labels, all_predictions, all_top_class_predictions, all_class_probs, all_sound_ids,
                misclassified, correct, top1_top, top2_correct, top2_top, total)

    def compute_metrics(correct, top1_top, top2_correct, top2_top, total):
        return {
            "accuracy": 100 * correct / total,
            "true_top_acc": 100 * top1_top / total,
            "top2_acc": 100 * top2_correct / total,
            "top2_top_acc": 100 * top2_top / total,
        }

    def save_misclassified(misclassified, id_to_class):
        df = pd.DataFrame({
            "sound_id": misclassified["sound_id"],
            "true_label": [id_to_class.get(lbl, str(lbl)) for lbl in misclassified["true"]],
            "pred_label": [id_to_class.get(lbl, str(lbl)) for lbl in misclassified["pred"]],
            "confidence": misclassified["conf"]
        })
        path = os.path.join(output_dir, "misclassifications", "misclassified_samples.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    def run_visualizations(all_latents, all_labels, all_top_labels,
                           all_predictions, all_top_class_predictions,
                           all_sound_ids):
        latents_vis = LatentSpaceVisualizer(os.path.join(output_dir, "latent_visualization"))
        cm_vis = ConfusionMatrixVisualizer(os.path.join(output_dir, "confusion_matrix_visualization"))
        acc_analyzer = ClassAccuracyAnalyzer(os.path.join(output_dir, "class_accuracy_analysis"))

        if os.path.exists(color_dict_path):
            latents_vis.load_color_dicts(color_dict_path)

        if len(all_latents) > 0:
            latents_2d = latents_vis.compute_tsne(all_latents)
            latents_vis.visualize(latents_2d, all_labels, all_top_labels, class_dict, top_class_dict, all_sound_ids)

        cm_vis.compute_and_visualize(
            all_predictions, all_labels, all_top_labels, all_top_class_predictions,
            class_dict, top_class_dict
        )
        acc_analyzer.analyze_accuracy_per_class(all_predictions, all_labels, class_dict)

    # -------------------- Execution --------------------
    (all_latents, all_labels, all_top_labels, all_predictions, all_top_class_predictions, all_class_probs, all_sound_ids, 
     misclassified, correct, top1_top, top2_correct, top2_top, total) = collect_outputs()

    # Flatten arrays
    if len(all_latents) > 0: all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_top_labels = np.concatenate(all_top_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_top_class_predictions = np.concatenate(all_top_class_predictions, axis=0) if len(all_top_class_predictions) > 0 else np.array([])

    id_to_class = {v: k for k, v in class_dict.items()} if class_dict else {}
    metrics = compute_metrics(correct, top1_top, top2_correct, top2_top, total)

    print(f"[{model_name} | Fold {fold_id}] Accuracy: {metrics['accuracy']:.2f}%")
    print(f"[{model_name} | Fold {fold_id}] True Top Class Acc: {metrics['true_top_acc']:.2f}%")
    print(f"[{model_name} | Fold {fold_id}] Acc incl. 2nd: {metrics['top2_acc']:.2f}%")
    print(f"[{model_name} | Fold {fold_id}] Top Class Acc incl. 2nd: {metrics['top2_top_acc']:.2f}%")

    save_misclassified(misclassified, id_to_class)
    run_visualizations(all_latents, all_labels, all_top_labels,
                       all_predictions, all_top_class_predictions,
                       all_sound_ids)

    return (metrics["accuracy"], metrics["true_top_acc"],
            metrics["top2_acc"], metrics["top2_top_acc"])



def build_class_to_top_class_mapping(class_dict, top_class_dict):
    class_to_top_class = {}

    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_id in top_class_dict.items():
            if class_name.startswith(top_class_name):
                class_to_top_class[class_id] = top_class_id
                break

    return class_to_top_class


def build_subclass_to_topclass_tensor(class_dict, top_class_dict, device):
    num_classes = len(class_dict)
    subclass_to_topclass = torch.zeros(num_classes, dtype=torch.long, device=device)
    
    for class_name, class_id in class_dict.items():
        for top_class_name, top_class_id in top_class_dict.items():
            if class_name.startswith(top_class_name):
                subclass_to_topclass[class_id] = top_class_id
                break
    
    return subclass_to_topclass


if __name__ == "__main__":
    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)

    random_seed = 1821
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    train_from_scratch = True
    activation_visualization = True
    attention_visualization = True
        
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    classification_weight = 1
    scheduler_type = 'step'
    patience = 5
    early_stopping_factor = 3
    k_folds = 5

    full_df = pd.read_csv(prepared_dataset_path)
    full_df_confidence = pd.read_csv(dataset_path)
    high_conf_df = full_df_confidence[full_df_confidence['confidence'] >= 0]
    full_df = full_df[full_df['index'].isin(high_conf_df['sound_id'])]

    datasets = {
        'full': {'df': full_df}
    }

    modes = ['both']
    hierarchies = {
        # 'h-contr_ce_penalty': {
        #     'top_class_weight': 1, 
        #     'contrastive_weight': 1,
        #     'hierarchical_ce_weight': 0,
        #     'hierarchical_triplet_weight': 0,
        #     'classification_weight': 1
        # },
        # 'h-contr_penalty': {
        #     'top_class_weight': 1, 
        #     'contrastive_weight': 1,
        #     'hierarchical_ce_weight': 0,
        #     'hierarchical_triplet_weight': 0,
        #     'classification_weight': 0
        # },
        't-contr_ce_penalty': {
            'classification_weight': 1,
            'top_class_weight': 1, 
            'contrastive_weight': 0.3,
            'hierarchical_ce_weight': 0,
            'hierarchical_triplet_weight': 0,
        },
    }

    for dataset_name, dataset_info in datasets.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        database = dataset_info['df']
        labels = database["class_idx"].tolist()

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        for hierarchy_name, weights in hierarchies.items():
            for mode in modes:
                print(f"\n=== Running experiments: Dataset={dataset_name} | Hierarchy={hierarchy_name} | Mode={mode} ===")

                for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                    print(f"\n==== Fold {fold} ====")

                    trainval_labels = [labels[i] for i in trainval_idx]
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
                    train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
                    train_idx = [trainval_idx[i] for i in train_idx_rel]
                    val_idx = [trainval_idx[i] for i in val_idx_rel]

                    train_df = database.iloc[train_idx].reset_index(drop=True)
                    val_df = database.iloc[val_idx].reset_index(drop=True)
                    test_df = database.iloc[test_idx].reset_index(drop=True)
                    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

                    train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.7)
                    val_dataset = HATRDataset(val_df, aug=False)
                    test_dataset = HATRDataset(test_df, aug=False)

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available()
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available()
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=torch.cuda.is_available()
                    )

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    emb_size_audio = 512 if mode in ['audio', 'both'] else 0
                    emb_size_text = 512 if mode in ['text', 'both'] else 0

                    hidden_size = 128
                    dropout = 0.1
                    use_batch_norm = True

                    model = BaseClassifier(
                        hidden_size=128,
                        num_classes=len(class_dict),
                        emb_size_audio=emb_size_audio,
                        emb_size_text=emb_size_text,
                        dropout=dropout,
                        use_batch_norm=use_batch_norm,
                        mode=mode
                    ).to(device)

                    class_to_top_class = build_class_to_top_class_mapping(class_dict, top_class_dict)
                    subclass_to_topclass_tensor = build_subclass_to_topclass_tensor(class_dict, top_class_dict, device)

                    classification_criterion = CrossEntropyLoss()
                    top_class_criterion = TopClassLoss(class_to_top_class, penalty_weight=1.0) if weights['top_class_weight'] > 0 else None
                    # contrastive_criterion = ContrastiveLoss_Top(class_to_top_class=class_to_top_class) if weights['contrastive_weight'] > 0 else None
                    contrastive_criterion = HierarchicalContrastiveLoss(subclass_to_topclass=subclass_to_topclass_tensor) if weights['contrastive_weight'] > 0 else None
                    hierarchical_ce_criterion = HierarchicalCrossEntropyLoss(subclass_to_topclass_tensor, alpha=0.5) if weights['hierarchical_ce_weight'] > 0 else None
                    # hierarchical_triplet_criterion = HierarchicalTripletLoss(subclass_to_topclass_tensor, margin_same_top=0.5, margin_diff_top=1.0) if weights['hierarchical_triplet_weight'] > 0 else None
                    hierarchical_triplet_criterion = CosineTripletLoss_Top(class_to_top_class) if weights['hierarchical_triplet_weight'] > 0 else None

                    top_class_weight = weights['top_class_weight']
                    contrastive_weight = weights['contrastive_weight']
                    hierarchical_ce_weight = weights['hierarchical_ce_weight']
                    hierarchical_triplet_weight = weights['hierarchical_triplet_weight']

                    output_dir = os.path.join(
                        './model_output',  # f'model_output_both_{current_time}',
                        hierarchy_name,
                        mode, f"fold_{fold}"
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    model_path = os.path.join(output_dir, "best_model.pth")
                    if train_from_scratch:
                        init_weights(model)

                        best_accuracy, history, trained_model = train_model(
                            model, train_loader, val_loader, device,
                            num_epochs=num_epochs, lr=learning_rate,
                            classification_weight=classification_weight,
                            contrastive_weight=contrastive_weight,
                            top_class_weight=top_class_weight,
                            hierarchical_ce_weight=hierarchical_ce_weight,
                            hierarchical_triplet_weight=hierarchical_triplet_weight,
                            classification_criterion=classification_criterion,
                            contrastive_criterion=contrastive_criterion,
                            top_class_criterion=top_class_criterion,
                            hierarchical_ce_criterion=hierarchical_ce_criterion,
                            hierarchical_triplet_criterion=hierarchical_triplet_criterion,
                            output_dir=output_dir,
                            scheduler_type=scheduler_type, patience=patience, early_stopping_factor=early_stopping_factor
                        )
                        print(f"Best validation accuracy: {best_accuracy:.2f}%")

                        # Save updated history with model info
                        history['model_info'] = {
                            'model_class': trained_model.__class__.__name__,
                            'hidden_size': hidden_size,
                            'num_classes': len(class_dict),
                            'emb_size_audio': emb_size_audio,
                            'emb_size_text': emb_size_text,
                            'dropout': dropout,
                            'use_batch_norm': True,
                            'mode': mode,
                            'fold_id': fold,
                            'random_seed': random_seed
                        }
                        history_path = os.path.join(output_dir, "history.json")
                        with open(history_path, "w") as f:
                            json.dump(make_serializable(history), f, indent=2)

                    else:
                        # to skip training, load a trained model 
                        # !! change trained_model to model in test !!
                        print(f"Loading pretrained weights from {model_path}")
                        state_dict = torch.load(model_path, map_location=device)
                        model.load_state_dict(state_dict)

                    
                    test_accuracy, true_top_class_accuracy, accuracy_incl_second, true_top_class_accuracy_incl_second = test_model(
                        trained_model, model_path, test_loader, device,
                        class_to_top_class,
                        output_dir=output_dir,
                        model_name="BaseClassifier",
                        fold_id=fold,
                        class_dict=class_dict,
                        top_class_dict=top_class_dict
                    )

                    sample_idxs = random.sample(range(len(val_dataset)), 5)
                    # Attention visualizer
                    if attention_visualization:
                        print("Generating attention visualizations...")
                        visualizer = AttentionVisualizer(
                            model=trained_model,
                            save_path=os.path.join(output_dir, "attention"),
                            subset=val_dataset,
                            device=device
                        )
                        for idx in sample_idxs:
                            visualizer.single_sample_attention(val_dataset[idx])
                        visualizer.avg_attention_by_class()

                    # Inter-embedding activation
                    if activation_visualization:
                        print("Generating activation visualizations...")
                        activ_savepath = os.path.join(output_dir, "activations")
                        if mode == 'both':
                            visualizer = ActivationVisualizer(
                                model=trained_model,  # model
                                save_path=activ_savepath,
                                subset=val_dataset,
                                features_to_plot=['audio_embedding', 'text_embedding'],
                                class_dict=class_dict,
                                top_class_dict=top_class_dict
                            )
                        for sample_idx in sample_idxs:
                            sample = val_dataset[sample_idx]
                            visualizer.single_sample_activation(sample)
                        for class_idx in range(len(class_dict)):
                            visualizer.avg_class_activation(class_idx=class_idx)
                        for top_class_idx in range(len(top_class_dict)):
                            visualizer.avg_top_class_activation(top_class_idx=top_class_idx)

                    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
                        f.write(f"cls_acc: {test_accuracy:.2f}%\n")
                        f.write(f"top_acc: {true_top_class_accuracy:.2f}%\n")
                        f.write(f"cls_acc_2: {accuracy_incl_second:.2f}%\n")
                        f.write(f"top_acc_2: {true_top_class_accuracy_incl_second:.2f}%\n")

                    print("\n===== Fold Results =====")
                    print(f"Final model true top-class accuracy: {true_top_class_accuracy:.2f}%")
                    print(f"Final model accuracy: {test_accuracy:.2f}%")
                    print(f"Final model true top-class accuracy (incl. second): {true_top_class_accuracy_incl_second:.2f}%")
                    print(f"Final model accuracy (incl. second): {accuracy_incl_second:.2f}%")
                    print("========================")

    print("All experiments done!")