import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.01)

    def forward(self, logits, labels):
        return self.cross_entropy(logits, labels)
    
class TopClassLoss(nn.Module):
    def __init__(self, class_to_top_class, penalty_weight=1.0):
        super().__init__()
        self.class_to_top_class = class_to_top_class
        self.penalty_weight = penalty_weight

    def forward(self, logits, labels):
        with torch.no_grad():
            _, predicted = torch.max(logits, dim=1)

            pred_top = torch.tensor(
                [self.class_to_top_class[p.item()] for p in predicted],
                device=logits.device
            )
            true_top = torch.tensor(
                [self.class_to_top_class[t.item()] for t in labels],
                device=logits.device
            )

            mismatches = (pred_top != true_top).float()

        return mismatches.mean() * self.penalty_weight

class ContrastiveLoss_Top(nn.Module):
    def __init__(self, temperature=0.5, class_to_top_class=None):
        super(ContrastiveLoss_Top, self).__init__()
        self.temperature = temperature
        self.class_to_top_class = class_to_top_class
    
    def forward(self, latents, labels):
        if self.class_to_top_class is not None:
            top_labels = torch.tensor(
                [self.class_to_top_class[label.item()] for label in labels],
                device=labels.device
            )
            labels = top_labels
        
        batch_size = latents.size(0)
        latents_norm = F.normalize(latents, dim=1)

        sim_matrix = torch.matmul(latents_norm, latents_norm.t()) / self.temperature
        mask_pos = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask_pos = mask_pos - torch.eye(batch_size, device=mask_pos.device)
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()

        return loss

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, subclass_to_topclass: torch.Tensor, alpha=0.5):
        super().__init__()
        self.subclass_to_topclass = subclass_to_topclass 
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_sub, logits_top, targets_sub):
        targets_top = self.subclass_to_topclass[targets_sub] 

        loss_sub = self.ce(logits_sub, targets_sub)
        loss_top = self.ce(logits_top, targets_top)
        loss = self.alpha * loss_sub + (1 - self.alpha) * loss_top

        return loss

class HierarchicalTripletLoss(nn.Module):
    def __init__(self, subclass_to_topclass: torch.Tensor, margin_same_top=0.5, margin_diff_top=1.0):
        super().__init__()
        self.subclass_to_topclass = subclass_to_topclass
        self.margin_same_top = margin_same_top
        self.margin_diff_top = margin_diff_top

    def forward(self, anchor, positive, negative, label_anchor, label_negative):
        top_anchor = self.subclass_to_topclass[label_anchor]
        top_negative = self.subclass_to_topclass[label_negative]

        same_top = (top_anchor == top_negative).float()
        margin = same_top * self.margin_same_top + (1 - same_top) * self.margin_diff_top

        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(dist_pos - dist_neg + margin)
        
        return loss.mean()

class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, subclass_to_topclass:torch.Tensor, temperature=0.5, top_weight=0.8):
        super().__init__()
        self.subclass_to_topclass = subclass_to_topclass
        self.temperature = temperature
        self.top_weight = top_weight

    def forward(self, latents, labels):
        device = latents.device
        labels = labels.to(device)
        top_labels = self.subclass_to_topclass[labels].to(device)

        batch_size = latents.size(0)
        latents = F.normalize(latents, dim=1)

        sim_matrix = torch.matmul(latents, latents.T) / self.temperature
        exp_sim = torch.exp(sim_matrix)

        eye_mask = torch.eye(batch_size, device=device).bool()

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        same_topclass = top_labels.unsqueeze(0) == top_labels.unsqueeze(1)

        same_class[eye_mask] = False
        same_topclass[eye_mask] = False

        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        subclass_log_prob = (same_class.float() * log_prob).sum(1) / (same_class.sum(1) + 1e-8)
        topclass_log_prob = (same_topclass.float() * log_prob).sum(1) / (same_topclass.sum(1) + 1e-8)

        loss = -((1 - self.top_weight) * subclass_log_prob + self.top_weight * topclass_log_prob)
        
        return loss.mean()



class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)

        cos_pos = (anchor_norm * positive_norm).sum(dim=1)
        cos_neg = (anchor_norm * negative_norm).sum(dim=1)

        loss = F.relu(cos_neg - cos_pos + self.margin)
        return loss.mean()


class CosineTripletLoss_Top(nn.Module):
    def __init__(self, margin=0.3, class_to_top_class=None):
        super().__init__()
        self.margin = margin
        self.class_to_top_class = class_to_top_class

    def forward(self, embeddings, labels):
        # Map to top-level classes if provided
        if self.class_to_top_class is not None:
            labels = torch.tensor(
                [self.class_to_top_class[label.item()] for label in labels],
                device=labels.device
            )

        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = embeddings.size(0)
        loss = 0.0
        triplet_count = 0

        for i in range(batch_size):
            anchor = embeddings[i]
            label = labels[i]

            # Find positive indices (same label, not self)
            pos_indices = (labels == label).nonzero(as_tuple=False).squeeze()
            pos_indices = pos_indices[pos_indices != i]

            # Find negative indices (different label)
            neg_indices = (labels != label).nonzero(as_tuple=False).squeeze()

            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue  # skip if no positive or negative samples

            # Pick one positive and one negative at random (could be improved with hard mining)
            pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
            neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]

            positive = embeddings[pos_idx]
            negative = embeddings[neg_idx]

            cos_pos = torch.dot(anchor, positive)
            cos_neg = torch.dot(anchor, negative)
            triplet_loss = F.relu(cos_neg - cos_pos + self.margin)

            loss += triplet_loss
            triplet_count += 1

        if triplet_count == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss / triplet_count


class TwoLevelHierarchicalTripletLoss(nn.Module):
    def __init__(self, margin_subclass=0.3, margin_top=1.0, lambda_top=0.5):
        super().__init__()
        self.margin_subclass = margin_subclass
        self.margin_top = margin_top
        self.lambda_top = lambda_top

    def forward(self, anchor, positive, negative, label_anchor_sub, label_positive_sub, label_negative_sub,
                label_anchor_top, label_positive_top, label_negative_top):

        # Compute subclass triplet loss (fine-grained)
        dist_pos_sub = F.pairwise_distance(anchor, positive, p=2)
        dist_neg_sub = F.pairwise_distance(anchor, negative, p=2)
        loss_sub = F.relu(dist_pos_sub - dist_neg_sub + self.margin_subclass)

        # Compute top-level triplet loss (coarse)
        dist_pos_top = F.pairwise_distance(anchor, positive, p=2)
        dist_neg_top = F.pairwise_distance(anchor, negative, p=2)
        loss_top = F.relu(dist_pos_top - dist_neg_top + self.margin_top)

        # Combine losses weighted by lambda
        loss = (1 - self.lambda_top) * loss_sub + self.lambda_top * loss_top
        return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, latents, labels):

        batch_size = latents.size(0)
        
        latents_norm = F.normalize(latents, dim=1)
        
        sim_matrix = torch.matmul(latents_norm, latents_norm.t()) / self.temperature
        mask_pos = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        mask_pos = mask_pos - torch.eye(batch_size, device=mask_pos.device)
        
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        
        return loss


'''
class HierarchicalConsistencyLoss(nn.Module):
    def __init__(self, class_to_top_class, penalty_weight=1.0):
        super().__init__()
        self.class_to_top_class = class_to_top_class
        self.penalty_weight = penalty_weight
        
        # Precompute mapping tensor for efficiency
        self.num_classes = max(class_to_top_class.keys()) + 1
        self.num_top_classes = max(class_to_top_class.values()) + 1
        
        # Create a matrix that maps fine classes to top classes
        self.hierarchy_matrix = torch.zeros(self.num_classes, self.num_top_classes)
        for class_idx, top_class_idx in class_to_top_class.items():
            self.hierarchy_matrix[class_idx, top_class_idx] = 1.0
    
    def forward(self, fine_logits, true_labels):
        device = fine_logits.device
        self.hierarchy_matrix = self.hierarchy_matrix.to(device)
        
        # Get predicted probabilities for fine classes
        fine_probs = F.softmax(fine_logits, dim=1)
        
        # Aggregate to get implied top-class probabilities
        implied_top_probs = torch.matmul(fine_probs, self.hierarchy_matrix)
        
        # Get true top-class labels
        true_top_labels = torch.tensor(
            [self.class_to_top_class[label.item()] for label in true_labels], 
            device=device
        )
        
        # Compute cross-entropy loss for top-class consistency
        top_class_loss = F.cross_entropy(
            torch.log(implied_top_probs + 1e-8), 
            true_top_labels
        )
        
        return self.penalty_weight * top_class_loss

class HierarchicalCrossEntropyLoss(nn.Module):
    def __init__(self, top_class_dict, top_penalty=1, second_penalty=1):
        super().__init__()
        self.top_class_dict = top_class_dict
        self.top_penalty = top_penalty
        self.second_penalty = second_penalty

        self.sub_to_top = {}
        for top_class, sub_classes in top_class_dict.items():
            for sub_class in sub_classes:
                self.sub_to_top[sub_class] = int(top_class)

    def forward(self, logits, labels):
        base_loss = F.cross_entropy(logits, labels, reduction='none') 

        top2_preds = torch.topk(logits, k=2, dim=1).indices  
        penalties = torch.ones_like(base_loss)

        for i in range(logits.size(0)):
            true_sub = labels[i].item()
            true_top = self.sub_to_top[true_sub]

            first_pred = top2_preds[i][0].item()
            second_pred = top2_preds[i][1].item()

            if first_pred == true_sub:
                continue  
            elif second_pred == true_sub:
                penalties[i] *= self.second_penalty 
            elif self.sub_to_top[first_pred] == true_top:
                penalties[i] *= self.top_penalty  

        final_loss = base_loss * penalties
        return final_loss.mean()
    
class ExpertOrthogonalityLoss_OLD(nn.Module):
    def __init__(self):
        super(ExpertOrthogonalityLoss_OLD, self).__init__()

    def forward(self, model):
        expert_params = []
        for expert in model.experts:
            params = torch.cat([p.view(-1) for p in expert.parameters()], dim=0)
            expert_params.append(params)
        
        expert_params = torch.stack(expert_params, dim=0)
        expert_params_norm = F.normalize(expert_params, p=2, dim=1)
        gram = torch.matmul(expert_params_norm, expert_params_norm.t())
        gram = gram - torch.eye(gram.shape[0], device=gram.device)
        orthogonal_loss = torch.norm(gram, p='fro')
        
        return orthogonal_loss
    
class ExpertOrthogonalityLoss(nn.Module):
    def __init__(self, orthogonality_weight=1.0, load_balance_weight=1.0, dead_expert_weight=1.0, dead_threshold=1e-2):
        super(ExpertOrthogonalityLoss, self).__init__()
        self.orthogonality_weight = orthogonality_weight
        self.load_balance_weight = load_balance_weight
        self.dead_expert_weight = dead_expert_weight
        self.dead_threshold = dead_threshold

    def forward(self, model, expert_weights):
        total_loss = 0.0

        # 1. Orthogonality Loss on expert params
        expert_params = []
        for expert in model.experts:
            params = torch.cat([p.view(-1) for p in expert.parameters()], dim=0)
            expert_params.append(params)

        expert_params = torch.stack(expert_params, dim=0)  # [num_experts, param_count]
        expert_params_norm = F.normalize(expert_params, p=2, dim=1)
        gram = torch.matmul(expert_params_norm, expert_params_norm.t())
        gram = gram - torch.eye(gram.shape[0], device=gram.device)
        orthogonal_loss = torch.norm(gram, p='fro')
        total_loss += self.orthogonality_weight * orthogonal_loss

        # 2. Load balancing loss on expert_weights (softmax gating)
        expert_mean_usage = expert_weights.mean(dim=0)  # [num_experts]
        entropy = -torch.sum(expert_mean_usage * torch.log(expert_mean_usage + 1e-8))
        load_balance_loss = entropy  # maximize entropy = uniform usage
        total_loss += self.load_balance_weight * load_balance_loss

        # 3. Dead expert penalty (experts rarely used in batch)
        expert_usage = expert_weights.sum(dim=0)  # [num_experts]
        dead_mask = (expert_usage < self.dead_threshold).float()
        dead_expert_penalty = dead_mask.sum()  # count dead experts
        total_loss += self.dead_expert_weight * dead_expert_penalty

        return total_loss

class Mixture_KL(nn.Module):
    def __init__(self, kl_weigth=1.0):
        super(Mixture_KL, self).__init__()
    
        self.kl_weigth = kl_weigth

    def forward(self, mu, logvar, pi):
        _, num_mixtures, _ = mu.shape
        prior_pi = torch.ones(num_mixtures, device=mu.device) / num_mixtures
        
        kl_gaussians = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2)
        kl_mixture = torch.sum(pi * (torch.log(pi + 1e-8) - torch.log(prior_pi + 1e-8)), dim=1)
        total_kl = torch.sum(pi * kl_gaussians, dim=1) + kl_mixture
        
        return total_kl.mean() * self.kl_weigth

class KL_Loss(nn.Module):
    def __init__(self, kl_weigth=1.0):
        super(KL_Loss, self).__init__()
        self.kl_weigth = kl_weigth

    def forward(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean() * self.kl_weigth

'''
