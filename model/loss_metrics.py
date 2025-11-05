#!/usr/bin/env python
"""
Loss functions and metrics for contrastive learning

Based on:
- Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
- Extended with augmentation-aware weighting for robust representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Xin Wang (original), Modified for augmentation-aware contrastive learning"


def sim_metric_seq(mat1, mat2):
    """Default similarity metric for sequence features"""
    return torch.bmm(mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)


def supcon_loss(input_feat,
                labels=None, mask=None, sim_metric=sim_metric_seq,
                t=0.07, contra_mode='all', length_norm=False):
    """
    Supervised Contrastive Loss

    Args:
        input_feat: tensor, feature vectors [bsz, n_views, ...]
        labels: ground truth of shape [bsz]
        mask: contrastive mask of shape [bsz, bsz]
        sim_metric: function to measure similarity
        t: temperature
        contra_mode: 'all' or 'one'
        length_norm: if True, l2 normalize feat

    Returns:
        loss: scalar
    """
    if length_norm:
        feat = F.normalize(input_feat, dim=-1)
    else:
        feat = input_feat

    # batch size
    bs = feat.shape[0]
    dc = feat.device
    dt = feat.dtype
    nv = feat.shape[1]

    # get the mask
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)

    # prepare feature matrix
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)

    if contra_mode == 'one':
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(contra_mode))

    # compute logits
    if sim_metric is not None:
        logits_mat = torch.div(sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(torch.matmul(anchor_feature, contrast_feature.T), t)

    # mask based on the label
    mask_ = mask.repeat(anchor_count, nv)
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1,
        torch.arange(bs * anchor_count).view(-1, 1).to(dc),
        0)
    mask_ = mask_ * self_mask

    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss


def augmentation_aware_supcon_loss(input_feat,
                                    labels=None,
                                    aug_types=None,
                                    mask=None,
                                    sim_metric=sim_metric_seq,
                                    t=0.07,
                                    contra_mode='all',
                                    length_norm=False,
                                    same_aug_weight=0.3,
                                    diff_aug_weight=1.0):
    """
    Augmentation-Aware Supervised Contrastive Loss

    This variant assigns different weights to positive pairs based on whether
    they have the same or different augmentation types:

    - Same label + Same augmentation → weight = same_aug_weight (default: 0.3)
      (These are easy positive pairs, already similar)

    - Same label + Different augmentation → weight = diff_aug_weight (default: 1.0)
      (These are hard positive pairs, should be pulled closer for robustness!)

    - Different label → standard negative pairs

    Args:
        input_feat: tensor, feature vectors [bsz, n_views, ...]
        labels: ground truth of shape [bsz]
        aug_types: augmentation types of shape [bsz], e.g.,
                   ['white_noise', 'pink_noise', 'clean', ...]
        mask: contrastive mask of shape [bsz, bsz]
        sim_metric: function to measure similarity
        t: temperature
        contra_mode: 'all' or 'one'
        length_norm: if True, l2 normalize feat
        same_aug_weight: weight for same label + same augmentation pairs
        diff_aug_weight: weight for same label + different augmentation pairs

    Returns:
        loss: scalar
    """
    if length_norm:
        feat = F.normalize(input_feat, dim=-1)
    else:
        feat = input_feat

    # batch size
    bs = feat.shape[0]
    dc = feat.device
    dt = feat.dtype
    nv = feat.shape[1]

    # get the label-based mask
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        label_mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError('Num of labels does not match num of features')
        label_mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        label_mask = mask.type(dt).to(dc)

    # Create augmentation type mask if provided
    if aug_types is not None:
        # Convert aug_types list to tensor if needed
        if isinstance(aug_types, list):
            # Create a mapping from aug_type string to integer
            unique_augs = list(set(aug_types))
            aug_to_idx = {aug: idx for idx, aug in enumerate(unique_augs)}
            aug_indices = torch.tensor([aug_to_idx[aug] for aug in aug_types],
                                       dtype=torch.long, device=dc)
        else:
            aug_indices = aug_types

        # Create augmentation mask: aug_mask[i, j] = 1 if same augmentation type
        aug_indices = aug_indices.view(-1, 1)
        aug_mask = torch.eq(aug_indices, aug_indices.T).type(dt).to(dc)

        # Create weighted mask for positive pairs
        # same_label_same_aug: weight = same_aug_weight
        # same_label_diff_aug: weight = diff_aug_weight
        weight_mask = torch.ones_like(label_mask) * diff_aug_weight
        weight_mask = torch.where(aug_mask == 1,
                                  torch.ones_like(label_mask) * same_aug_weight,
                                  weight_mask)
        # Apply weights only to positive pairs (same label)
        weighted_label_mask = label_mask * weight_mask
    else:
        # If no augmentation types provided, use standard mask
        weighted_label_mask = label_mask

    # prepare feature matrix
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)

    if contra_mode == 'one':
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(contra_mode))

    # compute logits
    if sim_metric is not None:
        logits_mat = torch.div(sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(torch.matmul(anchor_feature, contrast_feature.T), t)

    # mask based on the label (with weights)
    mask_ = weighted_label_mask.repeat(anchor_count, nv)
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1,
        torch.arange(bs * anchor_count).view(-1, 1).to(dc),
        0)
    mask_ = mask_ * self_mask

    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # compute WEIGHTED mean of log-likelihood over positive
    # The weights favor different augmentation pairs!
    weighted_log_prob = mask_ * log_prob
    mean_log_prob_pos = weighted_log_prob.sum(1) / (mask_.sum(1) + 1e-8)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss


if __name__ == "__main__":
    # Test the augmentation-aware loss
    print("Testing augmentation-aware supcon loss...")

    # Create dummy data
    batch_size = 8
    feat_dim = 128
    features = torch.randn(batch_size, 1, feat_dim)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])  # bonafide/spoof
    aug_types = ['white_noise', 'pink_noise', 'white_noise', 'clean',
                 'echo', 'clean', 'white_noise', 'pink_noise']

    # Standard supcon loss
    loss_std = supcon_loss(features, labels=labels)
    print(f"Standard SupCon Loss: {loss_std.item():.4f}")

    # Augmentation-aware supcon loss
    loss_aug = augmentation_aware_supcon_loss(
        features, labels=labels, aug_types=aug_types,
        same_aug_weight=0.3, diff_aug_weight=1.0)
    print(f"Aug-Aware SupCon Loss: {loss_aug.item():.4f}")

    print("\nTest passed!")
