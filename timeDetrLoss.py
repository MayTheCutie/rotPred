import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

class TimeSeriesDetrLoss(nn.Module):
    def __init__(self, num_classes: int, weight_set: float = 1.0, weight_bbox: float = 1.0, weight_class: float = 1.0):
        super(TimeSeriesDetrLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_set = weight_set
        self.weight_bbox = weight_bbox
        self.weight_class = weight_class

    def forward(self, pred_angles: Tensor, target_angles: Tensor, class_logits: Tensor, target_classes: Tensor):
        """
        Computes the combined loss for the time series object detection task.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).
            class_logits (Tensor): Predicted class logits of shape (batch_size, num_angles, num_classes).
            target_classes (Tensor): Target classes of shape (batch_size, num_angles).

        Returns:
            Tensor: Combined loss value.
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Match predicted angles with target angles
        indices = self.matcher(pred_angles, target_angles)

        # Compute set loss
        set_loss_value = self.set_loss(pred_angles, target_angles)

        # Compute bounding box loss
        bbox_loss_value = self.bbox_loss(pred_angles, target_angles, indices)

        # Compute classification loss
        class_loss_value = self.classification_loss(class_logits, target_classes, indices)

        # Combine losses
        combined_loss = self.weight_set * set_loss_value + self.weight_bbox * bbox_loss_value + self.weight_class * class_loss_value

        return combined_loss

    def matcher(self, pred_angles: Tensor, target_angles: Tensor) -> Tensor:
        """
        Matches predicted angles with target angles using the Hungarian algorithm.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).

        Returns:
            Tensor: Indices of matched predictions and targets of shape (batch_size, num_angles).
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Compute the pairwise distances between predicted and target angles
        dist_matrix = torch.cdist(pred_angles.view(-1, 2), target_angles.view(-1, 2), p=2)
        dist_matrix = dist_matrix.view(batch_size, num_angles, -1)

        # Use the Hungarian algorithm to find the optimal matching
        indices = []
        for batch_idx in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(dist_matrix[batch_idx].cpu().numpy())
            indices.append(col_ind)

        indices = torch.as_tensor(indices, device=pred_angles.device)

        return indices

    def set_loss(self, pred_angles: Tensor, target_angles: Tensor) -> Tensor:
        """
        Computes the set loss for predicted and target angles.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).

        Returns:
            Tensor: Set loss value.
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Compute the pairwise distances between predicted and target angles
        dist_matrix = torch.cdist(pred_angles.view(-1, 2), target_angles.view(-1, 2), p=2)
        dist_matrix = dist_matrix.view(batch_size, num_angles, -1)

        # Sum the minimum distances over all target angles
        loss = torch.sum(dist_matrix.min(dim=2)[0])

        # Normalize the loss by the total number of target angles
        loss = loss / (batch_size * num_angles)

        return loss

    def bbox_loss(self, pred_angles: Tensor, target_angles: Tensor, indices: Tensor) -> Tensor:
        """
        Computes the bounding box loss for matched predicted and target angles.

        Args:
            pred_angles (Tensor): Predicted angles of shape (batch_size, num_angles, 2).
            target_angles (Tensor): Target angles of shape (batch_size, num_angles, 2).
            indices (Tensor): Indices for matched predictions and targets of shape (batch_size, num_angles).

        Returns:
            Tensor: Bounding box loss value.
        """
        batch_size = pred_angles.size(0)
        num_angles = pred_angles.size(1)

        # Filter out unmatched angles
        matched_pred_angles = pred_angles[indices != -1]
        matched_target_angles = target_angles[indices != -1]

        # Compute the L1 loss between matched predicted and target angles
        bbox_loss = F.l1_loss(matched_pred_angles, matched_target_angles, reduction='sum')

        # Normalize the loss by the total number of matched angles
        bbox_loss = bbox_loss / (batch_size * num_angles)

        return bbox_loss

    def classification_loss(self, class_logits: Tensor, target_classes: Tensor, indices: Tensor) -> Tensor:
        """
        Computes the classification loss for matched predicted and target classes.

        Args:
            class_logits (Tensor): Predicted class logits of shape (batch_size, num_angles, num_classes).
            target_classes (Tensor): Target classes of shape (batch_size, num_angles).
            indices (Tensor): Indices for matched predictions and targets of shape (batch_size, num_angles).

        Returns:
            Tensor: Classification loss value.
        """
        batch_size = class_logits.size(0)
        num_angles = class_logits.size(1)

        # Filter out unmatched angles
        matched_class_logits = class_logits[indices != -1]
        matched_target_classes = target_classes[indices != -1]

        # Compute the binary cross-entropy loss
        class_loss = F.cross_entropy(matched_class_logits.view(-1, self.num_classes), matched_target_classes.view(-1))

        # Normalize the loss by the total number of matched angles
        class_loss = class_loss / (batch_size * num_angles)

        return class_loss