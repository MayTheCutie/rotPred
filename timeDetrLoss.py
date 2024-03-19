import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


        # Compute the giou cost betwen boxes TODO : fix to angle_box
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TimeSeriesDetrLoss(nn.Module):
    def __init__(self, num_classes: int, weight_set: float = 1.0, weight_bbox: float = 1.0, weight_class: float = 1.0):
        super(TimeSeriesDetrLoss, self).__init__()
        self.num_classes = num_classes
        self.weight_set = weight_set
        self.weight_bbox = weight_bbox
        self.weight_class = weight_class

    def forward(self, pred_bbox: Tensor, target_bbox: Tensor, class_logits: Tensor, target_classes: Tensor):
        """
        Computes the combined loss for the time series object detection task.

        Args:
            pred_bbox (Tensor): Predicted angles of shape (batch_size, num_obj, 2).
            target_bbox (Tensor): Target angles of shape (batch_size, num_obj, 2).
            class_logits (Tensor): Predicted class logits of shape (batch_size, num_obj, num_classes).
            target_classes (Tensor): Target classes of shape (batch_size, num_obj, num_classe).

        Returns:
            Tensor: Combined loss value.
        """
        batch_size = pred_bbox.size(0)
        num_angles = pred_bbox.size(1)

        # Match predicted angles with target angles
        indices = self.matcher(pred_bbox, target_bbox)

        # Compute set loss
        set_loss_value = self.set_loss(pred_bbox, target_bbox)

        # Compute bounding box loss
        bbox_loss_value = self.bbox_loss(pred_bbox, target_bbox, indices)

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