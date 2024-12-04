import torch
import torch.nn as nn
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment

def compute_loss(predictions, targets, num_classes=4):
    """
        Compute the YOLO loss and class accuracy for per-class bounding box predictions.

        Args:
            predictions (torch.Tensor): Model predictions of shape (1, 114, 2, 4, 5)
            targets (torch.Tensor): Ground truth labels of shape (N_objects, 5)
            num_classes (int): Number of classes (4)
            img_size (tuple): Image size (height, width)
            conf_threshold (float): Confidence score threshold for considering a prediction
            iou_threshold (float): IoU threshold for a correct prediction

        Returns:
            total_loss (torch.Tensor): Computed loss for the image
            class_accuracy (float): Class accuracy as a fraction of correctly predicted objects
        """
    device = predictions.device

    # Remove batch dimension (batch_size=1)
    predictions = predictions.squeeze(0)  # Shape: (114, 2, 4, 5)
    targets = targets[0]

    conf_threshold = 0.8
    iou_threshold = 0.5

    grid_h, grid_w = 6, 19
    img_size = (365, 1220)
    num_cells = grid_h * grid_w  # Should be 114
    num_anchors = 2

    # Verify the number of cells matches
    assert predictions.shape[0] == num_cells, "Number of cells does not match grid dimensions."

    # Initialize target tensor
    target_tensor = torch.zeros_like(predictions, device=device)

    # Calculate stride (pixels per grid cell)
    stride_x = img_size[1] / grid_w  # Width stride
    stride_y = img_size[0] / grid_h  # Height stride

    # Build grid cell coordinates
    grid_indices = [(i // grid_w, i % grid_w) for i in range(num_cells)]

    # Counters for accuracy calculation
    correct_predictions = 0
    total_objects = targets.shape[0]

    # For each ground truth object
    for gt in targets:
        left, top, right, bottom, class_id = gt
        class_id = int(class_id)

        # Compute ground truth bounding box center, width, and height
        x_center = (left + right) / 2.0
        y_center = (top + bottom) / 2.0
        w = right - left
        h = bottom - top

        # Determine which grid cell the center falls into
        cell_x = int(x_center / stride_x)
        cell_y = int(y_center / stride_y)

        # Handle edge cases
        if cell_x >= grid_w:
            cell_x = grid_w - 1
        if cell_y >= grid_h:
            cell_y = grid_h - 1

        # Determine grid cell index
        cell_index = cell_y * grid_w + cell_x

        # For each anchor box
        for anchor_idx in range(num_anchors):
            # For the specific class, set target values
            # Set bounding box coordinates
            target_tensor[cell_index, anchor_idx, class_id, 0] = left
            target_tensor[cell_index, anchor_idx, class_id, 1] = top
            target_tensor[cell_index, anchor_idx, class_id, 2] = right
            target_tensor[cell_index, anchor_idx, class_id, 3] = bottom
            # Set confidence score to 1
            target_tensor[cell_index, anchor_idx, class_id, 4] = 1.0
            # Assume only one anchor is responsible; break after setting one
            break  # Remove this line if multiple anchors can be responsible

        # Accuracy calculation
        # Get the predicted bounding boxes and confidence scores for this cell, anchor, and class
        pred_bboxes = predictions[cell_index, :, class_id, 0:4]  # Shape: (num_anchors, 4)
        pred_confs = predictions[cell_index, :, class_id, 4]  # Shape: (num_anchors,)

        # Apply sigmoid to confidence scores to get probabilities between 0 and 1
        pred_confs = torch.sigmoid(pred_confs)

        # Select the anchor with the highest confidence score
        max_conf, max_conf_idx = torch.max(pred_confs, dim=0)
        pred_bbox = pred_bboxes[max_conf_idx]

        # Only consider predictions with confidence above the threshold
        if max_conf >= conf_threshold:
            # Compute IoU between predicted bbox and ground truth bbox
            iou = bbox_iou(pred_bbox, torch.tensor([left, top, right, bottom], device=device))

            if iou >= iou_threshold:
                correct_predictions += 1

    # Calculate class accuracy
    class_accuracy = correct_predictions / total_objects if total_objects > 0 else 0.0

    # Masks
    obj_mask = target_tensor[..., 4] == 1.0  # Object mask
    noobj_mask = target_tensor[..., 4] == 0.0  # No-object mask

    # Localization Loss (only for objects)
    mse_loss = nn.MSELoss(reduction='sum')
    loc_loss = mse_loss(predictions[obj_mask][..., 0:4], target_tensor[obj_mask][..., 0:4])

    # Confidence Loss
    bce_loss_conf = nn.BCEWithLogitsLoss(reduction='sum')
    conf_loss_obj = bce_loss_conf(predictions[obj_mask][..., 4], target_tensor[obj_mask][..., 4])
    conf_loss_noobj = bce_loss_conf(predictions[noobj_mask][..., 4], target_tensor[noobj_mask][..., 4])

    # Total Loss
    lambda_coord = 5.0
    lambda_noobj = 0.5
    total_loss = (lambda_coord * loc_loss) + conf_loss_obj + (lambda_noobj * conf_loss_noobj)

    print(total_loss)

    return total_loss, class_accuracy


def bbox_iou(box1, box2):
    """
    Computes IoU between two bounding boxes in [left, top, right, bottom] format.

    Args:
        box1 (torch.Tensor): Predicted bounding box, shape (4,)
        box2 (torch.Tensor): Ground truth bounding box, shape (4,)

    Returns:
        iou (float): Intersection over Union value
    """
    # Intersection coordinates
    inter_left = torch.max(box1[0], box2[0])
    inter_top = torch.max(box1[1], box2[1])
    inter_right = torch.min(box1[2], box2[2])
    inter_bottom = torch.min(box1[3], box2[3])

    # Intersection area
    inter_width = torch.clamp(inter_right - inter_left, min=0)
    inter_height = torch.clamp(inter_bottom - inter_top, min=0)
    inter_area = inter_width * inter_height

    # Areas of the bounding boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = area1 + area2 - inter_area + 1e-16  # Avoid division by zero

    # IoU
    iou = inter_area / union_area

    return iou.item()
