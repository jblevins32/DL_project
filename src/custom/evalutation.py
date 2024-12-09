import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score

def compute_loss(predictions, targets, num_classes=5):
    """
    Compute the YOLO-style loss and class accuracy for a batch of images.

    Args:
        predictions (torch.Tensor): (batch_size, num_cells, num_anchors, 9)... this is the output of the model
        targets (torch.Tensor): (batch_size, N_objects, 5)

    Returns:
        total_loss (torch.Tensor): Loss averaged over the batch
        class_accuracy (float): Accuracy over all objects in the batch
    """
    # import these next time
    device = predictions.device
    batch_size = predictions.size(0)
    
    # Initialize loss counters
    total_loss_batch = torch.zeros(1, device=device, requires_grad=True)
    total_f1Score = 0
    
    # Compute loss for each image in a batch one at a time
    for singleImageFrameIdx in range(batch_size):
        # predictions for single image
        pred_single = predictions[singleImageFrameIdx] # (num_cells, num_anchors, 9)
        target_single = targets[singleImageFrameIdx]  # (N_objects, 5)

        single_loss, single_f1Score, specific_losses = compute_loss_single_image(
            pred_single, target_single, num_classes=num_classes
        )

        if singleImageFrameIdx == 0:
            total_loss_batch = single_loss
        else:
            total_loss_batch = total_loss_batch + single_loss

        total_f1Score += single_f1Score

    # Average loss over the batch
    avg_loss = total_loss_batch / batch_size
    f1_score = total_f1Score / batch_size

    return avg_loss, f1_score, specific_losses

def compute_loss_single_image(predictions, targets, num_classes=5, img_size=(365, 1220),
                             grid_h=6, grid_w=19, conf_threshold=0.8, iou_threshold=0.5):
    """
    Compute loss and accuracy for a single image.

    Args:
        predictions (torch.Tensor): (num_cells, num_anchors, 9)
                                    9 = [x, y, w, h, conf, class1, class2, class3, class4]
        targets (torch.Tensor): (N_objects, 5) [left, top, right, bottom, class_id]
        num_classes (int): number of classes
        img_size (tuple): (height, width)
        grid_h (int), grid_w (int): grid dimensions
        conf_threshold (float)
        iou_threshold (float)

    Returns:
        loss (torch.Tensor): Scalar loss for this image
        correct_predictions (int): Number of correctly predicted objects
        total_objects (int): Number of ground-truth objects
    """

    device = predictions.device
    num_cells = grid_h * grid_w # generalize this
    num_anchors = predictions.shape[1] # number of bounding boxes per gridbox
    assert predictions.shape == (num_cells, num_anchors, 10), "Prediction shape mismatch."

    img_h, img_w = img_size

    # Normalize targets, converting from 0 to 1
    normalized_targets = targets.clone().to(device)
    if normalized_targets.numel() > 0: # Check to be sure the targets are not empty
        normalized_targets[:, [0, 2]] /= img_w
        normalized_targets[:, [1, 3]] /= img_h

    # Initialize target tensor
    # shape: (num_cells, num_anchors, 9)
    # 0:4 -> box coords, 4 -> conf, 5:9 -> one-hot classes
    target_tensor = torch.zeros_like(predictions, device=device)

    # Assign targets to grid
    for gt in normalized_targets:

        # Extract normalized coordinates for grid target
        bbox_left_target, bbox_top_target, bbox_right_target, bbox_bottom_target, target_class_id = gt
        target_class_id = int(target_class_id)

        cell_index = getPredictionCellForTargetBBOX(gt, grid_w, grid_h)

        # Accuracy calculation
        cell_pred = predictions[cell_index]  # (num_anchors, 9)
        pred_anchor_bboxes = cell_pred[:, 0:4]
        pred_anchor_confs = cell_pred[:, 4]
        pred_anchor_classes = cell_pred[:, 5:]

        # determines which of the anchors for this cell had higher confidence in prediction
        max_conf, max_conf_idx = torch.max(pred_anchor_confs, dim=0)
        pred_bbox = pred_anchor_bboxes[max_conf_idx]
        pred_class = torch.argmax(pred_anchor_classes[max_conf_idx])

        # Assign to anchor with highest predicted confidence
        target_tensor[cell_index, max_conf_idx, 0] = bbox_left_target
        target_tensor[cell_index, max_conf_idx, 1] = bbox_top_target
        target_tensor[cell_index, max_conf_idx, 2] = bbox_right_target
        target_tensor[cell_index, max_conf_idx, 3] = bbox_bottom_target
        target_tensor[cell_index, max_conf_idx, 4] = 1.0

        # One-hot class
        class_vec = torch.zeros(num_classes, device=device)
        class_vec[target_class_id] = 1.0
        target_tensor[cell_index, max_conf_idx, 5:] = class_vec

        # pred_left = pred_bbox[0] * img_w
        # pred_top = pred_bbox[1] * img_h
        # pred_right = pred_bbox[2] * img_w
        # pred_bottom = pred_bbox[3] * img_h
        #
        # target_left = bbox_left_target * img_w
        # target_top = bbox_top_target * img_h
        # target_right = bbox_right_target * img_w
        # target_bottom = bbox_bottom_target * img_h

            # iou = bbox_iou(
            #     torch.tensor([pred_left, pred_top, pred_right, pred_bottom], device=device),
            #     torch.tensor([target_left, target_top, target_right, target_bottom], device=device)
            # )
            # if iou >= iou_threshold:
            #

    target_tensor = target_tensor.reshape(num_cells * num_anchors, 5 + num_classes)
    predictions = predictions.reshape(num_cells * num_anchors, 5 + num_classes)

    # Masks
    obj_mask = target_tensor[..., 4] == 1.0
    noobj_mask = target_tensor[..., 4] == 0.0

    # Loss functions
    loc_loss_fn = nn.SmoothL1Loss(reduction='sum')
    bce_loss_conf = nn.BCEWithLogitsLoss(reduction='sum')
    bce_loss_class = nn.CrossEntropyLoss(reduction='sum')

    # Localization loss
    pred_imgScale = torch.sigmoid(predictions[obj_mask][..., 0:4]) * torch.tensor([img_w, img_h, img_w, img_h], device=device)
    target_imgScale = target_tensor[obj_mask][..., 0:4] * torch.tensor([img_w, img_h, img_w, img_h], device=device)
    loc_loss = loc_loss_fn(pred_imgScale, target_imgScale)

    # Confidence loss
    conf_loss_obj = bce_loss_conf(predictions[obj_mask][..., 4], target_tensor[obj_mask][..., 4])
    conf_loss_noobj = bce_loss_conf(predictions[noobj_mask][..., 4], target_tensor[noobj_mask][..., 4])

    # Class loss
    class_loss = bce_loss_class(predictions[obj_mask][..., 5:5+num_classes], target_tensor[obj_mask][..., 5:5+num_classes])

    # Weights for each component of the loss function, currently evenly weighted
    lambda_boundingBoxes = 0.5
    lambda_confidence = 100.0
    lambda_noObjectBoxes = 1.0
    lambda_classScore = 100.0

    # Calculating each component of loss with weights
    bboxLoss = lambda_boundingBoxes * loc_loss
    confidenceLoss = lambda_confidence * conf_loss_obj
    backgroundLoss = lambda_noObjectBoxes * conf_loss_noobj
    classScoreLoss = lambda_classScore * class_loss

    # Combines loss function component functions into a total loss value
    total_loss = bboxLoss + confidenceLoss + backgroundLoss + classScoreLoss

    # Metrics Calculation
    predictedClassArray = torch.argmax(torch.softmax(predictions[:, 5:], dim=1), dim=1)
    targetClassArray = torch.argmax(target_tensor[:, 5:], dim=1)
    f1_score = multiclass_f1_score(predictedClassArray, targetClassArray, num_classes=num_classes)

    return total_loss, f1_score, (bboxLoss, confidenceLoss, backgroundLoss, classScoreLoss)

def bbox_iou(box1, box2):
    """
    Computes IoU between two bounding boxes in [left, top, right, bottom] format.
    """
    inter_left = torch.max(box1[0], box2[0])
    inter_top = torch.max(box1[1], box2[1])
    inter_right = torch.min(box1[2], box2[2])
    inter_bottom = torch.min(box1[3], box2[3])

    inter_width = torch.clamp(inter_right - inter_left, min=0)
    inter_height = torch.clamp(inter_bottom - inter_top, min=0)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area + 1e-16

    iou = inter_area / union_area
    return iou.item()

def getPredictionCellForTargetBBOX(gt, grid_w, grid_h):
    # Extract normalized coordinates for grid target
    bbox_left_target, bbox_top_target, bbox_right_target, bbox_bottom_target, _ = gt

    x_center = (bbox_left_target + bbox_right_target) / 2.0
    y_center = (bbox_top_target + bbox_bottom_target) / 2.0

    # rounding here to improve grid cell association rather than int truncation (basically floor)
    cell_x = int(torch.round(x_center * grid_w))
    cell_y = int(torch.round(y_center * grid_h))

    # Handle edges
    if cell_x >= grid_w:
        cell_x = grid_w - 1
    if cell_y >= grid_h:
        cell_y = grid_h - 1

    # Determine index of cell being estimated to contain this target
    cell_index = cell_y * grid_w + cell_x

    return cell_index

