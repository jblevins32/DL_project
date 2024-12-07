import torch
import torch.nn as nn

def compute_loss(predictions, targets, num_classes=4):
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
    total_correct = 0
    total_objects = 0
    
    # Compute loss for each image in a batch one at a time
    for b in range(batch_size):
        # predictions for single image
        pred_single = predictions[b] # (num_cells, num_anchors, 9)
        target_single = targets[b]  # (N_objects, 5)

        loss_b, correct_b, objects_b = compute_loss_single_image(
            pred_single, target_single, num_classes=num_classes
        )

        if b == 0:
            total_loss_batch = loss_b
        else:
            total_loss_batch = total_loss_batch + loss_b

        total_correct += correct_b
        total_objects += objects_b

    # Average loss over the batch
    avg_loss = total_loss_batch / batch_size
    class_accuracy = total_correct / total_objects if total_objects > 0 else 0.0

    return avg_loss, class_accuracy

def compute_loss_single_image(predictions, targets, num_classes=4, img_size=(365, 1220),
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
    assert predictions.shape == (num_cells, num_anchors, 9), "Prediction shape mismatch."

    img_h, img_w = img_size

    # Sigmoid for bbox coords
    predictions[..., 0:4] = torch.sigmoid(predictions[..., 0:4])

    # Normalize targets, converting from 0 to 1
    normalized_targets = targets.clone().to(device)
    if normalized_targets.numel() > 0: # Check to be sure the targets are not empty
        normalized_targets[:, [0, 2]] /= img_w
        normalized_targets[:, [1, 3]] /= img_h

    # Initialize target tensor
    # shape: (num_cells, num_anchors, 9)
    # 0:4 -> box coords, 4 -> conf, 5:9 -> one-hot classes
    target_tensor = torch.zeros_like(predictions, device=device)

    correct_predictions = 0
    total_objects = normalized_targets.shape[0]

    # Assign targets to grid
    for gt in normalized_targets:
        
        # Extract normalized coordiantes for grid target
        left_norm, top_norm, right_norm, bottom_norm, class_id = gt
        class_id = int(class_id)

        x_center = (left_norm + right_norm) / 2.0
        y_center = (top_norm + bottom_norm) / 2.0

        cell_x = int(x_center * grid_w)
        cell_y = int(y_center * grid_h)

        # Handle edges
        if cell_x >= grid_w:
            cell_x = grid_w - 1
        if cell_y >= grid_h:
            cell_y = grid_h - 1

        # Determine index of cell being esimated to contain this target
        cell_index = cell_y * grid_w + cell_x

        # Assign to first anchor for simplicity
        target_tensor[cell_index, 0, 0] = left_norm
        target_tensor[cell_index, 0, 1] = top_norm
        target_tensor[cell_index, 0, 2] = right_norm
        target_tensor[cell_index, 0, 3] = bottom_norm
        target_tensor[cell_index, 0, 4] = 1.0

        # One-hot class
        class_vec = torch.zeros(num_classes, device=device)
        class_vec[class_id] = 1.0
        target_tensor[cell_index, 0, 5:5+num_classes] = class_vec

        # Accuracy calculation
        cell_pred = predictions[cell_index]  # (num_anchors, 9)
        pred_bboxes = cell_pred[:, 0:4]
        pred_confs = torch.sigmoid(cell_pred[:, 4])
        pred_classes = torch.softmax(cell_pred[:, 5:5+num_classes], dim=-1)

        max_conf, max_conf_idx = torch.max(pred_confs, dim=0)
        pred_bbox = pred_bboxes[max_conf_idx]

        pred_left = pred_bbox[0] * img_w
        pred_top = pred_bbox[1] * img_h
        pred_right = pred_bbox[2] * img_w
        pred_bottom = pred_bbox[3] * img_h

        if max_conf >= conf_threshold:
            iou = bbox_iou(
                torch.tensor([pred_left, pred_top, pred_right, pred_bottom], device=device),
                torch.tensor([gt[0]*img_w, gt[1]*img_h, gt[2]*img_w, gt[3]*img_h], device=device)
            )
            if iou >= iou_threshold:
                pred_class_id = torch.argmax(pred_classes[max_conf_idx]).item()
                if pred_class_id == class_id:
                    correct_predictions += 1

    # Masks
    obj_mask = target_tensor[..., 4] == 1.0
    noobj_mask = target_tensor[..., 4] == 0.0

    # Loss functions
    loc_loss_fn = nn.SmoothL1Loss(reduction='sum')
    bce_loss_conf = nn.BCEWithLogitsLoss(reduction='sum')
    bce_loss_class = nn.BCEWithLogitsLoss(reduction='sum')

    # Localization loss
    loc_loss = loc_loss_fn(predictions[obj_mask][..., 0:4], target_tensor[obj_mask][..., 0:4])

    # Confidence loss
    conf_loss_obj = bce_loss_conf(predictions[obj_mask][..., 4], target_tensor[obj_mask][..., 4])
    conf_loss_noobj = bce_loss_conf(predictions[noobj_mask][..., 4], target_tensor[noobj_mask][..., 4])

    # Class loss
    class_loss = bce_loss_class(predictions[obj_mask][..., 5:5+num_classes], target_tensor[obj_mask][..., 5:5+num_classes])

    lambda_coord = 5.0
    lambda_noobj = 0.5
    lambda_class = 1.0

    total_loss = (lambda_coord * loc_loss) + conf_loss_obj + (lambda_noobj * conf_loss_noobj) + (lambda_class * class_loss)
    return total_loss, correct_predictions, total_objects

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

