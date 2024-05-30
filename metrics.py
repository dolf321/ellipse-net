import torch
import config
import utils
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
from torchvision import transforms as T
def pred_to_ellipse(prediction):
    """
    (Batch, S,S, C+6*B(...classes, cx, cy, width, height, confidence, angle))) -> ellipses (Batch, S, S, B, 7(cx, cy, width, height, confidence, angle, class))
    """
    # print(prediction.shape)
    classes = prediction[..., :config.C] # (Batch, S, S, C)
    bounding_boxes = prediction[..., config.C:].reshape(*prediction.shape[:-1], config.B, 6) # (Batch, S, S, B, 6)
    # print(bounding_boxes)
    # print(bounding_boxes.shape, prediction.shape)
    class_indices = torch.argmax(classes, dim=-1).unsqueeze(-1); # (Batch, S, S, 1)

    new_shape = (*prediction.shape[:-1], config.B, 7)
    ellipses = torch.zeros(new_shape, device=prediction.device) # (Batch, S, S, B, 7)

    # Fill in the ellipses tensor
    ellipses[..., 6] = class_indices
    ellipses[..., :6] = bounding_boxes
    # Confidence is p_class * p_is_object
    ellipses[..., 4] = ellipses[..., 4] * torch.max(classes, dim=-1)[0].unsqueeze(-1) # (Batch, S, S, B)

    # Reorder the bounding boxes to (cx, cy, width, height, angle, confidence, class)
    # ellipses = ellipses[..., [0, 1, 2, 3, 5, 4, 6]]

    return ellipses

def confidence_threshold(ellipses, threshold = 0.2):
    """
    ellipses (Batch, S, S, B, 7(cx, cy, width, height, confidence, angle, class)) -> ellipses (Ellipses with confidence > threshold, S,S, B, 7(cx, cy, width, height, confidence, angle, class))
    """
    ellipses[ellipses[..., 4] < threshold] = 0
    return ellipses

def relative_to_absolute(ellipses):
    """
    ellipses (Ellipses with relative coordinates(Batch), S, S, B, 7(cx, cy, width, height, confidence, angle, class)) -> ellipses (Ellipses with absolute coordinates, S, S, B, 7(cx, cy, width, height, confidence, angle, class))
    """
    Batches, S, _, _, _ = ellipses.shape
    # print(ellipses[0])
    grid_Y = torch.arange(config.S, device=ellipses.device).view(1, -1, 1, 1).float() * config.GRID_SIZE_Y;
    grid_X = torch.arange(config.S, device=ellipses.device).view(1, 1, -1, 1).float() * config.GRID_SIZE_X;
    grid_Y = grid_Y.expand(Batches, -1, config.S, 1);
    grid_X = grid_X.expand(Batches, config.S, -1, 1);
    
    ellipses[..., 0] = ellipses[..., 0] * config.IMAGE_SIZE[0] + grid_X;
    ellipses[..., 1] = ellipses[..., 1] * config.IMAGE_SIZE[1] + grid_Y;
    ellipses[..., 2] = ellipses[..., 2] * config.IMAGE_SIZE[0];
    ellipses[..., 3] = ellipses[..., 3] * config.IMAGE_SIZE[1];
    ellipses[..., 5] = ellipses[..., 5] * 180;
    
    return ellipses

def compute_ious(ellipses1, ellipses2, step=5):
    """
    Args:
        ellipses1(torch.Tensor): (num_ellipses, 7(cx, cy, width, height, confidence, angle, class))
        ellipses2(torch.Tensor): (num_ellipses, 7(cx, cy, width, height, confidence, angle, class))
    Returns:
        ellipses(torch.Tensor): (num_ellipses, num_ellipses)
    """

    # ious = torch.zeros((ellipses1.size(0), ellipses2.size(0)), device=ellipses1.device)
    # for i in range(ellipses1.size(0)):
    #     for j in range(ellipses2.size(0)):
    #         # ious[i, j] = utils.compute_iou_shapely((utils.ellipse_to_shapely(*ellipses1[i, :5], resolution=3), utils.ellipse_to_shapely(*ellipses2[j, :5], resolution=3)))
    #         ious[i,j] = utils.get_ellipse_iou(ellipses1[i, :5], ellipses2[j, :5])
    # ious = utils.get_ellipse_iou(ellipses1, ellipses2) # (Batch, S, S, B, B)
    return utils.get_ellipse_iou_array(ellipses1, ellipses2, step=step);

def non_max_suppression(ellipses, iou_threshold = 0.5, step=5):
    """
    Args:
        ellipses(torch.Tensor): Ellipses shape (Batch, S, S, B, 7(cx, cy, width, height, confidence, angle, class))
        iou_threshold(float): Intersection over union threshold
    Returns:
        torch.Tensor: Ellipses after non-max suppression (Batch, S * S * B, 7(cx, cy, width, height, confidence, angle, class))
    """
    B, S, _, _, _ = ellipses.shape
    ellipses = ellipses.view(B, -1, 7)
    
    for i in range(B):
        # Sort ellipses by confidence
        ellipses[i] = ellipses[i][ellipses[i][:, 4].argsort(descending=True)] # shape (S * S * B, 7(cx, cy, width, height, confidence, angle, class))
        # Compute IOUs
        ious = compute_ious(ellipses[i], ellipses[i], step=step)
        # For each ellipse, if there is another ellipse with higher confidence and iou > iou_threshold, set confidence to 0
        discarded = set()
        for j in range(len(ellipses[i])):
            if j in discarded:
                continue
            for k in range(len(ellipses[i])):
                if ious[j, k] > iou_threshold and ellipses[i][j][6] == ellipses[i][k][6] and k != j: # If iou is greater than threshold and same class
                    discarded.add(k);
        ellipses[i][list(discarded), 4] = 0
    return ellipses
    
def get_ellipse_predictions(ground_truth, predictions, conf_threshold = 0.2, nms_threshold = 0.5, iou_threshold = 0.4, step=5):
    """
    Figure out ellipse and whether it's correct or not
    predictions (Batch, S, S, C+6*B) -> ellipses (Ellipses after non-max suppression, 8(cx, cy, width, height, confidence, angle, class, correct))

    Args:
        ground_truth(torch.Tensor): Ground truth tensor(our labels) shape (Batch, S, S, C+6*B)
        predictions(torch.Tensor): Predictions tensor shape (Batch, S, S, C+6*B)
        conf_threshold(float): Confidence threshold to consider an ellipse
        nms_threshold(float): Non-max suppression threshold
        iou_threshold(float): Intersection over union threshold
        step(int): Step size for bitmasking IOU calculation. (A higher step is lower resolution/less accurate iou)
    Returns:
        torch.Tensor: Ellipses with columns [cx, cy, width, height, confidence, angle class, correct] (correct is 1 if the ellipse is correct, 0 otherwise)
    """
    ground_truth_ellipses = pred_to_ellipse(ground_truth) # (Batch,S,S,C+6*B) -> (Batch, S, S, B, 7(cx, cy, width, height, confidence, angle, class))
    ground_truth_ellipses = relative_to_absolute(ground_truth_ellipses)
    ground_truth_ellipses = ground_truth_ellipses.view(ground_truth_ellipses.size(0), -1, 7)
    ellipses = pred_to_ellipse(predictions)
    ellipses = relative_to_absolute(ellipses)
    ellipses = confidence_threshold(ellipses, conf_threshold)
    ellipses = non_max_suppression(ellipses, nms_threshold, step=step) # (Batch, S*S*B, 7(cx, cy, width, height, confidence, angle, class)) # CAREFUL YOU NEED TO FEED LESS E
    ellipses = torch.cat([ellipses, torch.zeros(ellipses.shape[:-1] + (1,), device=ellipses.device)], dim=-1) # Add correct column
    total_gt = torch.zeros(len(utils.load_class_array()), device=ellipses.device)
    batch_size = ellipses.size(0)
    selected_ellipses = []
    for i in range(batch_size):
        # Remove ellipses with confidence 0
        selected = ellipses[i][ellipses[i][:, 4] > 0] # (num_ellipses, 7(cx, cy, width, height, confidence, angle, class))
        selected_ellipses.append(selected)
        if len(selected) == 0 or len(ground_truth_ellipses[i]) == 0:
            continue

        # Compute IOUs
        # print("Selected shape", selected.shape)
        # print("Ground truth shape", ground_truth_ellipses[i].shape)
        ious = compute_ious(selected, ground_truth_ellipses[i], step=step) # (num_ellipses, num_ground_truth)

        # For each ellipse, if there is a ground truth ellipse with higher confidence and iou > iou_threshold, set correct to 1
        for k in range(len(ground_truth_ellipses[i])):
            gt_class_index = ground_truth_ellipses[i][k][6]
            if ground_truth_ellipses[i][k][4] == 0: # If confidence of ground truth is 0,
                continue
            total_gt[int(gt_class_index)] += 1
            for j in range(len(selected)):
                if ious[j, k] > iou_threshold and selected[j][6] == gt_class_index:
                    # print("Iou is", ious[j, k])
                    selected[j][7] = 1
                    break

    return torch.cat(selected_ellipses, dim=0), total_gt


def calculate_AP(ellipses, total_gt, step=10):
    """
    Calculate Average Precision (AP) using a precision-recall curve over multiple thresholds.
    
    Args:
        ellipses(torch.Tensor): Ellipses with columns [cx, cy, width, height, confidence, angle, class, correct]
        total_gt(int): Total number of ground truth positives

    Returns:
        float: The average precision (AP) for the ellipses.

    """
    ellipses = ellipses[ellipses[:, 4].argsort(descending=True)]
    precision = []
    recall = []
    thresholds = torch.linspace(1, 0, steps=step)  # thresholds from 0 to 1 w.t. step

    for threshold in thresholds:
        # Apply the current threshold
        selected = ellipses[ellipses[:, 4] >= threshold]

        # print('Threshold:', threshold, 'Selected:', len(selected))
        # True positives are those correctly identified
        tp = torch.sum(selected[:, 7])
        fp = len(selected) - tp # incorrect predictions
        # Calculate precision and recall for this threshold
        precision.append(tp / (tp + fp + config.EPSILON))
        recall.append(tp / (total_gt + config.EPSILON))

    # Convert lists to tensors for calculation
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    # If its the precision and recall are constant, then multiply the two (not sure about this)
    if torch.all(precision == precision[0]) and torch.all(recall == recall[0]):
        return precision[0] * recall[0]
    ap = torch.trapz(precision, recall).item()
    # print(f"AP Calculation: {ap}")
    return ap

def get_mAP(ellipses, total_gt):
    """
    Args:
        ellipses(torch.Tensor): Ellipses with columns [cx, cy, width, height, confidence, angle, class, correct]
        total_gt(torch.Tensor): Total number of ground truth positives for each class [Classes]
    Returns:
        float: Mean Average Precision (mAP) for the ellipses.
    """
    class_names = utils.load_class_array()  # Load class names
    ap_scores = []
    # # Filter out ellipses with confidence 0
    # ellipses = ellipses[ellipses[:, 5] > 0]
    # print(ellipses)
    for class_index, class_name in enumerate(class_names):
        # Filter ellipses for this class and check if its not suppressed
        class_ellipses = ellipses[ellipses[:, 6] == class_index]
        
        # Total ground truths for this class
        tgt = total_gt[class_index].item()
        if len(class_ellipses) == 0 and tgt == 0: # If no predictions and no ground truths dont add to unique classes or AP scores
            continue
        if tgt == 0:
            continue

        # Calculate AP for this class across multiple thresholds
        ap = calculate_AP(class_ellipses, tgt)
        ap_scores.append(ap);
        # print("im in")
        # print(ap);
    
    # Calculate mAP
    mAP = torch.mean(torch.tensor(ap_scores)).item() if ap_scores else 0.0 # If no AP scores mAP is 0
    return mAP


# TESTED AND WORKING
def plot_ellipses(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None, plot=False):
    """
    Plot ellipses on an image
    Args:
        data: Image Tensor (S, S, 3)
        labels: Predictions Tensor (S, S, C + 6 * B)
        classes: List of class names [str]
        color: Color of the ellipses
        min_confidence: Minimum confidence to consider an ellipse
        max_overlap: Maximum overlap between ellipses
        file: File to save the image
        save: Whether to save the image
    """
    # Initialize variables
    fig, ax = plt.subplots()
    data = T.ToPILImage()(data)
    ax.imshow(data)
    labels = labels.unsqueeze(0) # (1, S, S, C + 6 * B)
    ellipses = pred_to_ellipse(labels)
    ellipses = relative_to_absolute(ellipses)
    ellipses = confidence_threshold(ellipses, min_confidence)
    ellipses = non_max_suppression(ellipses, max_overlap)
    ellipses = ellipses.view(-1, 7)
    ellipses = ellipses[ellipses[:, 4] > 0]
    for ellipse in ellipses:
        cx, cy, width, height, confidence, angle, class_index = ellipse
        class_name = classes[int(class_index)]
        ellipse = patches.Ellipse((cx, cy), width, height, angle=angle, edgecolor=color, facecolor='none')
        ax.add_patch(ellipse)
        text = f'{class_name} {confidence:.2f}'
        ax.text(cx, cy, text, color=color)
    if file:
        plt.savefig(file)
    elif plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # Rand float 0-1
    prediction = torch.rand(100, 7, 7, config.C + 6 * config.B)
    # print(torch.max(prediction))
    # prediction = torch.Tensor([
    #     [
    #         [
    #             [0.1,0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #             [0.1,0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #         ],
    #         [
    #             [0.1,0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    #         ] 
    #     ]
    # ]);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(device)
    prediction = prediction.to(device)
    print(prediction.shape, prediction.device)
    ellipses = pred_to_ellipse(prediction)
    print(ellipses.shape)
    ellipses = relative_to_absolute(ellipses)
    print(ellipses.shape)
    ellipses = confidence_threshold(ellipses, 0.2)
    print(ellipses.shape)
    ellipses = non_max_suppression(ellipses)
    print(ellipses.shape)

    # print(ellipses)


   