import torch
import json
import os
import config
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from matplotlib import pyplot as plt

def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def scheduler_lambda(epoch):
    if epoch < config.WARMUP_EPOCHS + 75:
        return 1
    elif epoch < config.WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01


def load_class_dict():
    if os.path.exists(config.CLASSES_PATH):
        with open(config.CLASSES_PATH, 'r') as file:
            return json.load(file)
    new_dict = {}
    save_class_dict(new_dict)
    return new_dict


def load_class_array():
    classes = load_class_dict()
    result = [None for _ in range(len(classes))]
    for c, i in classes.items():
        result[i] = c
    return result


def save_class_dict(obj):
    folder = os.path.dirname(config.CLASSES_PATH)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(config.CLASSES_PATH, 'w') as file:
        json.dump(obj, file, indent=2)


def get_dimensions(label):
    size = label['annotation']['size']
    return int(size['width']), int(size['height'])


def get_bounding_boxes(label):
    width, height = get_dimensions(label)
    x_scale = config.IMAGE_SIZE[0] / width
    y_scale = config.IMAGE_SIZE[1] / height
    boxes = []
    objects = label['annotation']['object']
    for obj in objects:
        box = obj['bndbox']
        coords = (
            int(int(box['xmin']) * x_scale),
            int(int(box['xmax']) * x_scale),
            int(int(box['ymin']) * y_scale),
            int(int(box['ymax']) * y_scale)
        )
        name = obj['name']
        boxes.append((name, coords))
    return boxes


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::6]


def scale_bbox_coord(coord, center, scale):
    return ((coord - center) * scale) + center



# ----------------------------------ELLIPSE

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def prep_data(data, labels, originals = None):
    data = data.view(-1, *data.shape[2:]) # Squish the batch into one tensor (batch, S, S, IMAGEX, IMAGEY)
    labels = labels.view(-1, *labels.shape[2:])
    if originals is not None:
        originals = originals.view(-1, *originals.shape[2:])

    return data, labels, originals

def pred_to_ellipse(pred, min_confidence=0.2):
    """
    Convert a prediction to list of ellipses.
    """
    ellipses = []
    for i in range(config.S):
        for j in range(config.S):
            for k in range((pred.shape[2] - config.C) // 6):
                bbox_start = 6 * k + config.C
                bbox_end = bbox_start + 6
                bbox = pred[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(pred[i, j, :config.C]).item()
                confidence = pred[i, j, class_index].item() * bbox[4].item()
                if confidence > min_confidence:
                    cx = bbox[0] * config.IMAGE_SIZE[0] + (j * config.GRID_SIZE_X)
                    cy = bbox[1] * config.IMAGE_SIZE[1] + (i * config.GRID_SIZE_Y)
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    angle = bbox[5] * 180
                    ellipses.append((cx, cy, width, height, angle, confidence, class_index))
    return ellipses 
def non_max_suppression(ellipses, iou_threshold=0.5):
    """
    Apply non-maximum suppression to a list of ellipses.
    Args:
        ellipses: List of ellipses to suppress [(cx, cy, width, height, angle, confidence, class)]
        iou_threshold: Minimum IoU for a prediction to be considered correct (float)
    Returns:
        List of ellipses after non-maximum suppression
    """

    # Sort by highest to lowest confidence
    ellipses = sorted(ellipses, key=lambda x: x[5], reverse=True)
    num_boxes = len(ellipses)
    iou = torch.zeros((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(num_boxes):
            ellipse1 = ellipse_to_shapely(*ellipses[i][:5])
            ellipse2 = ellipse_to_shapely(*ellipses[j][:5])
            iou[i][j] = compute_iou_shapely((ellipse1, ellipse2))

    # Non-maximum suppression
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            curr_class = ellipses[i][6]
            for j in range(num_boxes):
                other_class = ellipses[j][6]
                if j != i and other_class == curr_class and iou[i][j] > iou_threshold:
                    discarded.add(j)

    # ellipses = [ellipses[i] for i in range(num_boxes) if i not in discarded]
    # # For every ellipse find 

    ellipse = []
    for i in range(num_boxes):
        if i not in discarded:
            ellipse.append(ellipses[i])
    # print(discarded)
    return ellipse

def plot_ellipses(data, labels, classes, color='orange', min_confidence=0.1, max_overlap=0.5, file=None, save=True):
    """
    Plots bounding ellipses of a given rotation on the given image.
    Args: 
        data: Image data tensor (3, H, W)
        labels: Predicted labels tensor (S, S, C + 6 * B(...classes, cx, cy, width, height, confidence, angle))
        classes: List of class names
        color: Color of bounding boxes
        min_confidence: Minimum confidence to display bounding box
        max_overlap: Maximum overlap between bounding boxes
        file: File to save image to
    Returns:
        None
    """
    # Define grid size based on image dimensions
    grid_size_x = data.size(dim=2) / config.S
    grid_size_y = data.size(dim=1) / config.S
    m = labels.size(dim=0)
    n = labels.size(dim=1)
    ellipses = [];
    # Iterate through each cell in the grid
    for i in range(m):
        for j in range(n):
            for k in range((labels.shape[2] - config.C) // 6):
                bbox_start = 6 * k + config.C
                bbox_end = bbox_start + 6
                bbox = labels[i, j, bbox_start:bbox_end]
                class_index = torch.argmax(labels[i, j, :config.C]).item() # Most likely class index for each cell(highest confidence) 
                # Confidence is the second last element
                confidence = labels[i, j, class_index].item() * bbox[4].item()  # that class confidence * bbox confidence(whether ellipse is present)
                # print(confidence, bbox[4].item(), labels[i, j, class_index].item())
                if confidence > min_confidence: 
                    # Calculate ellipse parameters
                    cx = bbox[0] * config.IMAGE_SIZE[0] + (j * grid_size_x)
                    cy = bbox[1] * config.IMAGE_SIZE[1] + (i * grid_size_y)
                    # If the cx, cy is outside the image, skip
                    if cx < 0 or cx > config.IMAGE_SIZE[0] or cy < 0 or cy > config.IMAGE_SIZE[1]:
                        continue
                    width = bbox[2] * config.IMAGE_SIZE[0]
                    height = bbox[3] * config.IMAGE_SIZE[1]
                    angle = bbox[5]*180
                    elem = (cx, cy, width, height, angle, confidence, class_index);
                    # print(cx, cy, width, height, angle, confidence, class_index)
                    # print(width, height)
                    elem = [x.item() if isinstance(x, torch.Tensor) else x for x in elem];
                    ellipses.append(elem);
                    
    # Sort by highest to lowest confidence
    ellipses = sorted(ellipses, key=lambda x: x[5], reverse=True)
    # Calculate IOUs between each pair of boxes
    num_boxes = len(ellipses);
    iou = torch.zeros((num_boxes, num_boxes));
    # Fix this bug THE POSITIONS ARE RELATIVE FOR IOU
    for i in range(num_boxes):
        for j in range(num_boxes):
            # Convert Ellipse to shapely
            ellipse1 = ellipse_to_shapely(*ellipses[i][:5]);
            ellipse2 = ellipse_to_shapely(*ellipses[j][:5]);
            iou[i][j] = compute_iou_shapely((ellipse1, ellipse2));

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data).convert('RGB')
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    discarded = set()
    for i in range(num_boxes): # For each box classified
        if i not in discarded: # If not already discarded, to avoid double counting
            cx, cy, width, height, angle, confidence, class_index = ellipses[i]
            # angle = angle - 90; # Fix angle appearajce
            # Discard overlapping ellipses that conflict with the highest confidence contender
            for j in range(num_boxes):
                other_class = ellipses[j][6]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Create and add the ellipse to the plot
            ellipse = patches.Ellipse((cx, cy), width, height, angle=angle, edgecolor=color, fill=False) # VIP 90-angle
            # Plot point at center of ellipse
            ax.plot(cx, cy, 'o', color=color, markersize=5)
            ax.add_patch(ellipse)

            # Add label with class name and confidence
            # print(classes, class_index, confidence)
            label = f'{classes[class_index]}: {confidence:.2%}'
            label_position = (cx, cy - height/2 - 10)  # Position label above the top of the ellipse
            plt.text(label_position[0], label_position[1], label, color=color, fontsize=8, ha='center')
    if file is None:
        plt.show()
    else:
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        plt.savefig(file)
        plt.close()
# # TESTED AND WORKING
# def plot_ellipses(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None, save=True):
#     """
#     Plots bounding ellipses of a given rotation on the given image.
#     Args: 
#         data: Image data tensor (3, H, W)
#         labels: Predicted labels tensor (S, S, C + 6 * B(...classes, cx, cy, width, height, confidence, angle))
#         classes: List of class names
#         color: Color of bounding boxes
#         min_confidence: Minimum confidence to display bounding box
#         max_overlap: Maximum overlap between bounding boxes
#         file: File to save image to
#     Returns:
#         None
#     """
#     ellipses = pred_to_ellipse(labels, min_confidence)
#     ellipses = non_max_suppression(ellipses, max_overlap);
#     image = T.ToPILImage()(data).convert('RGB')
#     _, ax = plt.subplots(1)
#     ax.imshow(image)
#     for ellipse in ellipses:
#         cx, cy, width, height, angle, confidence, class_index = ellipse;
#         ellipse = patches.Ellipse((cx, cy), width, height, angle=angle, edgecolor=color, fill=False)
#         ax.plot(cx, cy, 'o', color=color, markersize=5)
#         ax.add_patch(ellipse)
#         label = f'{classes[class_index]}: {confidence:.2%}'
#         label_position = (cx, cy - height/2 - 10)  # Position label above the top of the ellipse
#         plt.text(label_position[0], label_position[1], label, color=color, fontsize=8, ha='center')
#     if file is None:
#         plt.show()
#     elif save:
#         output_dir = os.path.dirname(file)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         if not file.endswith('.png'):
#             file += '.png'
#         plt.savefig(file)
#         plt.close()
#     else:
#         plt.close()


def get_iou_ellipse(p, a):

    """
    Compute the Intersection over Union (IoU) between batch of predicted ellipses and ground truth ellipses.
    Args:
        p (torch.Tensor): Predicted ellipses of shape (batch, S, S, C + 6 * config.B)
        a (torch.Tensor): Ground truth ellipses of shape (batch, S, S, C + 6 * config.B)
    Returns:
        torch.Tensor: IoU of each predicted ellipse with each ground truth ellipse for each grid cell in the batch of shape (batch, S, S, B, B)
    """

    batch, S, _, _ = p.shape;
    num_boxes = config.B;

    # Skip the first C classes cause we dont need them
    pt = p[..., config.C:]; # (batch, S, S, 6 * B)
    at = a[..., config.C:]; # (batch, S, S, 6 * B)
    pt = pt.view(batch, S, S, config.B, 6); # (batch, S, S, B, 6)
    at = at.view(batch, S, S, config.B, 6); # (batch, S, S, B, 6)

    ## FOR DEBUGGING SET ANGLE TO ZERO
    #pt[..., 5] = 0;
    #at[..., 5] = 0;

    # Remove Confidence
    pt = pt[..., [0, 1, 2, 3, 5]]; # (batch, S, S, B, 5)
    at = at[..., [0, 1, 2, 3, 5]]; # (batch, S, S, B, 5)
    intersection = torch.zeros((batch, S, S, config.B, config.B), device=p.device);
    for b in range(batch):
        for i in range(S):
            for j in range(S):

                # Prepare ellipses in Shapely
                ellipses_p = [ellipse_to_shapely(*pt[b, i, j, k, :],) for k in range(config.B)];
                ellipses_a = [ellipse_to_shapely(*at[b, i, j, k, :],) for k in range(config.B)];

                # Compute intersection
                for k in range(config.B):
                    for l in range(config.B):
                        intersection[b, i, j, k, l] = compute_iou_shapely((ellipses_p[k], ellipses_a[l]));

    return intersection;

def compute_iou_shapely(params):
    """Compute the IOU of two Shapely ellipses."""
    ellipse1, ellipse2 = params
    intersection = ellipse1.intersection(ellipse2).area
    union = ellipse1.union(ellipse2).area
    return intersection / union if union != 0 else 0

from shapely.geometry import Point
from shapely.affinity import scale, rotate
def ellipse_to_shapely(cx, cy, x_len, y_len, angle, resolution=5):
    """Create a Shapely ellipse from parameters."""
    # Shift coordinate frame 
    ellipse = Point(cx, cy).buffer(1, resolution=resolution);  # Create a circle around point
    ellipse = scale(ellipse, x_len, y_len);  # Scale to the ellipse size
    ellipse = rotate(ellipse, angle * 180, origin=(cx, cy)); # Rotate the ellipse, angle in degrees
    return ellipse;

# def ellipse_mask(ellipses, x, y):
#     """
#     Generate a mask for a batch of ellipses.
#     Args:
#         ellipses (torch.Tensor): Ellipses of shape (batch, S, S, config.B, 6) last 6 are (cx, cy, width, height, confidence, angle)
#         x (torch.Tensor): X-coordinates of shape (H, W)
#         y (torch.Tensor): Y-coordinates of shape (H, W)
#     Returns:
#         torch.Tensor: Mask of ellipses of shape (batch, S, S, config.B, H, W)
#     """

#     # Equation of ellipse
#     # \frac {((x-\Delta x)\cos\theta + (y-\Delta y)\sin \theta)^2}{a^2} + \frac {(-(x-\Delta x)\sin\theta + (y-\Delta y)\cos\theta)^2}{b^2} = 1

#     # Extract ellipse parameters
#     cx = ellipses[..., 0].unsqueeze(-1).unsqueeze(-1)  # (batch, S, S, 1, 1)
#     cy = ellipses[..., 1].unsqueeze(-1).unsqueeze(-1)  # (batch, S, S, 1, 1)
#     width = ellipses[..., 2].unsqueeze(-1).unsqueeze(-1)  # (batch, S, S, 1, 1)
#     height = ellipses[..., 3].unsqueeze(-1).unsqueeze(-1)  # (batch, S, S, 1, 1)
#     angle = ellipses[..., 5].unsqueeze(-1).unsqueeze(-1) * torch.pi  # (batch, S, S, 1, 1)
#     x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
#     y = y.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
#     a = width / 2
#     b = height / 2
#     cos = torch.cos(angle)
#     sin = torch.sin(angle)
    
#     # bbox_width = a * torch.abs(cos) + b * torch.abs(sin)
#     # bbox_height = a * torch.abs(sin) + b * torch.abs(cos)

#     # min_x = torch.clamp(cx - bbox_width, 0, config.IMAGE_SIZE[0])
#     # max_x = torch.clamp(cx + bbox_width, 0, config.IMAGE_SIZE[0])

#     # Compute the mask
#     mask = ((x - cx) * cos + (y - cy) * sin) ** 2 / a ** 2 + ((-(x - cx) * sin + (y - cy) * cos) ** 2) / b ** 2 <= 1
    
#     return mask.bool()

def ellipse_mask(ellipses, grid_x, grid_y):
    """
    Generate a mask for a batch of ellipses.
    Args:
        ellipses (torch.Tensor): Ellipses of shape (S, S, config.B, 6) last 6 are (cx, cy, width, height, confidence, angle)
        grid_x (torch.Tensor): X-coordinates of shape (H, W)
        grid_y (torch.Tensor): Y-coordinates of shape (H, W)
    Returns:
        torch.Tensor: Mask of ellipses of shape (S, S, config.B, H, W)
    """
    # print("ellipses shape", ellipses.shape)
    cx, cy, width, height, _, angle = ellipses[..., :6].unbind(-1) # (S, S, B)
    angle = angle * torch.pi
    a = width / 2
    b = height / 2
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    add_dims = lambda x: x.unsqueeze(-1).unsqueeze(-1)
    cx, cy, a, b, cos, sin = map(add_dims, (cx, cy, a, b, cos, sin)); # (S, S, B, 1, 1)
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, H, W)
    # print(cx.shape,grid_x.shape);
    # Calculate the mask
    mask = (
        ((grid_x - cx) * cos + (grid_y - cy) * sin) ** 2 / a ** 2
        + ((-(grid_x - cx) * sin + (grid_y - cy) * cos) ** 2) / b ** 2 <= 1
    )
    return mask.bool()

def compute_iou_bitmask(p,a, step=5, x_width = config.IMAGE_SIZE[0], y_width = config.IMAGE_SIZE[1]):
    """
    Args:
        p (torch.Tensor): Elipse of columns (cx, cy, width, height, confidence, angle, class)
        a (torch.Tensor): Elipse of columns (cx, cy, width, height, confidence, angle, class)
    """
    # Reshape the ellipses to (1, 7) and (1, 7)
    p = p.unsqueeze(0)
    a = a.unsqueeze(0)
    iou = get_ellipse_iou_array(p, a, step, x_width, y_width)
    return iou[0, 0].item()

def get_ellipse_iou_array(p,a, step=5, x_width = config.IMAGE_SIZE[0], y_width = config.IMAGE_SIZE[1]):
    """
    Args:
        p (torch.Tensor): Predicted ellipses of shape (num_ellipses, 7(cx, cy, width, height, confidence, angle, class))
        a (torch.Tensor): Ground truth ellipses of shape (num_ellipses, 7(cx, cy, width, height, confidence, angle, class))
    Returns:
        torch.Tensor: IoU of each predicted ellipse with each ground truth ellipse shape (num_p, num_a)
    """
    with torch.no_grad():
        num_p = p.shape[0]
        num_a = a.shape[0]
        ious = torch.zeros((num_p, num_a), device=p.device)
        y, x = torch.meshgrid(torch.arange(0, y_width, step=step, device=p.device), torch.arange(0, x_width, step=step, device=p.device))
        p = p.unsqueeze(0).unsqueeze(0) # (1, 1, num_p, 7)
        a = a.unsqueeze(0).unsqueeze(0) # (1, 1, num_a, 7)
        p_mask = ellipse_mask(p, x, y) # (1, 1, num_p, H, W)
        a_mask = ellipse_mask(a, x, y) # (1, 1, num_a, H, W)
        p_mask = p_mask.squeeze(0).squeeze(0) # (num_p, H, W)
        a_mask = a_mask.squeeze(0).squeeze(0) # (num_a, H, W)
        p_mask.unsqueeze_(1) # (num_p, 1, H, W)
        a_mask.unsqueeze_(0) # (1, num_a, H, W)
        # Print number of trues
        intersection = torch.sum(p_mask & a_mask, dim=(-1, -2), dtype=torch.int32) # (num_p, num_a)
        union = torch.sum(p_mask | a_mask, dim=(-1, -2), dtype=torch.int32)  # (num_p, num_a)
        ious = intersection / union.clamp(min=1e-6)  # Avoid division by zero
        return ious

def get_ellipse_iou_array_batched(p, a , step=5, x_width = config.IMAGE_SIZE[0], y_width = config.IMAGE_SIZE[1]):
    """
    Compute the Intersection over Union (IoU) between batch of predicted ellipses and ground truth ellipses using bitmasking techniques
    Args:
        p (torch.Tensor): Predicted ellipses of shape (batch, num_p, 7(cx, cy, width, height, angle, confidence, class))
        a (torch.Tensor): Ground truth ellipses of shape (batch, num_a, 7(cx, cy, width, height, angle, confidence, class))
    Returns:
        torch.Tensor: IoU of each predicted ellipse with each ground truth ellipse for each grid cell in the batch of shape (batch, num_p, num_a)
    """
    with torch.no_grad():
        batch, num_p, _ = p.shape
        _, num_a, _ = a.shape
        ious = torch.zeros((batch, num_p, num_a), device=p.device)
        y, x = torch.meshgrid(torch.arange(0, y_width, step=step, device=p.device), torch.arange(0, x_width, step=step, device=p.device))
        p = p.unsqueeze(2) # (batch, num_p, 1, 7)
        a = a.unsqueeze(1) # (batch, 1, num_a, 7)
        p_mask = ellipse_mask(p, x, y) # (batch, num_p, num_a, H, W)
        a_mask = ellipse_mask(a, x, y) # (batch, num_p, num_a, H, W)
        intersection = torch.sum(p_mask & a_mask, dim=(-1, -2) , dtype=torch.int32) # (batch, num_p, num_a)
        union = torch.sum(p_mask | a_mask, dim=(-1, -2), dtype=torch.int32) # (batch, num_p, num_a)
        ious = intersection / union.clamp(min=1e-6)
        return ious
        
# Compute ellipse iou using bitmasking
def get_ellipse_iou(p, a, step=5, x_width = config.IMAGE_SIZE[1], y_width = config.IMAGE_SIZE[0]):
    """
    Compute the Intersection over Union (IoU) between batch of predicted ellipses and ground truth ellipses using bitmasking techniques
    Args:
        p (torch.Tensor): Predicted ellipses of shape (batch, S, S, C + 6 * config.B)
        a (torch.Tensor): Ground truth ellipses of shape (batch, S, S, C + 6 * config.B)
    Returns:
        torch.Tensor: IoU of each predicted ellipse with each ground truth ellipse for each grid cell in the batch of shape (batch, S, S, B, B)
    """
    with torch.no_grad():
        batch, S, _, _ = p.shape;
        pt = p[..., config.C:]  # (batch, S, S, 6 * B)
        at = a[..., config.C:]  # (batch, S, S, 6 * B)
        B = pt.shape[-1] // 6
        pt = pt.view(batch, S, S, B, 6)  # (batch, S, S, B, 6)
        at = at.view(batch, S, S, B, 6)  # (batch, S, S, B, 6)
            
        # Generate ellipses, make sure IOU isn't calculated for those outside the image bounds
        y, x = torch.meshgrid(torch.arange(0, y_width, step=step, device=p.device), torch.arange(0, x_width, step=step, device=p.device))
        ious = torch.zeros((batch, S, S, B, B), device=p.device)
        for b in range(batch): # For every batch
            p_mask = ellipse_mask(pt[b], x, y)  # (S,S,B, 6) -> (S, S, B, H, W)
            a_mask = ellipse_mask(at[b], x, y)  # (S,S,B, 6) -> (S, S, B, H, W)
            p_mask = p_mask.unsqueeze(3)  # (S, S, B, H, W) -> (S, S, B, 1, H, W)
            a_mask = a_mask.unsqueeze(2) # (S, S, B, H, W) -> (S, S, 1, B, H, W)
            # print(p_mask.shape, a_mask.shape)
            # Get iou
            intersection = torch.sum(p_mask & a_mask, dim=(-1, -2), dtype=torch.int32) # (S, S, B, B)
            union = torch.sum(p_mask | a_mask, dim=(-1, -2), dtype=torch.int32) # (S, S, B, B)
            iou = intersection / union.clamp(min=1e-6) # (S, S, B, B)
            ious[b] = iou
        return ious

import math
def rotate_point(point, center, theta):
    """
    Rotate points around given centers with given rotation angles.
    
    Args:
        point (tuple): (x,y)
        center (tuple): (x,y)
        theta (float): Rotation angle in radians
    
    Returns:
        point (tuple): (x,y) Rotated point about center
    """
    # print('Point:', point, 'Center:', center, 'Theta:', theta)
    cos = math.cos(theta);
    sin = math.sin(theta);
    x = point[0] - center[0];
    y = point[1] - center[1];
    x_new = x * cos - y * sin;
    y_new = x * sin + y * cos;
    return (x_new + center[0], y_new + center[1]);