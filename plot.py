import torch
import config
import os
import utils
from tqdm import tqdm
from data import YoloPascalVocDataset
from models import *
from torch.utils.data import DataLoader
import metrics

MODEL_DIR = 'models/ellipse_net/05_08_2024/11_53_33'

MAX_IMAGES = 10
batch_size = 1

# Rotate Images, run inference, plot ellipses
def plot_test_ellipse(model_dir=MODEL_DIR, print_metrics=True, max_images=MAX_IMAGES, batch_size=batch_size, plot=False):
   
    classes = utils.load_class_array()

    dataset = YoloPascalVocDataset('test', normalize=False, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = YOLOv1ResNet()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'weights', 'final')))
    count = 0
    all_ellipse_predictions = torch.zeros((len(loader)*config.S*config.S*config.B, 8)) # (S*S*B, 8(cx, cy, width, height, angle, confidence, class, correct pred))
    start_index = 0
    num_predictions = 0
    with torch.no_grad():
        for image, labels, original in tqdm(loader):
            image, labels, original = utils.prep_data(image, labels, original)
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                save = count < max_images
                file = os.path.join('results', f'{count}.png') if save and not plot else None
                metrics.plot_ellipses(
                    original[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    file=file,
                    min_confidence=0.2,
                    max_overlap=0.4,
                    plot=plot
                )

                count += 1
                if count > max_images:
                    break
            # if print_metrics:
            #     temp_ellipse_predictions, _ = metrics.get_ellipse_predictions(labels, predictions, conf_threshold=0.01, nms_threshold=0.35, iou_threshold=0.35)
            #     # print(labels.view(-1, 6*config.B + config.C).shape, temp_ellipse_predictions.shape)
            #     # all_ellipse_predictions[batch_count*config.S*config.S*config.B:(batch_count+1)*config.S*config.S*config.B] = temp_ellipse_predictions
            #     end_index = start_index + temp_ellipse_predictions.shape[0]
            #     all_ellipse_predictions[start_index:end_index] = temp_ellipse_predictions
            #     start_index = end_index
            #     num_predictions += temp_ellipse_predictions.shape[0]

            # Concatenate the predictions
            if count > max_images:
                break

    # Calculate mAP
    # if print_metrics:
    #     mAP = metrics.get_mAP(all_ellipse_predictions[:num_predictions]);
    #     print(f'mAP: {mAP.item()}')

if __name__ == '__main__':
    plot_test_ellipse()
