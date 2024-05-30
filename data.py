import torch
import config
import utils
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset


class YoloPascalVocDataset(Dataset):
    def __init__(self, set_type, normalize=False, augment=False, duplicates = 2, angle_range=config.ROTATION_ANGLE_RANGE):
        assert set_type in {'train', 'test'}
        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year='2007',
            image_set=('train' if set_type == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
        )
        self.duplicates = 1 if set_type == 'test' else duplicates
        # self.dataset = torch.utils.data.Subset(self.dataset, range(0, len(self.dataset), 1));
        self.normalize = normalize
        self.augment = augment
        self.angle_range = angle_range
        self.classes = utils.load_class_dict()

        # Generate class index if needed
        index = 0
        if len(self.classes) == 0:
            for i, data_pair in enumerate(tqdm(self.dataset, desc=f'Generating class dict')):
                data, label = data_pair
                for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
                    name, coords = bbox_pair
                    if name not in self.classes:
                        self.classes[name] = index
                        index += 1
            utils.save_class_dict(self.classes)
    
    def process_data(self, data, label):
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()
        # angle = 30 * random.random() - 15 FAILED
        # angle = 0;
        # angle = -15;
        # angle = int(30 * random.random() - 15) FAILED
        # angle = random.random() * 15 WORKS
        # angle = -random.random() * 15
        # angle = int(90 * random.random() - 45) # With MSE loss
        # angle = -35;
        # angle = -35;
        angle = int(self.angle_range * random.random() - self.angle_range / 2)
    
        # Augment images
        data = TF.rotate(data, -angle) # Apple rotation (negate because clockwise is ccw as origin is top left)
        original_data = data
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 6 * config.B + config.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords
            
            half_width = config.IMAGE_SIZE[1] / 2
            half_height = config.IMAGE_SIZE[0] / 2
            # Augment labels
            if self.augment:
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            mid_x, mid_y = utils.rotate_point((mid_x, mid_y), (half_width, half_height), angle * torch.pi / 180);
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)
            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name
                    # Make sure angle between 0-180
                    temp_angle = (angle + 180) % 180 # we add 180 to make sure it's positive
                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coord relative to grid square
                            (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width
                            (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height
                            1.0,                                                    # Confidence
                            temp_angle / 180                                             # Angle [0-1]
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 6 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1
        return data, ground_truth, original_data

    def __getitem__(self, i):
        data, label = self.dataset[i]
        # if self.duplicates == 1:
        #     return self.process_data(data, label)
        # return self.process_data(data, label)
        # Generate duplicates and store in array
        dim_data = (self.duplicates, *data.size())
        dim_label = (self.duplicates, config.S, config.S, 6 * config.B + config.C)
        data_array = torch.zeros(dim_data)
        label_array = torch.zeros(dim_label)
        original_data = torch.zeros(dim_data)

        for i in range(self.duplicates):
            data_array[i], label_array[i], original_data[i] = self.process_data(data, label)
        return data_array, label_array, original_data
    
    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # # Display data
    # obj_classes = utils.load_class_array()
    train_set = YoloPascalVocDataset('train', normalize=False, augment=False, duplicates=1)

    # Test the dataset
    dataset = VOCDetection(
            root=config.DATA_PATH,
            year='2007',
            image_set=('train'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(config.IMAGE_SIZE)
            ])
    )

    data, label = dataset[15]
    print(data.shape, label)

    # Test the dataset
    pdata, ground_truth, original = train_set[15]
    pdata, ground_truth, original = pdata[0], ground_truth[0], original[0]
    print(pdata.shape, ground_truth.shape, original.shape)
    classes = utils.load_class_array()
    utils.plot_ellipses(original, ground_truth, classes, min_confidence=0.2, max_overlap=0.4, file='results/test.png');
