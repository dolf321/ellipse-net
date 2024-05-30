import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from models import *
import utils
import metrics

def train(root = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    writer = SummaryWriter()
    now = datetime.now()

    model = YOLOv1ResNet().to(device)
    loss_function = SumSquaredErrorLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # Load the dataset
    train_set = YoloPascalVocDataset('train', normalize=True, augment=True)
    test_set = YoloPascalVocDataset('test', normalize=True, augment=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True
    )
    if root is None:
        # Create folders
        root = os.path.join(
            'models',
            'ellipse_net',
            now.strftime('%m_%d_%Y'),
            now.strftime('%H_%M_%S')
        )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))
    test_mAP = np.empty((2,0))


    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)


    #####################
    #       Train       #
    #####################
    torch.cuda.empty_cache()
    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            # 3 tuple, each with 64, 2, 3, 448, 448 (64 images, 2 duplicates, 3 channels, 448x448)
            # Squeeze the batch into one tensor (batch, S, S, IMAGEX, IMAGEY)
            data = data.to(device)
            labels = labels.to(device)
            data, labels, _ = utils.prep_data(data, labels)
            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item();
            del data, labels
        train_loss /= len(train_loader)

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        writer.add_scalar('Loss/train', train_loss, epoch)
        print("Train Losses", train_loss);
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)
                    data, labels, _ = utils.prep_data(data, labels)
                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)
                    test_loss += loss.item() 

                    del data, labels, predictions
            test_losses /= len(test_loader)
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            print("Test Losses", test_loss);
            torch.save(model.state_dict(), os.path.join(weight_dir, str(epoch)))
            save_metrics()
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))

if __name__ == '__main__':      # Prevent recursive subprocess creation
    train()