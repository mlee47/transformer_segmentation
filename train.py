import torch
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import os
import random
import torch.backends.cudnn

torch.backends.cudnn.benchmark = True
random_seed = 77
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

import src.utils as utils
import src.transform as transforms
import src.dataset as dataset
import src.models.Unet as Unet
import src.losses as losses
import src.trainer as trainer
import src.metrics as metrics
import src.metrics as metrics_2


def run(): #training

    torch.multiprocessing.freeze_support()
    path_to_train_data = './dataset/train_data/train'
    path_to_val_data = './dataset/train_data/val'
    path_to_save_dir = '../result'

    train_batch_size = 1
    val_batch_size = 1
    num_workers = 0
    lr = 1e-3  # initial learning rate
    n_epochs = 100  # number of training epochs (300 was used in the paper)
    n_cls = 2  # number of classes to predict (background and tumor)
    in_channels = 2  # number of input modalities
    n_filters = 4  # number of filters after the input (24 was used in the paper)
    reduction = 2  # parameter controls the size of the bottleneck in SENorm layers
    T_0 = 25  # parameter for 'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts'
    eta_min = 1e-5

    train_paths = utils.get_paths_to_patient_files(path_to_train_data, True)
    val_paths = utils.get_paths_to_patient_files(path_to_val_data, True)

    # train and val data transforms:
    train_transforms = transforms.Compose([
        transforms.Mirroring(p=0.5),
        transforms.RandomRotation(p=0.5, angle_range=[0, 45]),
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])

    val_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])

    # datasets:
    train_set = dataset.HecktorDataset(train_paths, transforms=train_transforms)
    val_set = dataset.HecktorDataset(val_paths, transforms=val_transforms)

    #dataloaders:
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader}


    model = Unet.BaselineUNet(in_channels,n_cls,n_filters)
    criterion = losses.Dice_and_FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    metric = metrics.dice
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)

    trainer_ = trainer.ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        metric=metric,
        scheduler=scheduler,
        num_epochs=n_epochs,
        parallel=True)

    trainer_.train_model()
    trainer_.save_results(path_to_dir=path_to_save_dir)


if __name__ =='__main__':
    run()

