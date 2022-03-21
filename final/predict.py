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
import src.models.UNETR as UNETR
import src.losses as losses
import src.trainer as trainer
import src.metrics as metrics
import src.metrics as metrics_2

def run2():
    cuda_device = "cuda:0"
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    if device.type == 'cpu':
        print('Start training the model on CPU')
    else:
        print(f'Start training the model on {torch.cuda.get_device_name(torch.cuda.current_device())}')

    path_to_test_data =  './dataset/train_data/val'

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 4, 1)
    fn_class = lambda x: 1.0 * (x > 0.5)

    test_paths = utils.get_paths_to_patient_files(path_to_test_data, True)

    # train and val data transforms:
    test_transforms = transforms.Compose([
        transforms.NormalizeIntensity(),
        transforms.ToTensor()])

    # datasets:
    test_set = dataset.HecktorDataset(test_paths, transforms=test_transforms)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    criterion = losses.Dice_and_FocalLoss()
    metric = metrics_2.dice
    precision = metrics_2.precision
    recall = metrics_2.recall

    # new_model = models.BaselineUNet(2, 2, 4).to(device)
    new_model = UNETR.UNETR_ITN().to(device)
    new_model.load_state_dict(torch.load("./results/result(UNETR_ITN)/best_model_weights.pt"))  # 100 epoch
    new_model.eval()

    phase_loss = 0.0  # Train or val loss
    phase_metric = 0.0
    phase_precision = 0.0
    phase_recall = 0.0

    phase_loss_after = 0.0  # Train or val loss
    phase_metric_after = 0.0
    phase_precision_after = 0.0
    phase_recall_after = 0.0

    result_txt_before_threshold = "./results/result(UNETR_ITN)/test_result_threshold_before.txt"
    if os.path.isfile(result_txt_before_threshold):
        os.unlink(result_txt_before_threshold)
    if not os.path.isfile(result_txt_before_threshold):
        f = open(result_txt_before_threshold, 'w')
        f.close()

    result_txt_after_threshold = "./results/(UNETR_ITN)/test_result_threshold_after.txt"
    if os.path.isfile(result_txt_after_threshold):
        os.unlink(result_txt_after_threshold)
    if not os.path.isfile(result_txt_after_threshold):
        f = open(result_txt_after_threshold, 'w')
        f.close()

    # 결과 저장할 폴더 있으면 삭제하고 생성 없으면 생성
    result_path = "./results/result(UNETR_ITN)/result_image"
    if os.path.isdir(result_path):
        os.rmdir(result_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    with torch.no_grad():
        for data in test_loader:
            # forward pass
            id = data['id'][0]
            target = data['target']
            input = data['input']
            input, target = input.to(device), target.to(device)

            output = new_model(input)
            output_after_threshold = fn_class(output)

            # before threshold cal loss & DSC
            loss = criterion(output, target)
            metrics = metric(output.detach(), target.detach())
            precisions = precision(output.detach(), target.detach())
            recalls = recall(output.detach(), target.detach())

            phase_loss += loss.item()
            phase_metric += metrics.item()
            phase_precision += precisions.item()
            phase_recall += recalls.item()

            # after threshold cal loss & DSC
            loss_after = criterion(output_after_threshold, target)
            metrics_after = metric(output_after_threshold.detach(), target.detach())
            precisions_after = precision(output_after_threshold.detach(), target.detach())
            recalls_after = recall(output_after_threshold.detach(), target.detach())

            phase_loss_after += loss_after.item()
            phase_metric_after += metrics_after.item()
            phase_precision_after += precisions_after.item()
            phase_recall_after += recalls_after.item()

            # 사진으로 저장하기 위한 tensor to numpy
            input = np.squeeze(fn_tonumpy(input))
            input_ct = input[:, :, :, 0]
            input_pt = input[:, :, :, 1]
            target1 = np.squeeze(fn_tonumpy(target))
            output1 = fn_class(np.squeeze(fn_tonumpy(output)))

            with open(result_txt_before_threshold, 'a') as f:
                data = f'loss: {loss:.3f} \tmetric: {metrics:.3f} \tprecision: {precisions:.3f} \trecall: {recalls:.3f} \n'
                f.write(data)

            with open(result_txt_after_threshold, 'a') as f:
                data = f'loss: {loss_after:.3f} \tmetric: {metrics_after:.3f} \tprecision: {precisions_after:.3f} \trecall: {recalls_after:.3f}\n'
                f.write(data)

            # 사진 결과 저장하기 (thresholing한 것)
            proxy = nib.load(
                './dataset/train_data/val/' + id + '_ct.nii.gz')
            # img_pt = nib.Nifti1Image(input_pt, proxy.affine, proxy.header)
            # img_ct = nib.Nifti1Image(input_ct, proxy.affine, proxy.header)
            # img_target = nib.Nifti1Image(target1, proxy.affine, proxy.header)
            img_output = nib.Nifti1Image(output1, proxy.affine, proxy.header)
            # img_ct.to_filename(os.path.join(result_path, id + '_ct.nii.gz'))
            # img_pt.to_filename(os.path.join(result_path, id + '_pet.nii.gz'))
            # img_target.to_filename(os.path.join(result_path, id + '_roi.nii.gz'))
            img_output.to_filename(os.path.join(result_path, id + '_output.nii.gz'))

    phase_loss /= len(test_loader)
    phase_metric /= len(test_loader)
    phase_precision /= len(test_loader)
    phase_recall /= len(test_loader)

    phase_loss_after /= len(test_loader)
    phase_metric_after /= len(test_loader)
    phase_precision_after /= len(test_loader)
    phase_recall_after /= len(test_loader)

    with open(result_txt_before_threshold, 'a') as f:
        data = f'Test loss: {phase_loss:.3f} \tavg_metric: {np.mean(phase_metric):.3f} \t metric: {phase_metric:.3f} \tprecision: {phase_precision:.3f} \trecall: {phase_recall:.3f}'
        f.write(data)

    with open(result_txt_after_threshold, 'a') as f:
        data = f'Test loss: {phase_loss_after:.3f} \tavg_metric: {np.mean(phase_metric_after):.3f} \t metric: {phase_metric_after:.3f} \t precision: {phase_precision_after:.3f} \trecall: {phase_recall_after:.3f}'
        f.write(data)


if __name__ =='__main__':
    run2()
