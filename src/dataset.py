import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import pathlib

def read_data(path_to_nifti, return_numpy=True):
    """Read a NIfTI image. Return a numpy array (default) or `nibabel.nifti1.Nifti1Image` object"""
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))

class HecktorDataset(Dataset):
    def __init__(self, sample_path, transforms=None):
        self.sample_path = sample_path
        self.transforms = transforms

    def __len__(self):
        return len(self.sample_path)

    def __getitem__(self, index):
        sample = dict()

        before_id = pathlib.Path(self.sample_path[index][0]).stem
        after_id = before_id.replace("_ct.nii","")
        sample['id'] = after_id

        img = [read_data(self.sample_path[index][i]) for i in range(2)]
        img = np.stack(img, axis=-1)
        sample['input'] = img

        mask = read_data(self.sample_path[index][-1])
        mask = np.expand_dims(mask, axis=3)
        assert img.shape[:-1] == mask.shape[:-1]
        sample['target'] = mask

        if self.transforms:
            sample = self.transforms(sample)

        return sample