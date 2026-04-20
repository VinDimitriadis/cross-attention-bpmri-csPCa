"""Dataset class and consistent augmentations for prostate MRI."""
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Compose, RandFlipd
from torch.utils.data import Dataset

from .alignment import extract_pid


# ---------------------------------------------------------------------------
# Augmentations applied consistently across T2W / DWI / ADC
# ---------------------------------------------------------------------------
class ConsistentRotate2D:
    """Apply the same in-plane rotation to every sequence in the sample."""

    def __init__(self, rotate_range=(-np.pi / 16, np.pi / 16), prob: float = 1.0,
                 mode: str = "bilinear"):
        self.rotate_range = rotate_range
        self.prob = prob
        self.mode = mode

    def __call__(self, data_dict):
        if np.random.rand() > self.prob:
            return data_dict

        angle = np.random.uniform(*self.rotate_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        for key in ("t2w", "dwi", "adc"):
            arr = data_dict[key]
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()

            vol = torch.from_numpy(arr).float().squeeze(0)  # (D, H, W)
            D = vol.shape[0]
            vol = vol.unsqueeze(1)                          # (D, 1, H, W)

            affine = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                                  dtype=torch.float32).unsqueeze(0).repeat(D, 1, 1)
            grid = F.affine_grid(affine, vol.size(), align_corners=False)
            aug = F.grid_sample(vol, grid, mode=self.mode,
                                padding_mode="zeros", align_corners=False)

            data_dict[key] = aug.squeeze(1).unsqueeze(0)

        return data_dict


class ConsistentTranslate2D:
    """Apply the same in-plane translation to every sequence in the sample."""

    def __init__(self, translate_voxels=(20, 20), prob: float = 1.0,
                 mode: str = "bilinear"):
        self.translate_voxels = translate_voxels
        self.prob = prob
        self.mode = mode

    def __call__(self, data_dict):
        if np.random.rand() > self.prob:
            return data_dict

        for key in ("t2w", "dwi", "adc"):
            arr = data_dict[key]
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()

            vol = torch.from_numpy(arr).float().squeeze(0)  # (D, H, W)
            D, H, W = vol.shape
            vol = vol.unsqueeze(1)

            tx = np.random.uniform(-self.translate_voxels[0], self.translate_voxels[0]) / (W / 2)
            ty = np.random.uniform(-self.translate_voxels[1], self.translate_voxels[1]) / (H / 2)

            affine = torch.tensor([[1, 0, tx], [0, 1, ty]],
                                  dtype=torch.float32).unsqueeze(0).repeat(D, 1, 1)
            grid = F.affine_grid(affine, vol.size(), align_corners=False)
            aug = F.grid_sample(vol, grid, mode=self.mode,
                                padding_mode="zeros", align_corners=False)

            data_dict[key] = aug.squeeze(1).unsqueeze(0)

        return data_dict


def default_train_transforms():
    """Default training augmentation pipeline."""
    return Compose([
        RandFlipd(keys=["t2w", "dwi", "adc"], spatial_axis=[0], prob=0.4),
        ConsistentRotate2D(rotate_range=(-np.pi / 6, np.pi / 6), prob=0.3),
        ConsistentTranslate2D(translate_voxels=(20, 20), prob=0.3),
    ])


def default_eval_transforms():
    return Compose([])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ProstateMultiModalDataset(Dataset):
    """Dataset for aligned T2W / DWI / ADC volumes + clinical variables.

    Each file list must already be aligned by patient id (see
    :func:`src.data.alignment.align_sequences`). The same dataset class is used
    for training, validation, and test — only the augmentation pipeline and
    label dictionary differ between splits.

    Args:
        t2w_list, dwi_list, adc_list: Lists of ``.nii.gz`` paths.
        clinical_list: List of ``.npy`` paths; each file holds a 1D array of
            clinical scalars (e.g. age, PSA, PSA density).
        label_dict: Mapping ``{patient_id: label}`` with binary labels.
        transform: Optional MONAI-style dict transform over keys ``t2w``,
            ``dwi``, ``adc``.
        expected_shape: Required raw volume shape ``(D, H, W)``.
        expected_clinical_size: Required length of the clinical vector.
    """

    def __init__(self,
                 t2w_list, dwi_list, adc_list, clinical_list,
                 label_dict: dict,
                 transform=None,
                 expected_shape=(32, 224, 224),
                 expected_clinical_size: int = 3):
        assert len(t2w_list) == len(dwi_list) == len(adc_list) == len(clinical_list)
        self.t2w_list = t2w_list
        self.dwi_list = dwi_list
        self.adc_list = adc_list
        self.clinical_list = clinical_list
        self.label_dict = label_dict
        self.transform = transform
        self.expected_shape = expected_shape
        self.expected_clinical_size = expected_clinical_size

    def __len__(self):
        return len(self.t2w_list)

    @staticmethod
    def _assert_shape(arr, expected_shape, seq_name, path):
        if tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(
                f"[{seq_name}] bad shape {tuple(arr.shape)} "
                f"(expected {expected_shape}) — file: {path}"
            )

    @staticmethod
    def _normalize(img):
        """Min-max to [0, 1]; leave unchanged if constant."""
        lo, hi = img.min(), img.max()
        if hi == lo:
            return img
        return (img - lo) / (hi - lo)

    def __getitem__(self, idx):
        t2w_path = self.t2w_list[idx]
        dwi_path = self.dwi_list[idx]
        adc_path = self.adc_list[idx]
        cln_path = self.clinical_list[idx]

        pids = [extract_pid(p) for p in (t2w_path, dwi_path, adc_path, cln_path)]
        assert len(set(pids)) == 1, f"Patient-id mismatch at idx {idx}: {pids}"
        pid = pids[0]

        # Load volumes
        t2w = nib.load(t2w_path).get_fdata().astype(np.float32)
        dwi = nib.load(dwi_path).get_fdata().astype(np.float32)
        adc = nib.load(adc_path).get_fdata().astype(np.float32)

        self._assert_shape(t2w, self.expected_shape, "t2w", t2w_path)
        self._assert_shape(dwi, self.expected_shape, "dwi", dwi_path)
        self._assert_shape(adc, self.expected_shape, "adc", adc_path)

        # Normalize and add channel dim -> (1, D, H, W)
        t2w = self._normalize(t2w)[np.newaxis, ...]
        dwi = self._normalize(dwi)[np.newaxis, ...]
        adc = self._normalize(adc)[np.newaxis, ...]

        # Clinical scalars
        clinical = np.load(cln_path).astype(np.float32).reshape(-1)
        if clinical.size != self.expected_clinical_size:
            raise ValueError(
                f"Clinical vector for {pid} has size {clinical.size} "
                f"(expected {self.expected_clinical_size}). File: {cln_path}"
            )

        # Optional augmentations
        data = {"t2w": t2w, "dwi": dwi, "adc": adc}
        if self.transform is not None:
            data = self.transform(data)

        # Label lookup
        if pid not in self.label_dict:
            raise KeyError(f"Missing label for patient_id: {pid}")
        label = self.label_dict[pid]

        return data["t2w"], data["dwi"], data["adc"], clinical, label
