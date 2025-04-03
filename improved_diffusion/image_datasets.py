from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import os, glob
import scipy.io as sio
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_wavelets import DWTForward, DWTInverse
from scipy.interpolate import interp1d
import random

import numpy as np
import cv2

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
def fill_nan(A):
    """
    Interpolates data to fill nan values for multi-dimensional arrays.
    """

    if A.ndim == 1:
        inds = np.arange(A.shape[0])
        good = np.where(np.isfinite(A))
        f = interp1d(inds[good], A[good], bounds_error=False, fill_value="extrapolate")
        return np.where(np.isfinite(A), A, f(inds))

    elif A.ndim > 1:
        for dim in range(A.ndim):
            indices = np.indices(A.shape)
            for index in np.ndindex(*A.shape[:dim] + A.shape[dim + 1:]):
                slice_idx = tuple(index[:dim] + (slice(None),) + index[dim:])
                A_slice = A[slice_idx]

                if np.all(np.isnan(A_slice)):
                    continue

                if np.all(np.isfinite(A_slice)):
                    continue

                inds = indices[dim][slice_idx]
                good = np.isfinite(A_slice)
                f = interp1d(inds[good], A_slice[good], bounds_error=False, fill_value="extrapolate")
                A[slice_idx] = np.where(good, A_slice, f(inds))

        return A

    else:
        return A

def interpolate(input_arr, target_arr):
    y_zoom = target_arr.shape[0]/input_arr.shape[0]
    x_zoom = target_arr.shape[1]/input_arr.shape[1]
    zoom_factors = (y_zoom, x_zoom, 1)
    # zoom_factors = np.array(target_arr.shape) / np.array(input_arr.shape)
    interpolated_arr = zoom(input_arr, zoom_factors, order = 3)
    return interpolated_arr

def extract_prefix(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def load_paired_mat_data_test(
    *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '*.mat'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '*.mat'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = PairedMATDataset_test(
        image_size,
        all_inputs,
        all_targets,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_paired_npy_data(
    *, input_dir, target_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_inputs = glob.glob(os.path.join(input_dir, '*.npy'))
    if not target_dir:
        raise ValueError("unspecified data directory")
    all_targets = glob.glob(os.path.join(target_dir, '*.npy'))

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_targets]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = PairedNPYDataset(
        image_size,
        all_inputs,
        all_targets,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader

class PairedMATDataset_test(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            # inp = np.load(f).astype('float32')
            # inp = inp1[8:inp1.shape[1]-8,8:inp1.shape[1]-8,:]
            try:
                inp = sio.loadmat(f)['input'].astype('float32')
            except:
                inp = sio.loadmat(f)['input_tile'].astype('float32')
        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:

            tag = sio.loadmat(f)['target'].astype('float32')
            tag = tag * 255
        if inp.ndim < 3:
            inp = np.expand_dims(inp, axis=0)
        if tag.ndim < 3:
            tag = np.expand_dims(tag, axis=0)
        
        crop_y = np.random.randint(0, inp.shape[0]-self.resolution)
        crop_x = np.random.randint(0, inp.shape[1]-self.resolution)
        
        inp_copy = inp[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
        tag_copy = tag[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
            
        inp_copy = pixel_binning_5x(inp_copy)
        inp_copy = (inp_copy - inp_copy.mean()) / (inp_copy.std() + 1e-6)

        tag_copy = tag_copy.astype(np.float32)/ 127.5 - 1
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            
        return tag_copy.transpose([2,0,1]), inp_copy.transpose([2,0,1]), out_dict, os.path.basename(path), os.path.basename(os.path.dirname(path))


class PairedNPYDataset(Dataset):
    def __init__(self, resolution, input_images, target_images, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.input_images = input_images[shard:][::num_shards]
        self.target_images = target_images[shard:][::num_shards]
        self.input_fnames = [os.path.basename(fp) for fp in self.input_images]
        self.target_fnames = [os.path.basename(fp) for fp in self.target_images]
        self.common_fnames = [f for f in self.input_fnames if f in self.target_fnames]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.common_fnames)

    def __getitem__(self, idx):
        path = self.input_images[self.input_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            inp = np.load(f).astype('float32')
        path = self.target_images[self.target_fnames.index(self.common_fnames[idx])]
        with bf.BlobFile(path, "rb") as f:
            tag = np.load(f).astype('float32')
            tag = tag / 255.0
        if inp.ndim < 3:
            inp = np.expand_dims(inp, axis=0)
        if tag.ndim < 3:
            tag = np.expand_dims(tag, axis=0)
        
        tag_copy = np.ones_like(tag)
        n = 0
        while np.mean(tag_copy) * 255 >= 230:
            crop_y = np.random.randint(0, inp.shape[0]-self.resolution)
            crop_x = np.random.randint(0, inp.shape[1]-self.resolution)

            tag_copy = tag[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...].copy()
            n = n+1
            if n >5:
                break
            
        inp = inp[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution, ...]

        mode = random.randint(0, 7)
        inp, tag_copy = augment_img(inp, mode=mode), augment_img(tag_copy, mode=mode)

        inp = pixel_binning(inp, 3)
        
        inp = (inp - inp.mean()) / (inp.std() + 1e-6)
        tag_copy = tag_copy.astype(np.float32) * 255 / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return tag_copy.transpose([2,0,1]), inp.transpose([2,0,1]), out_dict



def pixel_binning(input_array, binning_factor=4):
    """
    Perform pixel binning on an input array of shape (256, 256, 4).
    
    Args:
        input_array (numpy.ndarray): Input array of shape (256, 256, 4).
        
    Returns:
        numpy.ndarray: Binned array of shape (256/4, 256/4, 4).
    """
    # Ensure the input dimensions are compatible
    # if input_array.shape[0] != 256 or input_array.shape[1] != 256:
    #     raise ValueError("Input dimensions must be 256x256.")
    
    # # Crop the array to make it divisible by 4
    # crop_size = 4 * (input_array.shape[0] // 4)  # Find the largest dimension divisible by 4
    # cropped_array = input_array[:crop_size, :crop_size, :]
    cropped_array = input_array
    
    # Reshape and take the mean across the 4x4 blocks
    binned_array = cropped_array.reshape(cropped_array.shape[0]//binning_factor, binning_factor, cropped_array.shape[1]//binning_factor, binning_factor, 4).mean(axis=(1,3))
    
    return binned_array