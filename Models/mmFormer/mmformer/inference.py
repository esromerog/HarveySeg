import os
import argparse
import numpy as np
import medpy.io as medio
import time
import logging
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import nibabel as nib

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import mmformer
from data.transforms import Compose
import preprocess

def preprocessing(mask, src_path):

    mask_name = ['flair', 't1ce', 't1', 't2']
    modality_array = [None] * len(mask_name)
    default_shape = None

    # Load data and find default shape
    for i, item in enumerate(mask[0]):
        if item:
            data, _ = medio.load(src_path + '_' + mask_name[i] + '.nii.gz')
            modality_array[i] = data
            if default_shape is None:
                default_shape = data.shape

    # Check if a valid file was found
    if default_shape is None:
        raise ValueError("No valid data files found to determine the default shape.")

    # Replace None values with zeros of the default shape
    for i in range(len(modality_array)):
        if modality_array[i] is None:
            modality_array[i] = np.zeros(default_shape)

    vol = np.stack(modality_array, axis=0).astype(np.float32)
    x_min, x_max, y_min, y_max, z_min, z_max = preprocess.crop(vol)
    vol1 = preprocess.normalize(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])
    vol1 = vol1.transpose(1,2,3,0)
    print(vol1.shape)

    return vol1, [x_min, x_max, y_min, y_max, z_min, z_max, default_shape]

def make_prediction(
    x_input,
    model,
    feature_mask = None):

    H, W, T = 240, 240, 155
    model.eval()
    patch_size = 128
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float()

    num_cls = 4
    class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    class_separate = 'ncr_net', 'edema', 'enhancing'

    names = ['BraTS2020']
    mask = feature_mask
    modal_ind = np.array([0, 1, 2, 3])

    
    x = x_input[None, ...]
    x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) # [Bsize,channels,Height,Width,Depth]
    x = x[:, modal_ind, :, :, :]
    x = torch.from_numpy(x)

    print(x.shape)
    _, _, H, W, Z = x.size()
    #########get h_ind, w_ind, z_ind for sliding windows
    h_cnt = np.int32(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
    h_idx_list = range(0, h_cnt)
    h_idx_list = [h_idx * np.int32(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
    h_idx_list.append(H - patch_size)

    w_cnt = np.int32(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
    w_idx_list = range(0, w_cnt)
    w_idx_list = [w_idx * np.int32(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
    w_idx_list.append(W - patch_size)

    z_cnt = np.int32(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
    z_idx_list = range(0, z_cnt)
    z_idx_list = [z_idx * np.int32(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
    z_idx_list.append(Z - patch_size)

    #####compute calculation times for each pixel in sliding windows
    weight1 = torch.zeros(1, 1, H, W, Z).float()
    for h in h_idx_list:
        for w in w_idx_list:
            for z in z_idx_list:
                weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
    weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

    #####evaluation
    pred = torch.zeros(len(names), num_cls, H, W, Z).float()
    model.module.is_training=False
    for h in h_idx_list:
        for w in w_idx_list:
            for z in z_idx_list:
                x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                pred_part = model(x_input, mask)
                pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
    pred = pred / weight
    pred = pred[:, :, :H, :W, :T]
    pred = torch.argmax(pred, dim=1)

    return pred
    


def get_mask_from_modalities(modalities):
    # modalities is an array with the available modalities, such as:
    # [t1, t1c] or [t2]
    mask = [False, False, False, False]
    mask_name = ['flair', 't1ce', 't1', 't2']
    for modality in modalities:
        if modality in mask_name:
            indx = mask_name.index(modality)
            mask[indx] = True
    
    return torch.tensor([mask], dtype=torch.bool)

def get_map_location():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():  # Metal Performance Shaders for macOS
        return 'mps'
    else:
        return 'cpu'


def uncrop(cropped_vol, original_shape, x_min, x_max, y_min, y_max, z_min, z_max):
    # Create an empty array with the original shape
    uncropped_vol = np.zeros(original_shape)
    
    # Calculate the slices for the original array based on the cropped dimensions
    x_slice = slice(x_min, x_max)
    y_slice = slice(y_min, y_max)
    z_slice = slice(z_min, z_max)
    
    # Place the cropped volume back into the original array shape
    uncropped_vol[x_slice, y_slice, z_slice] = cropped_vol
    
    return uncropped_vol

    
def _plot_results(prediction_results, src_path, target_path):
    # Size stats
    # [49, 185, 42, 214, 4, 138, (240, 240, 155)]
    # output_data = np.load(output_path)
    output_data = np.squeeze(prediction_results, axis=0)
    
    og_data_load = nib.load(src_path)
    og_data = og_data_load.get_fdata()

    new_vol=uncrop(output_data, og_data.shape, 49, 185, 42, 214, 4, 138)
    #plt.imshow(new_vol[80, :, :])
    #plt.show()

    img = nib.Nifti1Image(new_vol.astype(np.int32), og_data_load.affine)
    nib.save(img, target_path)


def segment_tumor(source_path, output_path, model_checkpoint_path, modalities):
    # Ensure reproducibility and performance consistency
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Initialize the model
    model = mmformer.Model(num_cls=4)
    model = torch.nn.DataParallel(model)

    # Load the trained model checkpoint
    map_location = get_map_location()
    checkpoint = torch.load(model_checkpoint_path, map_location=torch.device(map_location))
    logging.info('Best epoch: {}'.format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])

    # Record start time
    start_time = time.time()
    torch.set_grad_enabled(True)

    # Obtain mask from the specified modalities
    modality_mask = get_mask_from_modalities(modalities)  # Add location and loading here

    # Preprocess the input data
    preprocessed_input, size_statistics = preprocessing(modality_mask, source_path)

    # Make prediction using the model
    prediction = make_prediction(x_input=preprocessed_input, model=model, feature_mask=modality_mask)

    print("Tumor inference done")

    # Extract size statistics for uncropping
    x_min, x_max, y_min, y_max, z_min, z_max, original_shape = size_statistics

    # Uncrop the prediction to match the original image dimensions
    uncropped_prediction = uncrop(
        torch.squeeze(prediction, dim=0),
        original_shape,
        x_min, x_max, y_min, y_max, z_min, z_max
    )

    # Save the prediction as a NIfTI image
    nifti_image = nib.Nifti1Image(uncropped_prediction.astype(np.int32), og_data_load.affine)
    nib.save(nifti_image, output_path)
    
if __name__ == '__main__':
    src_path="/Users/esromerog/Developer/Galen/Segmentation/FNL/GitRepo/Data/Raw/BraTS20_Training_001/BraTS20_Training_001"
    model_path = "/Users/esromerog/Developer/Galen/Segmentation/FNL/GitRepo/Models/mmFormer/mmformer/output/model_last.pth"
    output_path = "/Users/esromerog/Developer/Galen/Segmentation/FNL/GitRepo/Data/Predictions/mmFormer/TumorSegmentation.nii.gz"

    segment_tumor(src_path, output_path, model_path, ['t1ce'])