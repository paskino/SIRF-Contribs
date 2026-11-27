import ismrmrd
import os
from sirf.Gadgetron import ImageData
from sirf.Gadgetron import CoilSensitivityData
from sirf.Gadgetron import AcquisitionModel
from cil.optimisation.utilities.callbacks import Callback
import math
from skimage.metrics import mean_squared_error

def change_ismrmrd(full_filename_in, full_filename_out):
    if full_filename_in == full_filename_out:
        raise ValueError('Input and output filename are the same. This would overwrite the original data.')

    # Set trajectory and save
    if os.path.exists(full_filename_out) == 1:
        os.remove(full_filename_out)
        print('{} deleted'.format(full_filename_out))
        
    with ismrmrd.File(full_filename_in, 'r') as file:
        ds = file[list(file.keys())[0]]
        ismrmrd_header = ds.header
        acquisitions = ds.acquisitions[:]

    # Modify header
    matrixSizeY = min(ismrmrd_header.encoding[0].encodedSpace.matrixSize.y,  ismrmrd_header.encoding[0].reconSpace.matrixSize.y)
    matrixSizeY = max(matrixSizeY, ismrmrd_header.encoding[0].encodingLimits.kspace_encoding_step_1.maximum+1)
    ismrmrd_header.encoding[0].encodedSpace.matrixSize.y = matrixSizeY
    ismrmrd_header.encoding[0].reconSpace.matrixSize.y = matrixSizeY

    # Create new file
    # https://github.com/ismrmrd/ismrmrd-python/blob/d55eed97e266e8a1339777379a1350a39c377c50/ismrmrd/hdf5.py#L165
    with ismrmrd.Dataset(full_filename_out) as ds:
        ds.write_xml_header(ismrmrd_header.toXML())

        for acq in acquisitions:
            ds.append_acquisition(acq)

    


# Convert Complex data to abs and save to DICOM
# https://github.com/SyneRBI/XNAT-SIRF/blob/2b0b6bf928df2793b27e4ce4ca4673b65db19a9e/docker/reco_scripts/sirf_util.py#L8
import numpy as np
from pathlib import Path
import pydicom
from pydicom.pixels import set_pixel_data
import datetime
import sirf.Reg as reg
import sirf.Gadgetron as mr
import tempfile
from nii2dcm.run import run_nii2dcm

import logging

logger = logging.getLogger(__name__)

def to_dicom_folder(
    data: mr.ImageData, 
    foldername: str | Path,
    filename_prefix: str = "sirf",
    series_uid: str | None = None,
    series_description: str | None = None,
    resolution: float = 1.0,
    **kwargs
) -> None:

    # cast to range of uint16
    pixel_data = data.abs()
    pixel_data = pixel_data / pixel_data.max() * (2**16 - 1)
    # pixel_data = np.swapaxes(pixel_data, -1, -2)

    # save to temporary nifti file
    nim = reg.NiftiImageData3D(pixel_data)
    
    nii_fname =os.path.join(foldername, "temp.nii")
    nim.write(nii_fname)
    
    run_nii2dcm(nii_fname, foldername, dicom_type="MR")

    os.remove(nii_fname)



class LogfileCallback(Callback):
    def __init__(self, log_file: str, verbose: bool = False):
        self.log_file = log_file
        self.verbose = verbose

    def __call__(self, algorithm) -> None:
        iteration = algorithm.iteration
        objective_value = algorithm.get_last_objective()
        if self.verbose and algorithm.iteration % algorithm.update_objective_interval == 0:
            with open(self.log_file, "a") as f:
                f.write(f"Iteration {iteration}: objective value = {objective_value}\n")


from collections.abc import Sequence
def gaussian_variable_density_samples(
        shape: Sequence[int], low: int, high: int, fwhm: float = float('inf'), always_sample: Sequence[int] = ()
    ) -> np.ndarray:
        """Generate Gaussian variable density samples.
        Generates indices in [low, high[ with a gaussian weighting.
        Parameters
        ----------
        shape
            Shape of the output tensor. The generated indices are 1D and in the last dimension.
            All other dimensions are batch dimensions.
        low
            Lower bound of the sampling domain.
        high
            Upper bound of the sampling domain.
        fwhm
            Full-width at half-maximum of the Gaussian.
        always_sample
            indices that should always included in the samples.
            For example, `range(-n_center//2, n_center//2)`
        Returns
        -------
            1D array of selected indices.
        """
        *n_batch, n_samples = shape
        if n_samples > high - low:
            raise ValueError('n_samples must be <= (high - low)')
        n_random = n_samples - len(always_sample)
        if n_random < 0:
            raise ValueError('more always sampled points requested than total number of samples')
        elif n_random == 0:
            return np.sort(np.broadcast_to(np.array(always_sample), (*n_batch, -1)))

        pdf = np.exp(-np.log(2.0) * (2 * np.arange(low, high) / fwhm) ** 2)
        pdf[[x - low for x in always_sample]] = 0

        if len(shape) > 1:
            pdf_batch = np.broadcast_to(pdf, (*n_batch, len(pdf))).reshape(-1, len(pdf))
            # normalize each row to a valid probability distribution
            pdf_batch = pdf_batch / pdf_batch.sum(axis=-1, keepdims=True)
            idx_rand = np.array([
                np.random.default_rng().choice(len(pdf), size=n_random, replace=False, p=row) + low
                for row in pdf_batch
            ]).reshape(*n_batch, n_random)
        else:
            pdf = pdf / pdf.sum()
            idx_rand = np.random.default_rng().choice(len(pdf), size=n_random, replace=False, p=pdf) + low

        idx_always = np.broadcast_to(np.array(always_sample), (*n_batch, len(always_sample)))
        return np.sort(np.concatenate([idx_rand, idx_always], axis=-1), axis=-1)


def plot_kspace_lines_memory(imgs, title=None):
    # Visualise images
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,len(imgs), figsize=(10,5), dpi=500)
    for idx, acq_data in enumerate(imgs):
        ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
        ky_index_max = np.max(ky_index)
        sampling_mask = np.zeros((ky_index_max+1,ky_index_max+1))
        sampling_mask[ky_index,:] = 1
        ax[idx].imshow(sampling_mask, vmin=0, vmax=1, cmap="gray_r")
        if title is not None and len(title) == len(imgs):
            logger.info(f"add title {title[idx]}")
            ax[idx].set_title(title[idx])

def best_subplot_grid(n, max_aspect=2.0):
    candidates = []
    for cols in range(1, int(np.ceil(np.sqrt(n))) + 1):
        rows = int(np.ceil(n / cols))
        if rows >= cols:
            blanks = rows * cols - n
            aspect = rows / cols
            candidates.append((aspect > max_aspect, blanks, aspect, rows, cols))
            # min() on a list of tuples returns one tuple from the list, chosen by lexicographical (left-to-right) comparison:
    _, _, _, rows, cols = min(candidates)
    return rows, cols

def plot_csms(csms, slice_num, coil_num, suptitle=None, data_type="ai", tight_layout=False):
    """
    Plot several coil sensitivity maps (CSM) for a given slice.

    Parameters:
    -----------
    csms: dict
        A dictionary containing the coil sensitivity maps to be plotted.
    slice_num: int
        The slice number to be plotted.
    coil_num: int
        The coil number to be plotted.
    suptitle: str, optional
        The title for the entire figure. Default is an empty string.
    data_type: str, optional
        The type of data to be plotted. Default is "ai".
    tight_layout: bool, optional
        Whether to use tight layout. Default is False.
    """
    rows, cols = best_subplot_grid(len(csms))
    
    fig, ax = plt.subplots(rows, cols, dpi=500)
    axes = ax.ravel()

    i = 0
    for k,v in csms.items():
        csm = v[data_type]
        cmsa = csm.as_array()
        axes[i].imshow(np.abs(cmsa[coil_num,slice_num,:,:]))
        axes[i].set_title(f"{k}")
        axes[i].axes.get_xaxis().set_ticks([])
        axes[i].axes.get_yaxis().set_ticks([])
        i += 1

    dtitle = f"Coil {coil_num}, Slice {slice_num}, "
    
    fig.suptitle(f"{dtitle} {suptitle}")
    if tight_layout:
        fig.tight_layout()

def plot_csm(csm, slice_num, suptitle="", tight_layout=False):
    """
    Plot the coil sensitivity maps (CSM) for a given slice.

    Parameters:
    -----------
    csm: CoilSensitivityData
        The coil sensitivity maps to be plotted.
    slice_num: int
        The slice number to be plotted.
    suptitle: str, optional
        The title for the entire figure. Default is an empty string.
    tight_layout: bool, optional
        Whether to use tight layout. Default is False.
    """
    cmsa = csm.as_array()
    num_coils = cmsa.shape[0]
    rows, cols = best_subplot_grid(num_coils)
    fig, ax = plt.subplots(rows, cols, dpi=500)
    
    for i in range(rows):
        for j in range(cols):
            idx = i + rows * j
            if idx < num_coils:
                ax[i][j].imshow(np.abs(cmsa[idx,slice_num,:,:]))
                ax[i][j].set_title(f"Coil {idx}")
    
    fig.suptitle(suptitle)
    if tight_layout:
        fig.tight_layout()

def rmse(input_im, ground_truth, border=20):
    """"Calculates the rmse between the input image and the ground truth, ignoring a border of specified width.
    
    Parameters:
    -----------
    input_im : np.ndarray
        The input image.
    ground_truth : np.ndarray
        The ground truth image.
    border : int, optional
        The width of the border to ignore when calculating the RMSE. Default is 20.

    Returns:
    --------
    float
        The RMSE between the input image and the ground truth, ignoring the specified border.
    """
    if np.shape(input_im)!=np.shape(ground_truth):
        raise ValueError('imgs have diff shape')
    mse = mean_squared_error(np.abs(input_im[border:-border,border:-border,border:-border]), \
        np.abs(ground_truth[border:-border,border:-border,border:-border]))
    return math.sqrt(mse)



def get_AcquisitionModel_CSM(acq_data, fully_sampled=None, smoothness=100, s_nit=0, s_ckhs=1):
    '''Given an AcquisitionData, returns an AcquisitionModel and the CoilSensitivityData object
    
    Parameters:
    -----------
    acq_data: AcquisitionData
    fully_sampled: None or list/tuple of ints, default None.
        If specified, the integers in the list should specify a contiguous range of acquisition
        k-space lines that are fully sampled
    s_nit: int positive, default 0. Number of smoothing iterations to suppress noise
    s_khs: int, default 1. Convolution kernel half size 
    '''
    if fully_sampled is None:
        acq_data_us = acq_data
    else:
        always_sampled = fully_sampled
        centre_line = int((max(acq_data.get_ISMRMRD_info('kspace_encode_step_1')) + \
                       min(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))) // 2 )
        always_sampled = [el + centre_line for el in fully_sampled]
        us_index = []
        for acq_idx, ky_idx in enumerate(acq_data.get_ISMRMRD_info('kspace_encode_step_1')):
            if ky_idx in always_sampled:
                us_index.append(acq_idx)
        acq_data_us = acq_data.get_subset(us_index)
        logger.info(f"CSM set up from centre fully sampled lines")


    
    csm = CoilSensitivityData()
    csm.smoothing_iterations = s_nit
    csm.conv_kernel_halfsize = s_ckhs
    logger.info(f"smoothing iterations {csm.smoothing_iterations}")
    logger.info(f"convolution kernel half size {csm.conv_kernel_halfsize}")
    csm.calculate(acq_data_us)
    
    E = AcquisitionModel(acqs=acq_data, imgs=csm)
    E.set_coil_sensitivity_maps(csm)
    
    # Use the result of the inverse as our starting point
    # logger.info(f"Creating inverse")
    inverse = E.inverse(acq_data)

    logger.info(f"Created inverse")

    inverse *= 0 
    E = AcquisitionModel(acqs=acq_data, imgs=inverse)
    logger.info(f"Created E")
    E.set_coil_sensitivity_maps(csm)

    # cache the norm of the operator
    import types
    E.calculate_norm = E.norm
    E._norm = None
    def norm(self):
        if self._norm is None:
            self._norm = self.calculate_norm()
        return self._norm
    E.norm = types.MethodType(norm, E)

    return E, csm

#%%
def save_ImageData_as_nii(recon, output_dir='/output/nii'):
    from sirf.Reg import NiftiImageData3D
    from sirf.Gadgetron import ImageData

    # cast to int16 range
    pixel_data = recon.abs()
    pixel_data = pixel_data / pixel_data.max() * np.iinfo(np.uint16).max

    niisolution = NiftiImageData3D(pixel_data)

    os.makedirs(output_dir, exist_ok=True)
    ofname = os.path.join(output_dir, f'solution.nii')
    niisolution.write(ofname)
    logger.info(f"Solution written to {ofname} {os.path.exists(ofname)}")

# Create undersampled data
def create_gaussian_variable_density_kspace_data(acq_data, acceleration_factor=3.5):
    
    ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))

    nky = int(np.max(ky_index)+1)
    ky_index_gauss = gaussian_variable_density_samples([1,int(len(ky_index)/acceleration_factor)], -nky//2, nky//2-1, nky*0.8, range(-13, 13)) + nky//2
    
    us_index = []
    for acq_idx, ky_idx in enumerate(acq_data.get_ISMRMRD_info('kspace_encode_step_1')):
        if ky_idx in ky_index_gauss:
            us_index.append(acq_idx)
    acq_data_us = acq_data.get_subset(us_index)

    return acq_data_us