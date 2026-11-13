import ismrmrd
import os
from sirf.Gadgetron import ImageData
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
    fig, ax = plt.subplots(1,len(imgs))
    for idx, acq_data in enumerate(imgs):
        ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
        ky_index_max = np.max(ky_index)
        sampling_mask = np.zeros((ky_index_max+1,ky_index_max+1))
        sampling_mask[ky_index,:] = 1
        ax[idx].imshow(sampling_mask, vmin=0, vmax=1)
        if title is not None and len(title) == len(imgs):
            print(f"add title {title[idx]}")
            ax[idx].set_title(title[idx])


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
