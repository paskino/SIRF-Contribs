import ismrmrd
import os
from sirf.Gadgetron import ImageData
from cil.optimisation.utilities.callbacks import Callback

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

    


# import pysnooper
# @pysnooper.snoop()
def _to_dicom_folder(
    data: np.ndarray,
    foldername: str | Path,
    filename_prefix: str = "sirf",
    series_uid: str | None = None,
    series_description: str | None = None,
    resolution: float = 1.0,
    **kwargs
) -> None:
    """Write image data to DICOM files in a folder.

    The data is always saved in a multi-frame DICOM files.

    Parameters
    ----------
    foldername
        Path to folder for DICOM files.
    filename_prefix
        Prefix for DICOM filenames.
    series_uid
        Series Instance UID to be used in the DICOM files. If None, a new UID will be generated.
    series_description
        String to be saved as the series description in the DICOM files.
    resolution
        Spacing between slices in mm.
    """
    print(
        f"Writing dicome files with prefix {filename_prefix} into folder {foldername} "
    )
    if not isinstance(foldername, Path):
        foldername = Path(foldername)
    foldername.mkdir(parents=True, exist_ok=True)

    acquisition_type = "2D"
    frame_dimension = next(
        (i for i in range(-3, -len(data.shape) - 1, -1) if data.shape[i] > 1), -3
    )
    number_of_frames = data.shape[frame_dimension]
    dcm_idata = data.swapaxes(frame_dimension, -3)

    # Metadata
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Dataset
    dataset = pydicom.Dataset()
    dataset.file_meta = file_meta
    studyInstanceUID = pydicom.uid.generate_uid()

    dataset.PatientName = "Unknown"
    dataset.PatientID = "Unknown"
    dataset.PatientSex = "O"

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    dataset.SeriesDate = timestamp.strftime("%Y%m%d")
    dataset.SeriesTime = timestamp.strftime("%H%M%S.%f")
    if series_description:
        dataset.SeriesDescription = series_description
        dataset.ProtocolName = series_description
    dataset.SeriesInstanceUID = series_uid if series_uid else pydicom.uid.generate_uid()

    dataset.PatientPosition = "HFS" if not 'patient_position' in kwargs else kwargs['patient_position']

    for file_index, other in enumerate(np.ndindex(dcm_idata.shape[:-3])):
        dcm_file_idata = dcm_idata[(*other, slice(None), slice(None), slice(None))]

        dataset.MRAcquisitionType = acquisition_type
        dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

        # (frames, rows, columns) for multi-frame grayscale data
        pixel_data = np.abs(dcm_file_idata)
        pixel_data = pixel_data / pixel_data.max() * (2**16 - 1)
        pixel_data = np.swapaxes(pixel_data, -1, -2)

        for frame in range(number_of_frames):
            image_position_patient = (
                np.asarray([0, 0, 0]) + np.asarray([1, 0, 0]) * resolution * file_index * frame
            )
            dataset.ImagePositionPatient = image_position_patient.tolist()

            # 'MONOCHROME2' means smallest value is black, largest value is white
            set_pixel_data(
                ds=dataset,
                arr=pixel_data[frame, ...].astype(np.uint16),
                photometric_interpretation="MONOCHROME2",
                bits_stored=16,
            )
            
            # Ensure required fields are set (set_pixel_data may have cleared them)
            dataset.SOPInstanceUID = pydicom.uid.generate_uid()
            dataset.PatientName = "Unknown"
            dataset.PatientID = "Unknown"
            dataset.StudyInstanceUID = studyInstanceUID
            
            dataset.save_as(
                foldername
                / f"{filename_prefix}_{str(np.prod(file_index) * number_of_frames + frame).zfill(4)}.dcm",
                enforce_file_format=True,
            )

def to_dicom_folder_simple(
    data: ImageData,
    foldername: str | Path,
    filename_prefix: str = "sirf",
    series_uid: str | None = None,
    series_description: str | None = None,
    resolution: float = 1.0,
    **kwargs
) -> None:
    """Write image data to DICOM files in a folder.

    The data is always saved in a multi-frame DICOM files.

    Parameters
    ----------
    foldername
        Path to folder for DICOM files.
    filename_prefix
        Prefix for DICOM filenames.
    series_uid
        Series Instance UID to be used in the DICOM files. If None, a new UID will be generated.
    series_description
        String to be saved as the series description in the DICOM files.
    resolution
        Spacing between slices in mm.
    """
    print(
        f"Writing dicome files with prefix {filename_prefix} into folder {foldername} "
    )
    if not isinstance(foldername, Path):
        foldername = Path(foldername)
    foldername.mkdir(parents=True, exist_ok=True)

    dcm_data = data.as_array()

    acquisition_type = "2D"
    frame_dimension = -3
    number_of_frames = dcm_data.shape[frame_dimension]
    
    # Metadata
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Dataset
    dataset = pydicom.Dataset()
    dataset.file_meta = file_meta
    studyInstanceUID = pydicom.uid.generate_uid()

    dataset.PatientName = "Unknown"
    dataset.PatientID = "Unknown"
    dataset.PatientSex = "O"

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    dataset.SeriesDate = timestamp.strftime("%Y%m%d")
    dataset.SeriesTime = timestamp.strftime("%H%M%S.%f")
    if series_description:
        dataset.SeriesDescription = series_description
        dataset.ProtocolName = series_description
    dataset.SeriesInstanceUID = series_uid if series_uid else pydicom.uid.generate_uid()

    dataset.PatientPosition = "HFS" if not 'patient_position' in kwargs else kwargs['patient_position']

    dataset.MRAcquisitionType = acquisition_type
    dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence()

    # (frames, rows, columns) for multi-frame grayscale data
    pixel_data = np.abs(dcm_data)
    pixel_data = pixel_data / pixel_data.max() * (2**16 - 1)
    pixel_data = np.swapaxes(pixel_data, -1, -2)

    for frame in range(number_of_frames):
        image_position_patient = (
            np.asarray([0, 0, 0]) + np.asarray([1, 0, 0]) * resolution * frame
        )
        dataset.ImagePositionPatient = image_position_patient.tolist()

        # 'MONOCHROME2' means smallest value is black, largest value is white
        set_pixel_data(
            ds=dataset,
            arr=pixel_data[frame, ...].astype(np.uint16),
            photometric_interpretation="MONOCHROME2",
            bits_stored=16,
        )
        
        # Ensure required fields are set (set_pixel_data may have cleared them)
        dataset.SOPInstanceUID = pydicom.uid.generate_uid()
        dataset.PatientName = "Unknown"
        dataset.PatientID = "Unknown"
        dataset.StudyInstanceUID = studyInstanceUID
        
        file_index = frame  

        dataset.save_as(
            foldername
            # / f"{filename_prefix}_{str(np.prod(file_index) * number_of_frames + frame).zfill(4)}.dcm
            / f"{filename_prefix}_{str(frame).zfill(4)}.dcm",
            enforce_file_format=True,
        )


def save_numpy_to_dicom_series(
    data: np.ndarray,
    foldername: str | Path,
    filename_prefix: str = "slice",
    series_uid: str | None = None,
    study_uid: str | None = None,
    series_description: str | None = None,
    frame_axis: int = -3,
    stack_index: int = 0,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_spacing: float = 1.0,
    image_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    image_orientation_patient: tuple[float, float, float, float, float, float] =
    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    patient_position: str = "HFS",
) -> None:
    """Save a NumPy array as a DICOM series from scratch.

    The selected volume is written as one file per slice with monotonic geometry.

    Accepted inputs:
    - 2D: (rows, cols) -> one-slice series
    - 3D: frame axis selected by ``frame_axis``
    - 4D+: leading dimensions are treated as stacks, selected by ``stack_index``
    """
    if not isinstance(foldername, Path):
        foldername = Path(foldername)
    foldername.mkdir(parents=True, exist_ok=True)

    volume = np.asarray(data)
    if np.iscomplexobj(volume):
        volume = np.abs(volume)

    if volume.ndim < 2:
        raise ValueError("data must be at least 2D")

    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    else:
        volume = np.moveaxis(volume, frame_axis, 0)
        if volume.ndim > 3:
            num_stacks = int(np.prod(volume.shape[1:-2]))
            if stack_index < 0 or stack_index >= num_stacks:
                raise ValueError(
                    f"stack_index {stack_index} out of range for {num_stacks} stack(s)"
                )
            volume = volume.reshape((volume.shape[0], num_stacks, volume.shape[-2], volume.shape[-1]))
            volume = volume[:, stack_index, :, :]

    # Normalize to 16-bit grayscale for broad viewer compatibility.
    volume = volume.astype(np.float32)
    max_val = float(np.max(volume))
    if max_val > 0.0:
        volume = volume / max_val
    volume_u16 = np.clip(np.round(volume * 65535.0), 0, 65535).astype(np.uint16)

    n_slices, rows, cols = volume_u16.shape

    series_instance_uid = series_uid if series_uid else pydicom.uid.generate_uid()
    study_instance_uid = study_uid if study_uid else pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    row_cosines = np.asarray(image_orientation_patient[:3], dtype=np.float64)
    col_cosines = np.asarray(image_orientation_patient[3:], dtype=np.float64)
    slice_normal = np.cross(row_cosines, col_cosines)
    normal_norm = np.linalg.norm(slice_normal)
    if normal_norm == 0.0:
        raise ValueError("image_orientation_patient must define two non-collinear direction vectors")
    slice_normal = slice_normal / normal_norm
    origin = np.asarray(image_origin, dtype=np.float64)

    for slice_idx in range(n_slices):
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

        ds = pydicom.dataset.FileDataset(
            "",
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.SOPClassUID = pydicom.uid.MRImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "MR"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
        ds.PatientName = "Unknown"
        ds.PatientID = "Unknown"
        ds.PatientSex = "O"
        ds.PatientPosition = patient_position

        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.AcquisitionNumber = 1
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.StudyDate = timestamp.strftime("%Y%m%d")
        ds.StudyTime = timestamp.strftime("%H%M%S.%f")
        ds.SeriesDate = ds.StudyDate
        ds.SeriesTime = ds.StudyTime
        if series_description:
            ds.SeriesDescription = series_description
            ds.ProtocolName = series_description

        ds.Rows = int(rows)
        ds.Columns = int(cols)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        ds.ImageOrientationPatient = [float(v) for v in image_orientation_patient]
        position = origin + slice_normal * float(slice_idx) * float(slice_spacing)
        ds.ImagePositionPatient = [float(v) for v in position]
        ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        ds.SliceThickness = float(slice_spacing)
        ds.SpacingBetweenSlices = float(slice_spacing)
        ds.ImagesInAcquisition = int(n_slices)
        ds.InstanceNumber = int(slice_idx + 1)
        ds.InStackPositionNumber = int(slice_idx + 1)
        ds.StackID = "1"
        ds.TemporalPositionIdentifier = 1
        ds.NumberOfTemporalPositions = 1
        ds.SliceLocation = float(slice_idx) * float(slice_spacing)

        ds.PixelData = volume_u16[slice_idx].tobytes()

        output_path = foldername / f"{filename_prefix}_{slice_idx:04d}.dcm"
        ds.save_as(output_path, enforce_file_format=True)


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


def plot_kspace_lines_memory(imgs, ky_index_max):
    # Visualise images
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,len(imgs))
    for idx, acq_data in enumerate(imgs):
        ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
        ky_index_max = np.max(ky_index)
        sampling_mask = np.zeros((ky_index_max+1,ky_index_max+1))
        sampling_mask[ky_index,:] = 1
        ax[idx].imshow(sampling_mask, vmin=0, vmax=1)
     