#%%
import subprocess
import os
import logging
import numpy as np
import importlib
import matplotlib.pyplot as plt
import glob

#### StGeorges Utils
import sys
sys.path.append("/home/jovyan/work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon/")
import stgeorges_utils
from stgeorges_utils import change_ismrmrd, to_dicom_folder, LogfileCallback
from stgeorges_utils import plot_kspace_lines_memory
from stgeorges_utils import rmse
from stgeorges_utils import get_AcquisitionModel_CSM
from stgeorges_utils import save_ImageData_as_nii
from stgeorges_utils import create_gaussian_variable_density_kspace_data

# SIRF
from sirf.Gadgetron import AcquisitionData, ImageData
from sirf.Gadgetron import AcquisitionModel
from sirf.Gadgetron import AcquisitionDataProcessor
from sirf.Gadgetron import CartesianGRAPPAReconstructor, FullySampledReconstructor
from sirf.Gadgetron import CoilSensitivityData
from sirf.Gadgetron import preprocess_acquisition_data

# CIL
from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.algorithms import FISTA, CGLS, GD
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.framework import DataContainer as cilDataContainer
from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities.callbacks import ProgressCallback, TextProgressCallback
from cil.optimisation.utilities.callbacks import Callback 
from cil.optimisation.utilities.callbacks import EarlyStoppingObjectiveValue
from cil.utilities.display import show2D
import tempfile

from cil.optimisation.functions import L1Sparsity
from cil.optimisation.operators import WaveletOperator

#%%

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# CONFIGURATION
create_undersampled_data = True
acceleration_factor = 3

# DATA IMPORT AND PREPROCESSING
command = "siemens_to_ismrmrd"

# input_files = ["/home/jovyan/work/person2/person2/meas_MID00595_FID133061_pd_tse_fs_cor_uflex_no_spine.dat",
#                "/home/jovyan/work/person2/person2/meas_MID00598_FID133064_pd_tse_fs_cor_uflex_no_spine.dat"]

input_files = glob.glob("/input/*.dat")
if len(input_files) == 0:
    exit(1, "No input file found.")
# if len(input_files) > 1:
#     exit(2, "More than one input file found.")


proc_dir = tempfile.mkdtemp(prefix="stgeorges_proc_")

output_files = []
for fname in input_files:
    logger.info(f"Processing file {fname}...")
    file_in = fname
    file_out = os.path.join(proc_dir, os.path.basename(file_in).replace(".dat", ".h5"))
    if os.path.exists(file_out):
        logger.warning(f"Output file {file_out} already exists. Removing it.")
        os.remove(file_out)
    
    out = subprocess.run(
        [command, "-f", file_in, "-o", file_out, "-z", "2", "-M"],
        capture_output=False,
        text=False,
    )
    
    logger.info(out.stdout)
    logger.error(out.stderr)
    
    # Change ISMRMRD file if needed
    file_out_mod = file_out.replace(".h5", "_mod.h5")
    change_ismrmrd(file_out, file_out_mod)
    logger.info(f"Modified ISMRMRD file saved as {file_out_mod}")
    output_files.append(file_out_mod)

#%%
# store all data in a dictionary
am = {}
#%%
nits = 0
ckhs = 1
dpars = (nits, ckhs)
for f in output_files:
    logger.info(f"Processing {os.path.basename(f)}")
    acq_data = AcquisitionData(f)
    acq_data = preprocess_acquisition_data(acq_data)

    ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
    ky_index_max = np.max(ky_index)
    ky_index_min = np.min(ky_index)


    if np.array_equal(np.arange(ky_index_min, ky_index_max + 1), ky_index):
        # Fully sampled dataset
        logger.info (f"Fully sampled data")
        which = "fs"
        am[which] = {'am': None, 'csm': None, 'data': acq_data, 'filename': os.path.basename(f)}
        from sirf.Gadgetron import FullySampledReconstructor
        
        acq_data = am[which]['data']
        
        recon = FullySampledReconstructor()
        recon.set_input(acq_data)
        recon.process()
        # store the reconstruction
        am[which]['recon'] = recon.get_output()
        # save_ImageData_as_nii(x_recon, prefix=which)
        # show2D(np.squeeze(np.abs(x_recon.asarray()[11,110:410, 80:380])).T)
        if create_undersampled_data:
            print (f"Creating under sampled data")
            importlib.reload(stgeorges_utils)
            
            acq_data_us = create_gaussian_variable_density_kspace_data(
                am['fs']['data'],
                acceleration_factor=acceleration_factor
            )
            am['us'] = {'data': acq_data_us}
            # Get the Coil Sensitivity Maps (CSM) from the under sampled data
            E, csm = get_AcquisitionModel_CSM(am['us']['data'], 
                fully_sampled=range(-13,13), 
                s_ckhs=ckhs, 
                s_nit=nits)
            am['us']['am'] = E
            am['us']['csm'] = csm
            am['us']['filename'] = os.path.basename(f)
    else:
        # sparse acquisition
        print ("Processing AI dataset")
        which = 'ai'
        am[which] = {'am': None, 'csm': None, 'data':  acq_data}
        # Get the Coil Sensitivity Maps (CSM) from the sparse (AI) data
        E, csm = get_AcquisitionModel_CSM(am['ai']['data'], 
                                          fully_sampled=range(-13,13),
                                          s_ckhs=ckhs, 
                                          s_nit=nits)
        am[which]['am'] = E
        am[which]['csm'] = csm
        am[which]['filename'] = os.path.basename(f)


# Clear temp dir
# delete proc_dir
if os.path.exists(proc_dir):
    logger.info(f"Deleting temporary processing directory {proc_dir}...")
    try:
        for fname in os.listdir(proc_dir):
            os.remove(os.path.join(proc_dir, fname))
        os.rmdir(proc_dir)
        logger.info("Temporary processing directory deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting temporary processing directory: {e}")

#%%
# DATA is not normalised. Print the sum of the absolute values of the data for each dataset to check the scale
sum_ai = np.sqrt(np.sum(np.abs(am['ai']['data'].asarray().ravel())))
sum_us = np.sqrt(np.sum(np.abs(am['us']['data'].asarray().ravel())))
sum_fs = np.sqrt(np.sum(np.abs(am['fs']['data'].asarray().ravel())))
logger.info(f"Sum of AI data: {sum_ai}")
logger.info(f"Sum of Undersampled data: {sum_us}")
logger.info(f"Sum of Fully Sampled data: {sum_fs}")

#%%
from cil.optimisation.utilities.callbacks import Callback, ProgressCallback
class LogAll(Callback):
    def __init__(self, interval=1, rmse_reference=None, rmse_border=20, plot=False):
        self.interval = interval
        self.iteration = []
        self.iterates = []
        self.rmse = []
        self.reference = rmse_reference
        self.rmse_border = rmse_border
        self.plot = plot

    def __call__(self, solver):
        if solver.iteration % self.interval == 0:
            self.iteration.append(solver.iteration)
            dslice = np.squeeze(
                    np.abs(solver.solution.asarray()[11,110:410, 80:380])
                    ).T
            self.iterates.append(dslice)
            if self.reference is not None:
                try:
                    drmse = rmse(solver.solution.asarray(), 
                                             self.reference.asarray(), 
                                             border=self.rmse_border)
                except ValueError as ve:
                    logger.warning(f"ValueError encountered in RMSE calculation: {ve}. Trying cropping...")
                    drmse = rmse(solver.solution.asarray(), 
                                             self.reference.asarray()[:,3:-3, :], 
                                             border=self.rmse_border)
                self.rmse.append(
                    drmse
                )
            if self.plot:
                show2D(dslice)




#%%
# define alphas for TV regularisation. Apparently the scale of the data
# in the AI and FullySampled data, hence undersampled, are not the same, 
# leading to a complete different scale of the reconstructed image and 
# reconstruction parameter

# TV
am['fs']['alpha'] = 8.5e-06
am['ai']['alpha'] = 3e-1
am['us']['alpha'] = 8.5e-6

# Wavelet sparsity regularisation
am['fs']['alpha'] = 8.5e-06
am['ai']['alpha'] = 3e-1
am['us']['alpha'] = 3e-6


#%%
# Define our objective/loss function as least squares between Ex and y
num_iterations = 2
start_new = True
from SIRFWavelets import FunctionOfAbs, SIRFWaveletOperator

for k,v in am.items():
    if k in ['fs']:
        logger.info(f"Skipping {k}")
        continue
    logger.info(f"Processing {k}")
    E = v['am']
    acq_data = v['data']
    x_init = E.inverse(acq_data)
    # x_init *= 0
    f = LeastSquares(E, acq_data, c=1)

    alpha = v['alpha']
    # alpha = 1
    TV = FGP_TV(alpha=alpha, nonnegativity=False, device='cpu', max_iteration=500)
    # # Tau in FISTA is 
    # tau = 0.99*2/am['us']['algo'].f.L
    # # alpha * tau ~ 2.e-5

    wop = SIRFWaveletOperator(domain_geometry=x_init, level=None, wname="db2")
    l1s = L1Sparsity(wop)
    G = alpha * FunctionOfAbs(l1s)
    
    logger.info(f"LS {f(x_init)}")
    logger.info(f"Regulariser {G(x_init)}")
    # add logger callback to FISTA


    # Set up FISTA
    if start_new:
        fista = FISTA(initial=x_init, f=f, g=G)
        fista.update_objective_interval = 1
        v['algo'] = fista
    else:
        fista = v['algo']
    # Run FISTA for least squares
    # v['callback'] = LogAll(rmse_reference=x_init, plot=True)

    fista.run(num_iterations, callbacks=[ProgressCallback(), 
                                        #  v['callback'], 
                                         EarlyStoppingObjectiveValue(threshold=1e-6)])
    



    # SAVE reconstruction to file
    prefix = k
    save_ImageData_as_nii(fista.solution, prefix=prefix, output_dir='/output/nii')
    import json
    try:
        with open(f'/output/nii/{prefix}/{prefix}-config.json', 'w') as f:
            json.dump({'filename': v['filename'], 
                    'iterations': fista.iterations, 
                    'loss': fista.loss,
                    'type': prefix,
                    'alpha': alpha,
                    'csm_nits': nits,
                    'csm_ckhs': ckhs}, f)
    except Exception as e:
        logger.error(f"Error saving config file: {e}")

# Clear temp dir
# delete proc_dir
if os.path.exists(proc_dir):
    logger.info(f"Deleting temporary processing directory {proc_dir}...")
    try:
        for fname in os.listdir(proc_dir):
            os.remove(os.path.join(proc_dir, fname))
        os.rmdir(proc_dir)
        logger.info("Temporary processing directory deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting temporary processing directory: {e}")
