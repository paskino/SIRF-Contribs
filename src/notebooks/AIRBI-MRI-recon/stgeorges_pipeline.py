#%%
import subprocess
import os
import logging
import numpy as np

import sys
sys.path.append("/home/jovyan/work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon/")
from stgeorges_utils import change_ismrmrd, LogfileCallback
from stgeorges_utils import plot_kspace_lines_memory

from sirf.Gadgetron import AcquisitionData, ImageData
from sirf.Gadgetron import AcquisitionModel
from sirf.Gadgetron import AcquisitionDataProcessor
from sirf.Gadgetron import CartesianGRAPPAReconstructor, FullySampledReconstructor
from sirf.Gadgetron import CoilSensitivityData
from sirf.Gadgetron import preprocess_acquisition_data

from cil.optimisation.functions import LeastSquares
from cil.optimisation.functions import ZeroFunction
from cil.optimisation.algorithms import FISTA, CGLS, GD
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.framework import DataContainer as cilDataContainer
from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities.callbacks import ProgressCallback, TextProgressCallback
from cil.utilities.display import show2D
import tempfile

from cil.optimisation.functions import L1Sparsity
from cil.optimisation.operators import WaveletOperator
import glob
from stgeorges_utils import rmse
# from AbsFunction import FunctionOfAbs
def save_ImageData_as_nii(recon, prefix, output_dir='/output/nii'):
    from sirf.Reg import NiftiImageData3D
    from sirf.Gadgetron import ImageData
    niisolution = NiftiImageData3D(recon)
    os.makedirs(output_dir, exist_ok=True)
    niisolution.write(f'/output/nii/{prefix}-solution.nii')
    print(f"Solution written to /output/nii/{prefix}-solution.nii {os.path.exists(f'/output/nii/{prefix}-solution.nii')}")
#%%

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

command = "siemens_to_ismrmrd"

# input_files = ["/home/jovyan/work/person2/person2/meas_MID00595_FID133061_pd_tse_fs_cor_uflex_no_spine.dat",
#                "/home/jovyan/work/person2/person2/meas_MID00598_FID133064_pd_tse_fs_cor_uflex_no_spine.dat"]

input_files = glob.glob("/input/*.dat")
if len(input_files) == 0:
    exit(1, "No input file found.")
if len(input_files) > 1:
    exit(2, "More than one input file found.")


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
# if data is fully sampled.
acq_data = AcquisitionData(output_files[0])
acq_data = preprocess_acquisition_data(acq_data)

ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))
ky_index_max = np.max(ky_index)
ky_index_min = np.min(ky_index)
#%%

if np.array_equal(np.arange(ky_index_min, ky_index_max + 1), ky_index):
# Fully sampled dataset
    am['fs'] = {'am': None, 'csm': None, 'data': acq_data}
    from sirf.Gadgetron import FullySampledReconstructor
    which = "fs"
    acq_data = am[which]['data']
    recon = FullySampledReconstructor()
    recon.set_input(acq_data)
    recon.process()
    x_recon = recon.get_output()
    save_ImageData_as_nii(x_recon, prefix=which)
    exit(0)
#%%

#%%
# AI protocol

# ad_ai /= sum_ai
print(f"Sum of AI data: {sum_ai}")
am['ai'] = {'am': None, 'csm': None, 'data':  ad_ai}

# %%

def get_AcquisitionModel_CSM(acq_data, smoothness=100, fully_sampled=True):
    '''Given an AcquisitionData, returns an AcquisitionModel and the CoilSensitivityData object'''
    if fully_sampled:
        acq_data_us = acq_data
    else:
        always_sampled = fully_sampled
        us_index = []
        for acq_idx, ky_idx in enumerate(acq_data.get_ISMRMRD_info('kspace_encode_step_1')):
            if ky_idx in always_sampled:
                us_index.append(acq_idx)
        acq_data_us = acq_data.get_subset(us_index)

    
    csm = CoilSensitivityData()
    csm.smoothness = smoothness
    csm.calculate(acq_data_us)
    logger.info(f"CSM set up from centre fully sampled lines")

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
# Get the Coil Sensitivity Maps (CSM) from the fully sampled data
E, csm_fs = get_AcquisitionModel_CSM(am['fs']['data'], fully_sampled=True)
am['fs']['am'] = E
am['fs']['csm'] = csm_fs
#%%
E, csm = get_AcquisitionModel_CSM(am['ai']['data'], fully_sampled=range(-13,13))
am['ai']['am'] = E
am['ai']['csm'] = csm
# am['ai']['norm'] = am['ai']['am'].norm()

# logger.info(f"Norm of E (AI): {am['ai']['norm']}")
# %%
# Undersample fully sample data with Gaussian variable density 

# Create undersampled data
def create_gaussian_variable_density_kspace_data(acq_data, acceleration_factor=3.5):
    from stgeorges_utils import gaussian_variable_density_samples

    ky_index = np.unique(acq_data.get_ISMRMRD_info('kspace_encode_step_1'))

    nky = int(np.max(ky_index)+1)
    ky_index_gauss = gaussian_variable_density_samples([1,int(len(ky_index)/acceleration_factor)], -nky//2, nky//2-1, nky*0.8, range(-13, 13)) + nky//2
    
    us_index = []
    for acq_idx, ky_idx in enumerate(acq_data.get_ISMRMRD_info('kspace_encode_step_1')):
        if ky_idx in ky_index_gauss:
            us_index.append(acq_idx)
    acq_data_us = acq_data.get_subset(us_index)

    return acq_data_us
#%%
import stgeorges_utils
import importlib
importlib.reload(stgeorges_utils)
from stgeorges_utils import plot_kspace_lines_memory
acq_data_us = create_gaussian_variable_density_kspace_data(am['fs']['data'],
                                                           acceleration_factor=4)
am['us'] = {'data': acq_data_us}
#%%
plot_kspace_lines_memory([am['fs']['data'], am['ai']['data'], acq_data_us], None)
                        #  np.max(ky_index))

# %%
# sum_us = np.sum(np.abs(acq_data_us.asarray().ravel()))
# print(f"Sum of Undersampled data: {sum_us}")
# acq_data_us /= sum_us

E, csm = get_AcquisitionModel_CSM(acq_data_us, fully_sampled=range(-13,13))
am['us']['am'] = E
am['us']['csm'] = csm

#%%
# Data should be normalised now
sum_ai = np.sqrt(np.sum(np.abs(am['ai']['data'].asarray().ravel())))
sum_us = np.sqrt(np.sum(np.abs(am['us']['data'].asarray().ravel())))
sum_fs = np.sqrt(np.sum(np.abs(am['fs']['data'].asarray().ravel())))
print(f"Sum of AI data: {sum_ai}")
print(f"Sum of Undersampled data: {sum_us}")
print(f"Sum of Fully Sampled data: {sum_fs}")

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
num_iterations = 10
start_new = True
from SIRFWavelets import FunctionOfAbs, SIRFWaveletOperator

for k,v in am.items():
    if k in ['fs', 'us']:
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
    alpha = 3.0e-1
    wop = SIRFWaveletOperator(domain_geometry=x_init, level=None, wname="haar")
    l1s = L1Sparsity(wop)
    G = alpha * FunctionOfAbs(l1s)
    
    logger.info(f"LS {f(x_init)}")
    logger.info(f"TV {G(x_init)}")
    # add logger callback to FISTA


    # Set up FISTA
    if start_new:
        fista = FISTA(initial=x_init, f=f, g=G)
        fista.update_objective_interval = 1
        v['algo'] = fista
    else:
        fista = v['algo']
    # Run FISTA for least squares
    v['callback'] = LogAll(rmse_reference=x_recon, plot=True)

    fista.run(num_iterations, callbacks=[ProgressCallback()])




save_ImageData_as_nii(fista.solution, prefix=am.keys()[0], output_dir='/output/nii')

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
