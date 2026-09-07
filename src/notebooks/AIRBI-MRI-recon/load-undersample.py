#%%
import subprocess
import os
import logging
import numpy as np

import sys
sys.path.append("/home/jovyan/work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon/")
from stgeorges_utils import change_ismrmrd, to_dicom_folder, LogfileCallback

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
import tempfile

from cil.optimisation.functions import L1Sparsity
from cil.optimisation.operators import WaveletOperator
# from AbsFunction import FunctionOfAbs

#%%

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

command = "siemens_to_ismrmrd"

# input_files = ["/home/jovyan/work/person2/person2/meas_MID00595_FID133061_pd_tse_fs_cor_uflex_no_spine.dat",
#                "/home/jovyan/work/person2/person2/meas_MID00598_FID133064_pd_tse_fs_cor_uflex_no_spine.dat"]

input_files = ["/input/meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat",
              "/input/meas_MID00619_FID129157_AI_RECON_SEQD_512_GF4_AX_RL.dat"]

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

# Fully sampled dataset
ad_fs = AcquisitionData(output_files[0])
ad_fs = preprocess_acquisition_data(ad_fs)

#%%
# AI protocol
ad_ai = AcquisitionData(output_files[1])
ad_ai = preprocess_acquisition_data(ad_ai)

#%%
from stgeorges_utils import plot_kspace_lines_memory
plot_kspace_lines_memory([ad_fs, ad_ai])
# %%

def get_AcquisitionModel_CSM(acq_data, smoothness=100):
    '''Given an AcquisitionData, returns an AcquisitionModel and the CoilSensitivityData object'''
    csm = CoilSensitivityData()
    csm.smoothness = smoothness
    csm.calculate(acq_data)
    logger.info(f"CSM done")

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
am = {}
E, csm = get_AcquisitionModel_CSM(ad_fs)
am['fs'] = {'am': E, 'csm': csm}
am['fs']['data'] = ad_fs
# logger.info(f"Norm of E: {norm}")
# %%
E, csm = get_AcquisitionModel_CSM(ad_ai)
am['ai'] = {'am': E, 'csm': csm}
# am['ai']['norm'] = am['ai']['am'].norm()
am['ai']['data'] = ad_ai
# logger.info(f"Norm of E (AI): {am['ai']['norm']}")
# %%
# Undersample fully sample data with Gaussian variable density 

# %%
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
acq_data_us = create_gaussian_variable_density_kspace_data(am['fs']['data'])
plot_kspace_lines_memory([am['fs']['data'], am['ai']['data'], acq_data_us], None)
                        #  np.max(ky_index))

# %%
E, csm = get_AcquisitionModel_CSM(acq_data_us)
am['us'] = {'am': E, 'csm': csm}
am['us']['data'] = acq_data_us
#%%

from cil.optimisation.utilities.callbacks import Callback, ProgressCallback
class LogAll(Callback):
    def __init__(self, interval=1):
        self.interval = interval
        self.iteration = []
        self.iterates = []

    def __call__(self, solver):
        if solver.iteration % self.interval == 0:
            self.iteration.append(solver.iteration)
            self.iterates.append(solver.solution.copy())

logall = LogAll()



#%%
# Define our objective/loss function as least squares between Ex and y

for k,v in am.items():
    logger.info(f"Processing {k}")
    E = v['am']
    acq_data = v['data']
    x_init = E.inverse(acq_data)
    x_init *= 0
    f = LeastSquares(E, acq_data, c=1)

    alpha = 0.3
    TV = FGP_TV(alpha=alpha, nonnegativity=False, device='cpu')
    G = TV

    # add logger callback to FISTA


    # Set up FISTA
    fista = FISTA(initial=x_init, f=f, g=G)
    fista.update_objective_interval = 1
    v['algo'] = fista

    # Run FISTA for least squares
    num_iterations = 10
    fista.run(num_iterations, callbacks=[ProgressCallback(), logall])
    

# %%
from cil.utilities.display import show2D

show2D(
    [np.abs(v['recon'].asarray()) for k,v in am.items()],
    title=[k for k in am.keys()],
    num_cols=3,
    slice_list=(0,6)

)
#%%
import matplotlib.pyplot as plt
plt.plot(fista.loss)
# %%
