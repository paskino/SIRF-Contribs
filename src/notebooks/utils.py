import matplotlib.pyplot as plt
import sirf.Gadgetron as pMR
import numpy as np
import os
import scipy.signal as sp_signal



# Plotting function

def plot_rpe_3d(dat, sl_idx, lbl, limits, title=None, size=None, save_dir="/home/jovyan/devel/mcir/"):
    if size is None:
        fig, ax = plt.subplots(2,len(dat), squeeze=False)
    else:
        fig, ax = plt.subplots(2,len(dat), squeeze=False, figsize=size)
        
    for ind in range(len(dat)):
        
        if limits[ind] is None:
            im = ax[0,ind].imshow(np.rot90(np.abs(dat[ind][:, sl_idx[0], :]), 1))
        else:
            im = ax[0,ind].imshow(np.rot90(np.abs(dat[ind][:, sl_idx[0], :]), 1), vmax=limits[ind])
            
        ax[0,ind].set_xticks([])
        ax[0,ind].set_yticks([])
        ax[0,ind].set_ylabel('Foot-Head')
        ax[0,ind].set_xlabel('Right-Left')
        ax[0,ind].set_title(lbl[ind])
        fig.colorbar(im, orientation = 'horizontal', location='bottom', shrink=0.5)
        
        if limits[ind] is None:
            im = ax[1,ind].imshow(np.rot90(np.abs(dat[ind][:, :, sl_idx[1]])))
        else:
            im = ax[1,ind].imshow(np.rot90(np.abs(dat[ind][:, :, sl_idx[1]])), vmax=limits[ind])

        ax[1,ind].set_xticks([])
        ax[1,ind].set_yticks([])
        ax[1,ind].set_ylabel('Anterior-Posterior')
        ax[1,ind].set_xlabel('Right-Left')
        ax[0,ind].set_title(lbl[ind])
        fig.colorbar(im, orientation = 'horizontal', location='bottom', shrink=0.5)
        
        if title is not None:
            fig.savefig(os.path.join(save_dir, title))


def plot_rpe_3d_simple(dat, sl_idx, lbl, fig_name=None, ax=None, cmap='gray', wspace=-.32, hspace=0.02):
    if ax is None:
        fig, ax = plt.subplots(2,len(dat), squeeze=True, figsize=(10,5), gridspec_kw={'wspace':wspace, 'hspace':hspace})
    for ind in range(len(dat)):
        ax[0,ind].imshow(np.rot90(np.abs(dat[ind][:, sl_idx[0], :]), 1), cmap=cmap)
        ax[0,ind].set_xticks([])
        ax[0,ind].set_yticks([])
        
        ax[0,ind].set_title(lbl[ind])
        
        sp = ax[1,ind].imshow(np.rot90(np.abs(dat[ind][:, :, sl_idx[1]])), cmap=cmap)
        ax[1,ind].set_xticks([])
        ax[1,ind].set_yticks([])
        
    # set ylabels
    ind = 0
    ax[0,ind].set_ylabel('Foot-Head')
    ax[1,ind].set_ylabel('Anterior-Posterior')
        
    # set xlabels:
    for ind in range (len(dat)):
        ax[1,ind].set_xlabel('Right-Left')

    plt.tight_layout()
    # plt.colorbar(sp, orientation='vertical', shrink=1, anchor=(0.0, 0.) )
    
    if fig_name is not None:
        plt.savefig('{}.png'.format(fig_name))

def plot_rpe_3d_simple_dir(dat, sl_idx, lbl, fig_name=None, ax=None, cmap='gray', wspace=-.32, hspace=0.02):
    if ax is None:
        fig, ax = plt.subplots(1,len(dat), squeeze=True, figsize=(10,5), gridspec_kw={'wspace':wspace, 'hspace':hspace})
    for ind in range(len(dat)):
        ax[ind].imshow(np.rot90(np.abs(dat[ind][:, :, sl_idx[ind]]), 1), cmap=cmap)
        ax[ind].set_xticks([])
        ax[ind].set_yticks([])
        
        ax[ind].set_title(lbl[ind])
        
    # set ylabels
    ind = 0
    ax[ind].set_ylabel('Foot-Head')
    ax[ind].set_ylabel('Anterior-Posterior')
        
    # set xlabels:
    for ind in range (len(dat)):
        ax[ind].set_xlabel('Right-Left')

    plt.tight_layout()
    # plt.colorbar(sp, orientation='vertical', shrink=1, anchor=(0.0, 0.) )
    
    if fig_name is not None:
        plt.savefig('{}.png'.format(fig_name))


def saveCallback(algo, prefix, save_dir, iteration, objective, solution):
    
        if iteration > 0:
            try:
                os.makedirs(save_dir)
            except FileExistsError:
                pass
            
            solution.write(os.path.join(save_dir, f'{prefix}_it_{iteration:003}.h5'))


            
### MR Data preparation

# MR functions
def add_HORIZONTAL_phase_shift(acq_data, shift):
   ktraj = pMR.get_data_trajectory(acq_data)
   phase_factor = np.exp(1j*2*np.pi*ktraj[:,2]*shift)

   acq_data_new = acq_data.copy()
   acq_data_new.fill(acq_data.as_array()*phase_factor[:,None,None])
   return acq_data_new

def split_data_with_horizontal_shifts(acq_data, shifts):
    '''Generate Nms AcquisitionModel objects
    
    Parameters
    ----------
    acq_data : AcquisitionData
        Acquisition data object
    Nms : int
        Number of motion states
    csm: Coil sensitivity maps
        Coil sensitivity maps object
    shifts: list
        List of horizontal shifts for each motion state

    Returns
    -------
    list: list of AcquisitionData splitted in Nms and 
          list of AcquisitionModel objects relative to the splitted AcquisitionData
    '''

    # Go through each motion states, create corresponding k-space and acquisition model

    Nms = len(shifts)

    acq_ms = [0] * Nms
    
    first_dim = acq_data.dimensions()[0]
    indeces_to_split = np.linspace(0, first_dim, Nms+1).astype(int)

    for ind in range(Nms):
        
        # indeces should be selected randomly
        indeces_to_select = np.arange(indeces_to_split[ind], indeces_to_split[ind+1], 1)
        part_of_data = acq_data.get_subset(indeces_to_select)
        
        data_with_shift = add_HORIZONTAL_phase_shift(part_of_data, shifts[ind])
        
        acq_ms[ind] = data_with_shift      
        acq_ms[ind].sort_by_time()
    return acq_ms

from functools import cached_property
class AcquisitionModel(pMR.AcquisitionModel):
    @cached_property
    def norm(self):
        return pMR.AcquisitionModel.norm(self)
    


class MRAcquisitionModelWithNormCache(pMR.AcquisitionModel):

    def __init__(self, acqs, imgs):
        
        self._norm = None
        super().__init__(acqs, imgs)

    def norm(self):
        if self._norm is not None:
            return self._norm
        self._norm = self.calculate_norm()
        return self._norm

    def calculate_norm(self):
        return pMR.AcquisitionModel.norm(self)


def create_AcquisitionModel_for_each_motion_state(acq_ms, csm):            
    Nms = len(acq_ms)
    E_ms = [0] * Nms

    for ind in range(Nms):
        # Create acquisition model
        E_tmp = pMR.AcquisitionModel(acqs=acq_ms[ind], imgs=csm)
        E_tmp.set_coil_sensitivity_maps(csm)
        im_ms = E_tmp.inverse(acq_ms[ind])

        # E_ms[ind] = pMR.AcquisitionModel(acqs=acq_ms[ind], imgs=im_ms)
        E_ms[ind] = MRAcquisitionModelWithNormCache(acqs=acq_ms[ind], imgs=im_ms)
        
        E_ms[ind].set_coil_sensitivity_maps(csm)

    return E_ms


def save_itrobj(algo, data_path, recons, Nms):
    '''Save iterations and objective function values for algorithm'''
    itrobj = [algo.iterations, algo.objective]
    itrobj[0][0] = 0
    print(itrobj)
    fname = os.path.join(data_path, recons, f'MS_{Nms}', '{}_itrobj.npy'.format(algo.__class__.__name__))
    print (fname)
    np.save(fname, np.asarray(itrobj))


import cil
from packaging.version import Version

if Version(f"{cil.version.major}.{cil.version.minor}.{cil.version.patch}") >= Version("24.0.0"):
    from cil.optimisation.utilities.callbacks import Callback

    class SaveCallback(Callback):
        def __init__(self, save_interval, save_dir, prefix):
            self._save_interval = save_interval
            self.save_dir = save_dir
            self.prefix = prefix
            try:
                os.makedirs(self.save_dir)
            except FileExistsError:
                pass
        @property
        def save_interval(self):
            return self._save_interval
            
        def set_save_interval(self, value):
            if isinstance(value, Integral):
                if value < 0:
                    raise ValueError("Expected positive integer or 0. Got {value}")
                self._save_interval = value

        def __call__(self, algorithm):
            if algorithm.iteration > 0 and (algorithm.iteration % self.save_interval == 0):            
                algorithm.solution.write(os.path.join(self.save_dir, f'{self.prefix}_it_{algorithm.iteration:003}.h5'))


    class SaveObjectiveCallback(SaveCallback):

        def __call__(self, algorithm):
            if algorithm.iteration > 0 and (algorithm.iteration % algorithm.update_objective_interval == 0):            
                itrobj = [algorithm.iterations, algorithm.objective]
                # print(itrobj)
                fname = os.path.join(self.save_dir, '{}_itrobj.npy'.format(algorithm.__class__.__name__))
                # print (fname)
                np.save(fname, np.asarray(itrobj))
