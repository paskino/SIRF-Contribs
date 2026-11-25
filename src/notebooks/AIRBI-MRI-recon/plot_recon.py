#%%
import numpy as np
from cil.utilities.display import show2D

def get_percentiles(array, percentiles=(10,98.1)):
    import numpy as np
    mp = np.percentile(array, percentiles[0])
    Mp = np.percentile(array, percentiles[1])
    return (mp, Mp)
#%%
# 1 load data

# Load Original Siemens AI reconstruction
# Data loaded from DICOM files, exported by Slicer3D as NIfTI file

siemens_ai_recon_fname = "/output/recon_MID00619_FID129157_AI_RECON_SEQD_512_GF4_AX_RL/PERSON_DICOM_siemens_ai_recon.nii"
from sirf.Reg import NiftiImageData3D
from sirf.Gadgetron import ImageData
siemens_ai_recon = NiftiImageData3D(siemens_ai_recon_fname)  # Load the Siemens AI reconstruction from the NIfTI file

from skimage.transform import downscale_local_mean
print (siemens_ai_recon.asarray().shape)

#%%
########## SIEMENS FULLY SAMPLED RECONSTRUCTION
siemens_fs_recon_fname = "/output/recon_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL/PERSON_DICOM_siemens_fully_sampled.nii"
from sirf.Reg import NiftiImageData3D
from sirf.Gadgetron import ImageData
siemens_fs_recon = NiftiImageData3D(siemens_fs_recon_fname)  # Load the Siemens fully sampled reconstruction from the NIfTI file
#%%
siemens_fs_recon = np.squeeze(np.abs(np.fliplr(np.flipud(siemens_fs_recon.asarray()[110:410, 130:430,6])))).T
# show2D(siemens_fs_recon)
#%%
# Try to put the Siemens AI reconstruction into a lower resolution grid to match the undersampled data and fully sampled data
binned = downscale_local_mean(siemens_ai_recon.asarray(), (2, 2, 1))  # keep Z, bin 2x2 in XY
# find the image to approximately the same spot as the reconstructed images from the data with SIRF
siemens_rebinned_recon = np.abs(np.fliplr(np.flipud(binned[110:410, 130:430,  6]))).T


# import our reconstructions
import os
dirname = '/output/nii'

kind = ['ai', 'us']
import json
am = {}
for k in kind:
    with open(os.path.join(dirname, f'{k}/{k}-config.json'), 'r') as f:
        settings = json.load(f)
    am[k]['alpha'] = settings['alpha']
    am[k]['csm_nits'] = settings['nits']
    am[k]['csm_ckhs'] = settings['ckhs']
    am[k]['iterations'] = settings['iterations']
    am[k]['loss'] = settings['loss']
    
    am[k]['recon'] = ImageData(os.path.join(dirname, f'{k}/{k}-recon.h5'))  # Load the reconstruction
    

_data = [siemens_fs_recon,
        siemens_rebinned_recon, 
        np.squeeze(np.abs(am['ai']['recon'].asarray()[11,110:410, 80:380])).T,
        np.squeeze(np.abs(am['us']['recon'].asarray()[11,110:410, 80:380])).T
        ]
_title = ['Fully Sampled', 
              'Siemens AI recon', 
              f'AI data +{am["ai"]["alpha"]}Wavelet',
              f'Undersampled LS+{am['us']['alpha']}Wavelet'
              ]
ranges = [get_percentiles(el) for el in _data]

show2D(_data,
       title=_title,
       fix_range=ranges)