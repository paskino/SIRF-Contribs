# -*- coding: utf-8 -*-

"""MCIR for PET with primal-dual algorithms.

Usage:
  PET_MCIR_PD [--help | options]

Options:
  -T <pattern>, --trans=<pattern>   transformation pattern, * or % wildcard
                                    (e.g., tm_ms*.txt). Enclose in quotations.
  -t <str>, --trans_type=<str>      transformation type (tm, disp, def)
                                    [default: tm]
  -S <pattern>, --sino=<pattern>    sinogram pattern, * or % wildcard
                                    (e.g., sino_ms*.hs). Enclose in quotations.
  -a <pattern>, --attn=<pattern>    attenuation pattern, * or % wildcard
                                    (e.g., attn_ms*.hv). Enclose in quotations.
  -R <pattern>, --rand=<pattern>    randoms pattern, * or % wildcard
                                    (e.g., rand_ms*.hs). Enclose in quotations.
  -n <norm>, --norm=<norm>          ECAT8 bin normalization file
  -e <int>, --epoch=<int>           num epochs [default: 10]
  -r <string>, --reg=<string>       regularisation ("None","FGP_TV","explicit_TV", ...)
                                    [default: None]
  -o <outp>, --outp=<outp>          output file prefix [default: recon]
  --outpath=<string>                output folder path [default: './']
  --param_path=<string>             param folder path [default: './']
  --nxny=<nxny>                     image x and y dimension [default: 127]
  --dxdy=<dxdy>                     image x and y spacing
                                    (default: determined by scanner)
  -I <str>, --initial=<str>         Initial estimate
  --visualisations                  show visualisations
  --nifti                           save output as nifti
  --gpu                             use GPU projector
  --niftypet                        use NiftyPET GPU projector, needs to be used together with --gpu
  --parallelproj                    use Parallelproj GPU projector, needs to be used together with --gpu
  -v <int>, --verbosity=<int>       STIR verbosity [default: 0]
  -s <int>, --save_interval=<int>   save every x iterations [default: 10]
  --descriptive_fname               option to have descriptive filenames
  --update_obj_fn_interval=<int>    frequency to update objective function
                                    [default: 1]
  --alpha=<val>                     regularisation strength (if used)
                                    [default: 0.5]      
  --reg_iters=<val>                 Number of iterations for the regularisation
                                    subproblem [default: 100]
  --precond                         Use preconditioning
  --numSegsToCombine=<val>          Rebin all sinograms, with a given number of
                                    segments to combine. Increases speed.
  --numViewsToCombine=<val>         Rebin all sinograms, with a given number of
                                    views to combine. Increases speed.
  --normaliseDataAndBlock           Normalise raw data and block operator by
                                    multiplying by 1./normK.
  --algorithm=<string>              Which algorithm to run [default: spdhg]
  --numThreads=<int>                Number of threads to use
  --numSubsets=<int>                Number of physical subsets to use [default: 1]
  --gamma=<val>                     parameter controlling primal-dual trade-off (>1 promotes dual)
                                    [default: 1.]
  --PowerMethod_iters=<val>         number of iterations for the computation of operator norms
                                    with the power method [default: 10]
  --templateAcqData                 Use template acd data
  --StorageSchemeMemory             Use memory storage scheme
"""

# SyneRBI Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# This is software developed for the Collaborative Computational
# Project in Synergistic Reconstruction for Biomedical Imaging
# (formerly CCP PETMR)
# (http://www.ccpsynerbi.ac.uk/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from functools import partial
from os import path
import os
from glob import glob
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
import sys

from sirf.Utilities import error, show_2D_array, examples_data_path
import sirf.Reg as reg
import sirf.STIR as pet
from cil.framework import BlockDataContainer, ImageGeometry, BlockGeometry
from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import \
    KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm, ScaledFunction
from cil.optimisation.operators import \
    CompositionOperator, BlockOperator, LinearOperator, GradientOperator, ScaledOperator
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from ccpi.filters import regularisers
from cil.utilities.multiprocessing import NUM_THREADS
import cProfile

__version__ = '0.1.0'

class MCIR(object):
    def __init__(self, docoptargs=None):
        self.num_ms             = -1
        self.trans_files        = []
        self.sino_files         = []
        self.attn_files         = []
        self.rand_files         = []
        self._resampled_attns   = None
        self._rands             = None
        self._resamplers        = None
        self._template_acq_data = None
        
    @property
    def resampled_attns(self):
        return self._resampled_attns

    @resampled_attns.setter
    def resampled_attns(self, value):
        self._resampled_attns = value
    @property
    def rands(self):
        return self._rands
    @rands.setter
    def rands(self, value):
        self._rands = value

    @property
    def resamplers(self):
        ###########################################################################
        # Set up resamplers
        ###########################################################################
        if self._resamplers is not None:
            return self._resamplers
        else:
            if self.trans is None:
                self._resamplers = None
                return self._resamplers
            else:
                self._resamplers = [self.get_resampler(self.image, trans=tran) for tran in self.trans]    
                return self._resamplers
    

    @property
    def template_acq_data(self):
        if self.args['--templateAcqData'] and self._template_acq_data is not None:
            self._template_acq_data = pet.AcquisitionData('Siemens_mMR', span=11, max_ring_diff=15, view_mash_factor=1)
        return self._template_acq_data

    def set_docopt_args(self, args):
        # copy the args
        self.args = dict(args)
        return self
    
    def get_filenames(self):
        """Get filenames."""
        trans = self.args['--trans']
        sino = self.args['--sino']
        attn = self.args['--attn']
        rand = self.args['--rand']

        trans_pattern = str(trans).replace('%', '*')
        sino_pattern = str(sino).replace('%', '*')
        attn_pattern = str(attn).replace('%', '*')
        rand_pattern = str(rand).replace('%', '*')    
        
        if sino_pattern is None:
            raise AssertionError("--sino missing")
        trans_files = sorted(glob(trans_pattern))
        sino_files  = sorted(glob(sino_pattern))
        attn_files  = sorted(glob(attn_pattern))
        rand_files  = sorted(glob(rand_pattern))
        
        num_ms = len(sino_files)
        # Check some sinograms found
        if num_ms == 0:
            raise AssertionError("No sinograms found at {}!".format(sino_pattern))
        # Should have as many trans as sinos
        if len(trans_files) > 0 and num_ms != len(trans_files):
            raise AssertionError("#trans should match #sinos. "
                                "#sinos = " + str(num_ms) +
                                ", #trans = " + str(len(trans_files)))
        # If any rand, check num == num_ms
        if len(rand_files) > 0 and len(rand_files) != num_ms:
            raise AssertionError("#rand should match #sinos. "
                                "#sinos = " + str(num_ms) +
                                ", #rand = " + str(len(rand_files)))

        # For attn, there should be 0, 1 or num_ms images
        if len(attn_files) > 1 and len(attn_files) != num_ms:
            raise AssertionError("#attn should be 0, 1 or #sinos")

        # return [num_ms, trans_files, sino_files, attn_files, rand_files]
        self.num_ms = num_ms
        self.trans_files = trans_files
        self.sino_files = sino_files
        self.attn_files = attn_files
        self.rand_files = rand_files
        return self

    def read_files(self):
        """Read files."""
        if self.trans_files == []:
            trans = None
        else:
            trans_type = self.args['--trans_type']
            if trans_type == "tm":
                trans = [reg.AffineTransformation(f) for f in self.trans_files]
            elif trans_type == "disp":
                trans = [reg.NiftiImageData3DDisplacement(f)
                        for f in self.trans_files]
            elif trans_type == "def":
                trans = [reg.NiftiImageData3DDeformation(f)
                        for f in self.trans_files]
            else:
                raise ValueError("Unknown transformation type")

        sinos_raw = [pet.AcquisitionData(f) for f in self.sino_files]
        attns = [pet.ImageData(f) for f in self.attn_files]
        
        # fix a problem with the header which doesn't allow
        # to do algebra with randoms and sinogram
        rands_arr = [pet.AcquisitionData(f).as_array() for f in self.rand_files]
        rands_raw = [ s * 0 for s in sinos_raw ]
        for r,a in zip(rands_raw, rands_arr):
            r.fill(a)
        
        # return [trans, sinos_raw, attns, rands_raw]
        self.trans = trans
        self.sinos_raw = sinos_raw
        self.attns = attns
        self.rands_raw = rands_raw
        return self


    
    def main(self):
        
        """Run main function."""

        use_gpu = self.args['--gpu'] 

        ###########################################################################
        # Parse input files
        ###########################################################################

        # [num_ms, trans_files, sino_files, attn_files, rand_files] = \
        #     get_filenames(args['--trans'],args['--sino'],args['--attn'],args['--rand'])
        self.get_filenames()

        ###########################################################################
        # Read input
        ###########################################################################

        # [trans, sinos_raw, attns, rands_raw] = \
        #     read_files(trans_files, sino_files, attn_files, rand_files, args['--trans_type'])

        self.read_files()

        self.pre_process_sinos()
        self.pre_process_rands()

        ###########################################################################
        # Initialise recon image
        ###########################################################################

        self.get_initial_estimate()

        ###########################################################################
        # Set up resamplers
        ###########################################################################
        # it's done implicitly
        # if trans is None:
        #     resamplers = None
        # else:
        #     resamplers = [get_resampler(image, trans=tran) for tran in trans]
        

        ###########################################################################
        # Resample attenuation images (if necessary)
        ###########################################################################

        # resampled_attns = resample_attn_images(num_ms, attns, trans, use_gpu, image)
        self.resample_attn_images()
        print ("resampled_attns", len (self.resampled_attns))

        ###########################################################################
        # Set up acquisition models (one per motion state)
        ###########################################################################

        # acq_models, masks = set_up_acq_models(
        #     num_ms, sinos, rands, resampled_attns, image, use_gpu)
        self.set_up_acq_models()

        ###########################################################################
        # Set up reconstructor
        ###########################################################################

        if args['--reg']=='explicit_TV':
            # [F, G, K, normK, tau, sigma, use_axpby, prob, gamma] = set_up_explicit_reconstructor(
            #     use_gpu, num_ms, image, acq_models, resamplers, masks, sinos, rands) 
            self.set_up_explicit_reconstructor()
        else:
            # [F, G, K, normK, tau, sigma, use_axpby, prob, gamma] = set_up_reconstructor(
            #     use_gpu, num_ms, acq_models, resamplers, masks, sinos, rands)
            self.set_up_reconstructor()

        ###########################################################################
        # Get output filename
        ###########################################################################

        self.get_output_filename()

        ###########################################################################
        # Get algorithm
        ###########################################################################

        # algo, num_iter = get_algo(F, G, K, normK, tau, sigma, gamma, use_axpby, prob, outp_file,image)
        self.get_algo()
        print ("algo", self.algo)
        print ("num_iter", self.num_iter)
        ###########################################################################
        # Create save call back function
        ###########################################################################

        self.get_save_callback_function()
        print ("save callback", self.save_callback)

        # ###########################################################################
        # # Run the reconstruction
        # ###########################################################################
        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()
        
        # cProfile.run('self.algo.run(num_iter, verbose=2, print_interval=1, callback=self.save_callback)',
        #             filename='class.cprof')
        # # algo.run(num_iter, verbose=2, print_interval=1, callback=save_callback)
        # # # algo.run(num_iter, verbose=2, callback=save_callback)
        
        # # ... do something ...
        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.dump_stats('pdhg_scaled_operator_2parallelproj.cprof')
    
    def pre_process_sinograms(self, sinos_raw, num_ms):
        """Preprocess raw sinograms.

        Make positive if necessary and do any required rebinning."""
        # If empty (e.g., no randoms), return
        if not sinos_raw:
            return sinos_raw
        # Loop over all sinograms
        sinos = [0]*num_ms
        for ind in range(num_ms):
            # If any sinograms contain negative values
            # (shouldn't be the case), set them to 0
            sino_arr = sinos_raw[ind].as_array()
            if (sino_arr < 0).any():
                print("Input sinogram " + str(ind) +
                    " contains -ve elements. Setting to 0...")
                sinos[ind] = sinos_raw[ind].clone()
                sino_arr[sino_arr < 0] = 0
                sinos[ind].fill(sino_arr)
            else:
                sinos[ind] = sinos_raw[ind]
            # If rebinning is desired
            segs_to_combine = 1
            if self.args['--numSegsToCombine'] is not None:
                segs_to_combine = int(self.args['--numSegsToCombine'])
            views_to_combine = 1
            if self.args['--numViewsToCombine'] is not None:
                views_to_combine = int(self.args['--numViewsToCombine'])
            if segs_to_combine * views_to_combine > 1:
                sinos[ind] = sinos[ind].rebin(segs_to_combine, views_to_combine, do_normalisation=False)
                # only print first time
                if ind == 0:
                    print("Rebinned sino dimensions: {sinos[ind].dimensions()}")
        return sinos
    def pre_process_sinos(self):
        self.sinos = self.pre_process_sinograms(self.sinos_raw, self.num_ms)
    def pre_process_rands(self):
        self.rands = self.pre_process_sinograms(self.rands_raw, self.num_ms)
    def clear_raw_data(self):
        # self.trans_files = []
        # self.sino_files = []
        # self.attn_files = []
        # self.rand_files = []
        self.sinos_raw = None
        self.rands_raw = None
        return self

    def get_initial_estimate(self):
        """Get initial estimate."""

        # from the arguments
        initial_estimate = self.args['--initial']
        nxny = int(self.args['--nxny'])
        

        if initial_estimate is not None:
            image = pet.ImageData(initial_estimate)
        elif self.args['--templateAcqData']:
            # should this use self.template_acq_data instead?
            image = self.sinos[0].create_uniform_image(0., (127, 220, 220))
            image.initialise(dim=(127, 220, 220), vsize=(2.03125, 1.7080754, 1.7080754))
        else:
            # Create image based on ProjData
            image = self.sinos[0].create_uniform_image(0.0, (nxny, nxny))
            # If using GPU, need to make sure that image is right size.
            if self.args['--gpu']:
                dim = (127, 320, 320)
                spacing = (2.03125, 2.08626, 2.08626)
            # elif non-default spacing desired
            elif self.args['--dxdy'] is not None:
                dim = image.dimensions()
                dxdy = float(self.args['--dxdy'])
                spacing = (image.voxel_sizes()[0], dxdy, dxdy)
            if self.args['--gpu'] or self.args['--dxdy'] is not None:
                image.initialise(dim=dim,
                                vsize=spacing)
                image.fill(0.0)

        # return image
        self.image = image
        return self


    def resample_attn_images(self):
        """Resample attenuation images if necessary."""
        resampled_attns = None
        if self.trans is None:
            resampled_attns = self.attns
        else:
            if len(self.attns) > 0:
                resampled_attns = [0]*self.num_ms
                # if using GPU, dimensions of attn and recon images have to match
                ref = self.image if self.args['--gpu'] else None
                for i in range(self.num_ms):
                    # if we only have 1 attn image, then we need to resample into
                    # space of each gate. However, if we have num_ms attn images,
                    # then assume they are already in the correct position, so use
                    # None as transformation.
                    tran = self.trans[i] if len(self.attns) == 1 else None
                    # If only 1 attn image, then resample that. If we have num_ms
                    # attn images, then use each attn image of each frame.
                    attn = self.attns[0] if len(self.attns) == 1 else self.attns[i]
                    resam = self.get_resampler(attn, ref=ref, trans=tran)
                    resampled_attns[i] = resam.forward(attn)
        # return resampled_attns
        self.resampled_attns = resampled_attns
        return self

    def get_resampler(self, img, ref=None, trans=None):
        """Return a NiftyResample object for the specified transform and image."""
        if ref is None:
            ref = self.image
        resampler = reg.NiftyResample()
        resampler.set_reference_image(ref)
        resampler.set_floating_image(img)
        resampler.set_padding_value(0)
        resampler.set_interpolation_type_to_linear()
        if trans is not None:
            resampler.add_transformation(trans)
        return resampler
    def resample_attn_images(self):
        """Resample attenuation images if necessary."""
        self.resampled_attns = None
        if self.trans is None:
            resampled_attns = self.attns
        else:
            if len(self.attns) > 0:
                resampled_attns = [0]*self.num_ms
                # if using GPU, dimensions of attn and recon images have to match
                ref = self.image if self.args['--gpu'] else None
                for i in range(self.num_ms):
                    # if we only have 1 attn image, then we need to resample into
                    # space of each gate. However, if we have num_ms attn images,
                    # then assume they are already in the correct position, so use
                    # None as transformation.
                    tran = self.trans[i] if len(self.attns) == 1 else None
                    # If only 1 attn image, then resample that. If we have num_ms
                    # attn images, then use each attn image of each frame.
                    attn = self.attns[0] if len(self.attns) == 1 else self.attns[i]
                    resam = self.get_resampler(attn, ref=ref, trans=tran)
                    resampled_attns[i] = resam.forward(attn)
        #return resampled_attns
        self.resampled_attns = resampled_attns
        return self

    def set_up_acq_models(self):
        """Set up acquisition models."""
        print("Setting up acquisition models...")

        # From the arguments
        algo = str(self.args['--algorithm'])
        nsub = int(self.args['--numSubsets']) if self.args['--numSubsets'] and algo=='spdhg' else 1
        norm_file = self.args['--norm']
        verbosity = int(self.args['--verbosity'])
    
        print ("args gpu", self.args['--gpu'])
        if not self.args['--gpu']:
            # acq_models = [pet.AcquisitionModelUsingRayTracingMatrix() for k in range(nsub * num_ms)]
            acq_models = [pet.AcquisitionModelUsingParallelproj() for k in range(nsub * self.num_ms)]
        else:
            acq_models = [pet.AcquisitionModelUsingNiftyPET() for k in range(nsub * self.num_ms)]
            for acq_model in acq_models:
                acq_model.set_use_truncation(True)
                acq_model.set_cuda_verbosity(verbosity)
                acq_model.set_num_tangential_LORs(10)

        # create masks
        im_one = self.image.clone().allocate(1.)
        masks = []



        # If present, create ASM from ECAT8 normalisation data
        asm_norm = None
        if norm_file:
            if not path.isfile(norm_file):
                raise error("Norm file not found: " + norm_file)
            asm_norm = pet.AcquisitionSensitivityModel(norm_file)
        

        # Loop over each motion state
        for ind in range(self.num_ms):
            # Create attn ASM if necessary
            asm_attn = None
            if self.resampled_attns is not None: 
                s = self.sinos[ind]
                ra = self.resampled_attns[ind]
                # am = pet.AcquisitionModelUsingRayTracingMatrix()
                am = pet.AcquisitionModelUsingParallelproj()
                asm_attn = self.get_asm_attn(s,ra,am)

            # Get ASM dependent on attn and/or norm
            asm = None
            if asm_norm and asm_attn:
                if ind == 0:
                    print("ASM contains norm and attenuation...")
                asm = pet.AcquisitionSensitivityModel(asm_norm, asm_attn)
            elif asm_norm:
                if ind == 0:
                    print("ASM contains norm...")
                asm = asm_norm
            elif asm_attn:
                if ind == 0:
                    print("ASM contains attenuation...")
                asm = asm_attn
                    
            # Loop over physical subsets
            for k in range(nsub):
                current = k * self.num_ms + ind

                if asm:
                    acq_models[current].set_acquisition_sensitivity(asm)
                #KT we'll set the background in the KL function below
                #KTif len(rands) > 0:
                #KT    acq_models[ind].set_background_term(rands[ind])

                # Set up
                acq_models[current].set_up(self.sinos[ind], self.image)    
                acq_models[current].num_subsets = nsub
                acq_models[current].subset_num = k 

                # compute masks 
                if ind == 0:
                    mask = acq_models[current].direct(im_one)
                    masks.append(mask)

                # rescale by number of gates
                if self.num_ms > 1:
                    acq_models[current] = ScaledOperator(acq_models[current], 1./self.num_ms)

        #return acq_models, masks
        self.acq_models = acq_models
        self.masks = masks
        return self

    def get_asm_attn(self,sino, attn, acq_model):
        """Get attn ASM from sino, attn image and acq model."""
        asm_attn = pet.AcquisitionSensitivityModel(attn, acq_model)
        # temporary fix pending attenuation offset fix in STIR:
        # converting attenuation into 'bin efficiency'
        asm_attn.set_up(sino)
        bin_eff = pet.AcquisitionData(sino)
        bin_eff.fill(1.0)
        asm_attn.unnormalise(bin_eff)
        asm_attn = pet.AcquisitionSensitivityModel(bin_eff)
        return asm_attn

    def set_up_reconstructor(self):
        """Set up reconstructor."""

        # From the arguments
        algo = str(self.args['--algorithm'])
        regularizer = str(self.args['--reg'])
        r_iters = int(self.args['--reg_iters'])
        r_alpha = float(self.args['--alpha'])
        nsub = int(self.args['--numSubsets']) if self.args['--numSubsets'] and algo=='spdhg' else 1
        precond = self.args['--precond']
        param_path = str(self.args['--param_path'])
        normalise = self.args['--normaliseDataAndBlock']
        gamma = float(self.args['--gamma'])
        output_name = str(self.args['--outp'])
        

        if not os.path.exists(param_path):
            os.makedirs(param_path)

        if normalise:
            raise ValueError('options {} and regularization={} are not yet implemented together'.format(normalise, regularizer))

        # We'll need an additive term (eta). If randoms are present, use them
        # Else, use a scaled down version of the sinogram
        etas = self.rands if self.rands is not None else [sino * 0 + 1e-5 for sino in self.sinos]

        # Create composition operators containing linear
        # acquisition models and resamplers,
        # and create data fit functions
        
        if nsub == 1:
            if self.resamplers is None:
                #KT C = [am.get_linear_acquisition_model() for am in acq_models]
                C = [am for am in self.acq_models]
            else:
                C = [CompositionOperator(
                        #KTam.get_linear_acquisition_model(),
                        am,
                        res, preallocate=True)
                        for am, res in zip(*(self.acq_models, self.resamplers))]
            fi = [KullbackLeibler(b=sino, eta=eta, mask=self.masks[0].as_array(),use_numba=True) 
                    for sino, eta in zip(self.sinos, etas)]
        else:
            C = [am for am in self.acq_models]
            fi = [None] * (self.num_ms * nsub)
            for (k,i) in np.ndindex((nsub,self.num_ms)):
                # resample if needed
                if self.resamplers is not None:
                    C[k * num_ms + i] = CompositionOperator(
                        #KTam.get_linear_acquisition_model(),
                        C[k * num_ms + i],
                        self.resamplers[i], preallocate=True)
                fi[k * num_ms + i] = KullbackLeibler(b=self.sinos[i], eta=etas[i], 
                                              mask=self.masks[k].as_array(),use_numba=True)


        if regularizer == "FGP_TV":
            r_tolerance = 1e-7
            r_iso = 0
            r_nonneg = 1
            r_printing = 0
            device = 'gpu' if self.args['--gpu'] else 'cpu'
            device = 'cpu'
            G = FGP_TV(r_alpha, r_iters, r_tolerance,
                    r_iso, r_nonneg, r_printing, device)
            if precond:
                FGP_TV.proximal = precond_proximal
        elif regularizer == "None":
            G = IndicatorBox(lower=0)
        else:
            raise ValueError("Unknown regularisation. Expected FGP_TV or None, got {}".format(regularizer))
        
        F = BlockFunction(*fi)
        K = BlockOperator(*C)

        if algo == 'spdhg':
            prob = [1./ len(K)] * len(K)
        else:
            prob = None

        if not precond:
            if algo == 'pdhg':
                # we want the norm of the whole physical BlockOp
                normK = self.get_proj_norm(BlockOperator(*C),param_path)
                sigma = gamma/normK
                tau = 1/(normK*gamma)
            elif algo == 'spdhg':
                # we want the norm of each component
                normK = self.get_proj_normi(BlockOperator(*C),nsub,param_path)
                # we'll let spdhg do its default implementation
                sigma = None
                tau = None
            use_axpby = True
        else:
            normK=None
            if algo == 'pdhg':
                tau = K.adjoint(K.range_geometry().allocate(1.))
                # CD take care of edge of the FOV
                filter = pet.TruncateToCylinderProcessor()
                filter.apply(tau)
                backproj_np = tau.as_array()
                vmax = np.max(backproj_np[backproj_np>0])
                backproj_np[backproj_np==0] = 10 * vmax
                tau_np = 1/backproj_np
                tau.fill(tau_np)
                # apply filter second time just to be sure
                filter.apply(tau)
                tau_np = tau.as_array()
                tau_np[tau_np==0] = 1 / (10 * vmax)
            elif algo == 'spdhg':
                taus_np = []
                for (Ki,pi) in zip(K,prob):
                    tau = Ki.adjoint(Ki.range_geometry().allocate(1.))
                    # CD take care of edge of the FOV
                    filter = pet.TruncateToCylinderProcessor()
                    filter.apply(tau)
                    backproj_np = tau.as_array()
                    vmax = np.max(backproj_np[backproj_np>0])
                    backproj_np[backproj_np==0] = 10 * vmax
                    tau_np = 1/backproj_np
                    tau.fill(tau_np)
                    # apply filter second time just to be sure
                    filter.apply(tau)
                    tau_np = tau.as_array()
                    tau_np[tau_np==0] = 1 / (10 * vmax)
                    taus_np.append(pi * tau_np)
                taus = np.array(taus_np)
                tau_np = np.min(taus, axis = 0)
            tau.fill(tau_np)
            # save
            np.save('{}/tau_{}.npy'.format(param_path, output_name), tau_np, allow_pickle=True)

            i = 0
            sigma = []
            xx = K.domain_geometry().allocate(1.)
            for Ki in K:
                tmp_np = Ki.direct(xx).as_array()
                tmp_np[tmp_np==0] = 10 * np.max(tmp_np)
                sigmai = Ki.range_geometry().allocate(0.)
                sigmai.fill(1/tmp_np)
                sigma.append(sigmai)
                # save
                # np.save('{}/sigma_{}.npy'.format(param_path,i), 1/tmp_np, allow_pickle=True)
                i += 1
            sigma = BlockDataContainer(*sigma)
            # trade-off parameter
            sigma *= gamma
            tau *= (1/gamma)
            use_axpby = False


        # return [F, G, K, normK, tau, sigma, use_axpby, prob, gamma]
        self.F = F
        self.G = G
        self.K = K
        self.normK = normK
        self.tau = tau
        self.sigma = sigma
        self.use_axpby = use_axpby
        self.prob = prob
        self.gamma = gamma
        return self

    def set_up_explicit_reconstructor(self):
        """Set up reconstructor."""

        # From the arguments
        algo = str(self.args['--algorithm'])
        r_alpha = float(self.args['--alpha'])
        nsub = int(self.args['--numSubsets']) if self.args['--numSubsets'] and algo=='spdhg' else 1
        precond = self.args['--precond']
        param_path = str(self.args['--param_path'])
        normalise = self.args['--normaliseDataAndBlock']
        gamma = float(self.args['--gamma'])

        if precond:
            raise ValueError('Options precond and explicit TV are not yet implemented together')

        # We'll need an additive term (eta). If randoms are present, use them
        # Else, use a scaled down version of the sinogram
        etas = self.rands if self.rands is not None else [sino * 0 + 1e-5 for sino in self.sinos]

        # Create composition operators containing linear
        # acquisition models and resamplers,
        # and create data fit functions

        if nsub == 1:
            if self.resamplers is None:
                #KT C = [am.get_linear_acquisition_model() for am in acq_models]
                C = [am for am in self.acq_models]
            else:
                C = [CompositionOperator(
                        #KTam.get_linear_acquisition_model(),
                        am,
                        res, preallocate=True)
                        for am, res in zip(*(self.acq_models, self.resamplers))]
            fi = [KullbackLeibler(b=sino, eta=eta, mask=self.masks[0].as_array(),use_numba=True) 
                    for sino, eta in zip(self.sinos, etas)]
        else:
            C = [am for am in self.acq_models]
            fi = [None] * (self.num_ms * nsub)
            for (k,i) in np.ndindex((nsub,self.num_ms)):
                # resample if needed
                if self.resamplers is not None:            
                    C[k * num_ms + i] = CompositionOperator(
                        #KTam.get_linear_acquisition_model(),
                        C[k * num_ms + i],
                        self.resamplers[i], preallocate=True)
                fi[k * num_ms + i] = KullbackLeibler(b=self.sinos[i], eta=etas[i], 
                                             mask=self.masks[k].as_array(),use_numba=True)

        # define gradient
        Grad = GradientOperator(self.image, backend='c', correlation='SpaceChannel')
        normGrad = get_grad_norm(Grad,param_path)

        # define data fit
        data_fit = MixedL21Norm()
        MixedL21Norm.proximal = MixedL21Norm_proximal

        if algo == 'pdhg':
            # we want the norm of the whole physical BlockOp
            normProj = self.get_proj_norm(BlockOperator(*C),param_path)
            if normalise:
                C_rs = [ScaledOperator(Ci,1/normProj) for Ci in C]
                Grad_rs = ScaledOperator(Grad,1/normGrad)
                C_rs.append(Grad_rs)
                f_rs = [ScaledFunction(f,normProj) 
                        for f in fi]
                f_rs.append(ScaledFunction(data_fit,r_alpha * normGrad))
                normK = np.sqrt(2)
            else:
                C.append(Grad)
                fi.append(ScaledFunction(data_fit,r_alpha))
                normK = np.sqrt(normProj**2 + normGrad**2)
            sigma = gamma/normK
            tau = 1/(normK*gamma)
            prob = None
                
        elif algo == 'spdhg':
            # we want the norm of each component
            normProj = get_proj_normi(BlockOperator(*C),nsub,param_path)
            if normalise:
                C_rs = [ScaledOperator(Ci,1/normProji) for Ci, normProji in zip(C,normProj)]
                Grad_rs = ScaledOperator(Grad,1/normGrad)
                C_rs.append(Grad_rs)
                f_rs = [ScaledFunction(f,normProji) 
                        for f, normProji in zip(fi, normProj)]
                f_rs.append(ScaledFunction(data_fit,r_alpha * normGrad))
                normK = [1.] * len(C_rs)
                prob = [1./(2 * (len(C_rs)-1))] * (len(C_rs)-1) + [1./2]
            else:
                C.append(Grad)
                fi.append(ScaledFunction(data_fit,r_alpha))
                normK = normProj + [normGrad]
                prob = [1./(2 * (len(C)-1))] * (len(C)-1) + [1./2]
            # we'll let spdhg do its default stepsize implementation
            sigma = None
            tau = None        
        else:
            raise error("algorithm '{}' is not implemented".format(algo))

        G = IndicatorBox(lower=0)

        if normalise:
            F = BlockFunction(*f_rs)
            K = BlockOperator(*C_rs)
        else:
            F = BlockFunction(*fi)
            K = BlockOperator(*C)
        use_axpby = False

        # return [F, G, K, normK, tau, sigma, use_axpby, prob, gamma]
        self.F = F
        self.G = G
        self.K = K
        self.normK = normK
        self.tau = tau
        self.sigma = sigma
        self.use_axpby = use_axpby
        self.prob = prob
        self.gamma = gamma
        return self
    
    def get_proj_norm(self, K,param_path):
        # load or compute and save norm of whole operator
        param_path = str(self.args['--param_path'])
        file_path = '{}/normK.npy'.format(param_path)
        if os.path.isfile(file_path):
            print('Norm file {} exists, load it'.format(file_path))
            normK = float(np.load(file_path, allow_pickle=True))
        else:
            print('Norm file {} does not exist, compute it'.format(file_path))
            # normK = PowerMethod(K)[0]
            normK = K.norm()
            # save to file
            np.save(file_path, normK, allow_pickle=True)
        return normK

    def get_proj_normi(self, K,nsub,param_path):
        # load or compute and save norm of each sub-operator
        # (over motion states and subsets)
        param_path = str(self.args['--param_path'])
        file_path = '{}/normK_nsub{}.npy'.format(param_path, nsub)
        if os.path.isfile(file_path):
            print('Norm file {} exists, load it'.format(file_path))
            normK = np.load(file_path, allow_pickle=True).tolist()
        else: 
            print('Norm file {} does not exist, compute it'.format(file_path))
            # normK = [PowerMethod(Ki)[0] for Ki in K]
            normK = [Ki.norm() for Ki in K]
            # save to file
            np.save(file_path, normK, allow_pickle=True)
        return normK

    def get_output_filename(self):
        """Get output filename."""

        # From the arguments
        outp_file = self.args['--outp']
        descriptive_fname = self.args['--descriptive_fname']
        norm_file = self.args['--norm']
        includesRand = True if self.args['--rand'] is not None else False
        algorithm = str(self.args['--algorithm'])
        precond = self.args['--precond']
        regularisation = self.args['--reg']
        r_iters = int(self.args['--reg_iters'])
        r_alpha = float(self.args['--alpha'])
        gamma = float(self.args['--gamma'])
        nsub = int(self.args['--numSubsets']) if self.args['--numSubsets'] is not None and algorithm=='spdhg' else 1
        normalise = self.args['--normaliseDataAndBlock']

        if descriptive_fname:
            outp_file += "_Reg-" + regularisation
            if regularisation is not None:
                outp_file += "-alpha" + str(r_alpha)
            outp_file += "_nGates" + str(len(self.sino_files))
            outp_file += "_nSubsets" + str(nsub)
            outp_file += '_' + algorithm
            if not precond:
                outp_file += "_noPrecond"
            else:
                outp_file += "_wPrecond"
            outp_file += "_gamma" + str(gamma)
            if normalise:
                outp_file += "normalise"
            if len(self.attn_files) > 0:
                outp_file += "_wAC"
            if norm_file:
                outp_file += "_wNorm"
            if includesRand:
                outp_file += "_wRands"
            if self.args['--gpu']:
                outp_file += "_wGPU"
            if regularisation == 'FGP_TV':          
                outp_file += "-riters" + str(r_iters)
            if self.resamplers is None:
                outp_file += "_noMotion"
        # return outp_file
        self.outp_file = outp_file
        return self

    def get_algo(self):

        # from the arguments:
        algorithm = str(self.args['--algorithm'])
        num_epoch = int(self.args['--epoch'])
        update_obj_fn_interval = int(self.args['--update_obj_fn_interval'])
        regularisation = self.args['--reg']

        """Get the reconstruction algorithm."""
        if algorithm == 'pdhg':
            num_iter = num_epoch
            algo = PDHG(
                    f=self.F,
                    g=self.G,
                    operator=self.K,
                    tau=self.tau,
                    sigma=self.sigma, 
                    initial=self.image,
                    use_axpby=self.use_axpby,
                    max_iteration=num_epoch,           
                    update_objective_interval=update_obj_fn_interval,
                    log_file=self.outp_file+".log",
                    )
        elif algorithm == 'spdhg':
            if regularisation == 'explicit_TV':
                num_iter = 2 * (len(K)-1) * num_epoch
            else:
                num_iter = len(self.K) * num_epoch
            algo = SPDHG(            
                    f=self.F, 
                    g=self.G, 
                    operator=self.K,
                    tau=self.tau,
                    sigma=self.sigma,
                    gamma=self.gamma,
                    initial=self.image,
                    prob=self.prob,
                    use_axpby=self.use_axpby,
                    norms=self.normK,
                    max_iteration=num_iter,         
                    update_objective_interval=update_obj_fn_interval,
                    log_file=self.outp_file+".log",
                    )
        else:
            raise ValueError("Unknown algorithm: " + algorithm)
        # return algo, num_iter
        self.algo = algo
        self.num_iter = num_iter
        return self

    def get_save_callback_function(self):
        """Get the save callback function."""

        # from the arguments
        save_interval = int(self.args['--save_interval']) if self.args['--save_interval'] is not None else None
        nifti = self.args['--nifti']
        outpath = str(self.args['--outpath']) if self.args['--outpath'] is not None else None


        if not os.path.exists(outpath):
            os.makedirs(outpath)
        save_interval = min(save_interval, self.num_iter)

        def save_callback(save_interval, nifti, outpath, outp_file,
                        num_iter, iteration,
                        last_objective, x):
            """Save callback function."""
            #completed_iterations = iteration + 1
            completed_iterations = iteration
            if completed_iterations % save_interval == 0 or \
                    completed_iterations == num_iter:
                # print("File should be saved at {}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
                # print(os.getcwd())
                if not nifti:
                    x.write("{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
                else:
                    reg.NiftiImageData(x).write(
                        "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))

            # if not nifti:
            #     x.write("{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))
            # else:
            #     reg.NiftiImageData(x).write(
            #         "{}/{}_iters_{}".format(outpath,outp_file, completed_iterations))

        psave_callback = partial(
            save_callback, save_interval, nifti, outpath, self.outp_file, self.num_iter)
        # return psave_callback
        self.save_callback = psave_callback
        return self

    def get_from_args(self, keyword):
        return self.args[keyword]




def display_results(out_arr, slice_num=None):
    """Display results if desired."""

    # from the arguments
    visualisations = True if args['--visualisations'] else False
    if visualisations:
        # show reconstructed image
        # out_arr = algo.get_output().as_array()
        if slice_num is None:
            z = out_arr.shape[0]//2
        else:
            z = slice_num
        show_2D_array('Reconstructed image', out_arr[z, :, :])
        plt.show()

    
def get_domain_sirf2cil(domain_sirf):
    
    return ImageGeometry(
            voxel_num_x = domain_sirf.shape[2], 
            voxel_num_y = domain_sirf.shape[1], 
            voxel_num_z = domain_sirf.shape[0],
            voxel_size_x = domain_sirf.voxel_sizes()[2], 
            voxel_size_y = domain_sirf.voxel_sizes()[1],
            voxel_size_z = domain_sirf.voxel_sizes()[0])


if __name__ == "__main__":
    ###########################################################################
    # Global set-up
    ###########################################################################
    args = docopt(__doc__, version=__version__)
    
    # storage scheme
    if args['--StorageSchemeMemory']:
        pet.AcquisitionData.set_storage_scheme('memory')
    else:
        pet.AcquisitionData.set_storage_scheme('default')
    # Verbosity
    pet.set_verbosity(int(args['--verbosity']))
    if int(args['--verbosity']) == 0:
        msg_red = pet.MessageRedirector(None, None, None)
    # Number of threads
    numThreads = args['--numThreads'] if args['--numThreads'] else NUM_THREADS
    pet.set_max_omp_threads(numThreads)

    if args['--templateAcqData']:
        template_acq_data = pet.AcquisitionData('Siemens_mMR', span=11, max_ring_diff=15, view_mash_factor=1)
    
    # loc_data = '/home/edo/scratch/Dataset/PETMR/2020MCIR/cardiac_resp/'
    # args = {}
   
           
    # args['-o'] = 'gated_pdhg'
    # args['--algorithm'] = 'pdhg'
    # args['-r'] = 'FGP_TV'
    # args['--outpath'] = '/home/edo/scratch/Dataset/PETMR/2020MCIR/results/pdhg/new_motion_rescaled_gamma_1_noprecond_alpha_3_gated_pdhg/recons'
    # args['--param_path'] = '/home/edo/scratch/Dataset/PETMR/2020MCIR/results/pdhg/new_motion_rescaled_gamma_1_noprecond_alpha_3_gated_pdhg/params'
    # args['--epoch'] = '1'
    # args['--update_obj_fn_interval'] = 10
    # args['--descriptive_fname'] = True
    # args['-v'] = '0'
    # args['-S'] = loc_data + "pet/EM_g*.hs"
    # args['-R'] = loc_data + "pet/total_background_g*.hs"
    # args['-n'] = loc_data + "pet/NORM.n.hdr"
    # args['-a'] = loc_data + "pet/MU_g*.nii"
    # args['-T'] = loc_data + "pet/transf_g*.nii"
    # args['-t'] = 'def'
    # args['--nifti'] = True
    # args['--alpha'] = '3'
    # args['--gamma'] = '1'
    # args['--dxdy'] ='3.12117'
    # args['--nxny'] = '180'
    # args['-s'] = '50'
    # args['--numThreads'] = '16'
    # args['--gpu'] = False

    # args['--trans_type'] = args['-t']
    # args['--trans'] = args['-T']
    # args['--sino'] = args['-S']
    # args['--rand'] = args['-R']
    # args['--norm'] = args['-n']
    # args['--outp'] = args['-o']
    # args['--attn'] = args['-a']
    # args['--verbosity'] = args['-v']
    # args['--save_interval'] = args['-s']
    
    # args['--numSegsToCombine'] = '11'
    # args['--numViewsToCombine'] = '2'
    # args['--initial'] = None
    # args['--templateAcqData'] = False
    # args['--numSubsets'] = '1'
    # args['--reg'] = 'FGP_TV'
    # args['--reg_iters'] = '100'
    # args['--precond'] = None

    # args['--normaliseDataAndBlock'] = False

    # # main()
    
    dmcir = MCIR()
    # this should be taken out of the MCIR class, which should be given the 
    # information it requires without being fixed to a particular docopt version
    dmcir.set_docopt_args(args)
    # for k,v in dmcir.args.items():
    #     print (k,v)
    # dmcir.get_filenames()
    # dmcir.main()

    ###########################################################################
    # Pre-Process Input Data
    ###########################################################################

    ###########################################################################
    # Parse input files
    ###########################################################################

    dmcir.get_filenames()

    ###########################################################################
    # Read input
    ###########################################################################

    dmcir.read_files()
    dmcir.pre_process_sinos()
    dmcir.pre_process_rands()
    dmcir.clear_raw_data()

    ###########################################################################
    # Initialise recon image
    ###########################################################################

    dmcir.get_initial_estimate()
   

    ###########################################################################
    # Resample attenuation images (if necessary)
    ###########################################################################

    dmcir.resample_attn_images()
    print ("resampled_attns", len (dmcir.resampled_attns))

    ###########################################################################
    # Set up acquisition models (one per motion state)
    ###########################################################################

    dmcir.set_up_acq_models()

    ###########################################################################
    # Set up reconstructor
    ###########################################################################

    if args['--reg']=='explicit_TV':
        dmcir.set_up_explicit_reconstructor()
    else:
        dmcir.set_up_reconstructor()

    ###########################################################################
    # Get output filename
    ###########################################################################

    dmcir.get_output_filename()

    ###########################################################################
    # Get algorithm
    ###########################################################################

    dmcir.get_algo()
    print ("algo", dmcir.algo)
    print ("num_iter", dmcir.num_iter)
    ###########################################################################
    # Create save call back function
    ###########################################################################

    dmcir.get_save_callback_function()
    print ("save callback", dmcir.save_callback)


    ###########################################################################
    # Run the reconstruction
    ###########################################################################
    # import cProfile, pstats, io
    # from pstats import SortKey
    # pr = cProfile.Profile()
    # pr.enable()
    
    # cProfile.run('dmcir.algo.run(dmcir.num_iter, verbose=2, print_interval=1, callback=dmcir.save_callback)')
    dmcir.algo.run(dmcir.num_iter, verbose=2, print_interval=1, callback=dmcir.save_callback)
    
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.dump_stats('pdhg_scaled_operator_2parallelproj.cprof')
    
