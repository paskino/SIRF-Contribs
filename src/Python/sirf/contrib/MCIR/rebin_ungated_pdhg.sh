#! /bin/bash

# defaults
gamma=1.0
alpha=0.5
epochs=200
transform="pet/transf_g*.nii"

while getopts ha:g:e:i:t: option
 do
 case "${option}"
 in
  a) alpha=${OPTARG};;
  g) gamma=${OPTARG};;
  e) epochs=${OPTARG};;
  i) initial=${OPTARG};;
  t) transform=${OPTARG};;
  h)
   echo "Usage: $0 -a alpha -g gamma -e epochs"
   exit 
   ;;
  *)
   echo "Wrong option passed. Use the -h option to get some help." >&2
   exit 1
  ;;
 esac
done


# base directory
mcir_dir=/home/vol05/scarf595/MCIR
work_dir=/work3/cse/synerbi/
loc_data=${work_dir}/cardiac_resp
# loc_algo=${mcir_dir}/SIRF/examples/Python/PETMR
loc_algo=${work_dir}/mcir_build/SIRF-Contribs/src/Python/sirf/contrib/MCIR

base_result=${work_dir}/results/pdhg
##############    RUN NAME    ################
run_name=rebin_rescaled_gamma_${gamma}_alpha_${alpha}_ungated_pdhg

loc_reco=${base_result}/${run_name}/recons
loc_param=${base_result}/${run_name}/params
                       
# create the run directory and get in there
mkdir -p ${base_result}/${run_name}
cd ${base_result}/${run_name}

script_name=PET_recon_file.py
cp ${loc_algo}/${script_name} ${base_result}/${run_name}
cd ${base_result}/${run_name}

###############   CONFIGURATION   ###################

#####   TEST   #####
#epochs=2
#update_interval=48      
#####   RUN   ##### 
# epochs=5000
update_interval=10
save_interval=50

python ${script_name}                                   \
-o ungated_pdhg                                         \
--algorithm=pdhg                                        \
-r FGP_TV                                               \
--outpath=$loc_reco                                     \
--param_path=$loc_param                                 \
-e ${epochs}                                            \
--update_obj_fn_interval=${update_interval}             \
--descriptive_fname                                     \
-v 0                                                    \
-S "$loc_data/pet/EM_g*.hs"                             \
-R "$loc_data/pet/total_background_g*.hs"               \
-n "$loc_data/pet/NORM.n.hdr"                           \
-a "$loc_data/pet/MU_g*.nii"                            \
--nifti                                                 \
--alpha=${alpha}                                        \
--gamma=${gamma}                                        \
--numSegsToCombine=11                                   \
--numViewsToCombine=2                                   \
--dxdy=3.12117                                          \
--nxny=180                                              \
-s ${save_interval}                                     \
--numThreads=32 2>&1 > script.log
