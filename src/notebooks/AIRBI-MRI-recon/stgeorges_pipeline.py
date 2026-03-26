import subprocess
import os
import logging

logger = logging.getLogger(__name__)

# https://discord.com/channels/1242028164105109574/1481253905852792962/1481263393733087345
command = "siemens_to_ismrmrd"
data_dir = "/home/jovyan/work/data/"
proc_dir = os.path.join(data_dir, "proc")

input_files = [ "meas_MID00613_FID129151_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat",             
                "meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat",
                "meas_MID00617_FID129155_AI_RECON_SEQD_512_GF4_AX_RL.dat",
                "meas_MID00619_FID129157_AI_RECON_SEQD_512_GF4_AX_RL.dat"
]

for fname in input_files:
    file_in = os.path.join(data_dir, fname)
    file_out = os.path.join(proc_dir, os.path.basename(file_in).replace(".dat", ".h5"))
    if os.path.exists(file_out):
        logger.warning(f"Output file {file_out} already exists. Removing it.")
        os.remove(file_out)

    out = subprocess.run(
        [command, "-f", file_in, "-o", file_out, "-Z", "-M"],
        capture_output=True,
        text=True,
    )

    logger.info(out.stdout)
    logger.error(out.stderr)


# siemens_to_ismrmrd -f meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.dat -o meas_MID00614_FID129152_CONVENTIONAL_RECON_SEQD_GF2_AX_RL.h5 -Z -M