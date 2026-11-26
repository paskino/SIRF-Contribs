FROM synerbi/sirf:edge AS base

USER jovyan
RUN pip install pydicom
RUN pip install ismrmrd
RUN pip install nii2dcm


# RUN git clone --branch downsampled-MRI-recon-stgeorge --depth 1 https://github.com/paskino/SIRF-Contribs.git


RUN mkdir -p /home/jovyan/recon_scripts
COPY --chown=jovyan:users mr_recon.sh /home/jovyan/recon_scripts/
COPY --chown=jovyan:users stgeorges_pipeline.py /home/jovyan/recon_scripts/
COPY --chown=jovyan:users stgeorges_utils.py /home/jovyan/recon_scripts/
COPY --chown=jovyan:users load-undersample.py /home/jovyan/recon_scripts/

FROM base AS exec
WORKDIR /home/jovyan/recon_scripts
# uncomment the following if you want to launch the container as executable to run a reconstruction
CMD ["bash" , "mr_recon.sh"]

FROM base AS jupyter
RUN conda install scikit-image ipykernel -y
# uncomment the following if you want to launch the container as executable to run a reconstruction
# CMD ["bash" , "mr_recon.sh"]
