## St Georges Pipeline

### On a running docker container

```
docker exec sirf-jupyter bash -lc 'source /opt/SIRF-SuperBuild/INSTALL/bin/env_sirf.sh && cd work/SIRF-Contribs/src/notebooks/AIRBI-MRI-recon && python stgeorges_pipeline.py '
```

Copy the DICOM files
```
scp -r ubuntu@172.16.111.141:workdir/data/recon ~/Data/Emily/
```

### As executable

A command to execute is defined in the `docker-compose.yml` at the with the `command` keyword.
To run the container as executable it should contain `command: ["bash", "mr_recon.sh"]`. If running a container with jupyter the `command` line should be commented out.  

```
docker build . --no-cache
docker compose build
docker compose up
```

Make sure that `/input` and `/output` are bound to the container as this is where the script expects to find files

The `build` context: The `.` tells Docker Compose that the "build context" is the current directory (the same folder where the `docker-compose.yml` lives).
The default filename: By default, when you specify a build context, Docker Compose automatically looks for a file named exactly `Dockerfile` inside that directory.
The image tag: `sirf_for_xnat:latest`, Docker Compose won't try to download this image from the internet. Instead, it will build the image using your local `Dockerfile` and then name (tag) the resulting image `sirf_for_xnat:latest`.