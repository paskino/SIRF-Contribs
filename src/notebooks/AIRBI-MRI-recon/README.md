# St Georges Pipeline

## Creation of Docker image

The `compose.sh` script enables users to create a docker container either to be launched by XNAT to process all files in the `/input` directory and save into `/output` or as development container.

| Option | Description |
| :--- | :--- |
| `exec` (default) | Builds the `exec` target image, starts the container, and runs `bash mr_recon.sh` in the foreground. |
| `exec-build` | Builds the `exec` executable image and stops (doesn't start the container with `docker compose up`). This image should be used by XNAT|
| `jupyter` | Builds the `jupyter` target image, starts Jupyter Notebook in the background, and prints the URL to access it. |
| `rebuild` | Force rebuilds the `base` image target without using the Docker cache (`--no-cache`). |
| `down` | Stops and removes the running containers, networks, and volumes by running `docker compose down`. |



## Development on a running docker container

With VSCode install:
- Dev Containers
- Jupyter
- Remote - SSH (optional)

Start the image with `./compose.sh jupyter` and then attach to the running container.