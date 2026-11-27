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

> [!NOTE]
> Due to https://github.com/SyneRBI/SIRF-SuperBuild/issues/1049 it is necessary to delete a symlink that is created when the container starts and it's not removed at stop.
> The location of the symbolic link can be set by the SYMLINK_PATH environment variable.

Basic usage:
```
SYMLINK_PATH="/path/to/my/symlink" ./compose.sh [option]
```

> [!NOTE]
> Users should update the `/input` and `/output` bound volume locations in `docker-compose.yml` to match their configuration.

## Development on a running docker container

With VSCode install:
- Dev Containers
- Jupyter
- Remote - SSH (optional)

Start the image with `./compose.sh jupyter` and then attach to the running container.