#!/bin/bash

# Define the path to the problematic symlink on your host machine
# (Update this to the actual path if needed)
SYMLINK_PATH="/home/ubuntu/devel/SIRF-contrib"

fix_symlink() {
    if [ -L "$SYMLINK_PATH" ]; then
        echo "Removing problematic symlink..."
        rm "$SYMLINK_PATH"
    fi
}

# Set the default option to 'exec' if no argument is passed
COMMAND=${1:-exec}

case "$COMMAND" in
    exec)
        fix_symlink
        echo "Starting container to run mr_recon.sh..."
        export START_CMD="bash mr_recon.sh"
        docker compose build
        docker compose up
        ;;
    jupyter)
        fix_symlink
        echo "Starting Jupyter Notebook in the background..."
        export START_CMD=""
        docker compose build
        docker compose up -d
        ;;
    rebuild)
        fix_symlink
        echo "Rebuilding and starting with default exec..."
        docker build . --no-cache
        ;;
    down)
        echo "Stopping the container..."
        docker compose down
        ;;
    *)
        echo "Usage: $0 {exec|jupyter|rebuild|down}"
        echo "  exec    (default) Runs bash mr_recon.sh in the foreground"
        echo "  jupyter Starts Jupyter Notebook in the background and prints the URL"
        exit 1
        ;;
esac
