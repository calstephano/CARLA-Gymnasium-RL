#!/bin/bash

# Script to launch CARLA Viz using Docker on different systems

# Constants
DOCKER_IMAGE="mjxu96/carlaviz:0.9.15"
CARLA_PORT=2000

# Function to check if CARLA Simulator is running
check_carla() {
    if ! nc -z localhost $CARLA_PORT; then
        echo "CARLA Simulator is not running on localhost:$CARLA_PORT."
        echo "Please start the CARLA Simulator and try again."
        exit 1
    fi
}

# Function to launch CARLA Viz for Linux
run_linux() {
    echo "Launching CARLA Viz on Linux..."
    docker run -it --network="host" $DOCKER_IMAGE \
        --simulator_host localhost \
        --simulator_port $CARLA_PORT
}

# Function to launch CARLA Viz for MacOS/Windows
run_macos_windows() {
    echo "Launching CARLA Viz on MacOS/Windows..."
    docker run -it -p 8080-8081:8080-8081 $DOCKER_IMAGE \
        --simulator_host host.docker.internal \
        --simulator_port $CARLA_PORT
}

# Detect OS
case "$(uname -s)" in
    Linux*)
        OS="Linux"
        ;;
    Darwin*)
        OS="MacOS"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        OS="Windows"
        ;;
    *)
        echo "Unsupported OS. This script supports Linux, MacOS, and Windows."
        exit 1
        ;;
esac

# Main logic
echo "Detected OS: $OS"
check_carla
if [ "$OS" == "Linux" ]; then
    run_linux
else
    run_macos_windows
fi

if [ $? -eq 0 ]; then
    echo "CARLA Viz launched successfully."
    echo "Open your browser and go to http://localhost:8080 to view the interface."
else
    echo "Failed to launch CARLA Viz. Check Docker logs for more details."
fi
