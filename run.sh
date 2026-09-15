#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Usage: ./run.sh [-c] []
# -c: copy files first

# Resolving flags
while getopts 'fb' OPTION; do
    case "$OPTION" in
        f)
          FOO=1
          ;;
        b) 
          BAR=1
          ;;
    esac
done

# Sourcing .env file
source "$SCRIPT_DIR/.env"

# Function that copies all needed files to host
copy_files_to_host() {
    sshpass -p "$PASSWD" scp -P "$PORT" \
      "Makefile" \
      "main.cu" \
      "util.cuh" \
      "kernels.cuh" \
      "${USER}@${HOST}:${REMOTE_PATH}"
}

# Copy files if flag is passed

echo "Copying files to host..."
copy_files_to_host
echo "Successfully copied all files to host."

echo "Compiling program..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-compile"

echo "Running..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-run"

# echo "Cleaning files..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-clean"
echo "Program Compiled."

# Copy output from host to local
# sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/OUTPUT_SAMPLE.csv" "OUTPUT_SAMPLE.csv" 
