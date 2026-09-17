#!/bin/bash

set -e

timestamp=$(date +%Y%m%d_%H%M%S)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Usage: ./run.sh [-f] [-b]
# -f: foo foo foo
# -b: bar bar bar

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

echo "Copying files to remote host..."
copy_files_to_host
echo "Successfully copied all files to remote host."

echo "Compiling program..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-compile"
echo "Program compiled successfully."

echo "Running..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-run"

# echo "Cleaning files..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-clean"

# Copy output from remote to local
sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/Results.csv" "results/${timestamp}.csv" 
