#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR="$SCRIPT_DIR/.."

# Sourcing .env file
source "$REPO_DIR/.env"

# Function that copies all needed files to host
copy_files_to_host() {
    sshpass -p "$PASSWD" scp -P "$PORT" \
      "${LOCAL_PATH}/roofline/roofline.cu" \
      "${LOCAL_PATH}/Makefile"\
      "${USER}@${HOST}:${REMOTE_PATH}"
}

# Main flow: Copy to host, run and copy back results
echo "Starting microbenchmark in order to get roofline model parameters."

# Copy to
copy_files_to_host
echo "Successfully copied files to host."

# Runs microbenchmark
echo "Running benchmark..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make roofline"

# Copies back the results file
echo "Done! Copying results back..."
sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/roofline.json" "${LOCAL_PATH}/results/roofline.json" 

echo "Results saved in /results/roofline.json"
