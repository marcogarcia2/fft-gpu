#!/bin/bash

set -e

timestamp=$(date +%Y%m%d_%H%M%S)
kernel_list=("FFT_Naive")

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
RESULTS_DIR="results/$timestamp"

# Usage: ./run.sh [-e] [-p]
# -e: runs complete benchmark experiment
# -p: runs NCU profiler on each kernel

# Resolving flags
while getopts 'ep' OPTION; do
    case "$OPTION" in
        e)
          EXPERIMENTS=1
          ;;
        p) 
          PROFILER=1
          ;;
    esac
done

# Sourcing .env file
source "$SCRIPT_DIR/.env"

# Add ping test

# Function that copies all needed files to host
copy_files_to_host() {
    sshpass -p "$PASSWD" scp -P "$PORT" \
      "$SCRIPT_DIR/Makefile" \
      "$SCRIPT_DIR/src/main.cu" \
      "$SCRIPT_DIR/src/util.cuh" \
      "$SCRIPT_DIR/src/kernels.cuh" \
      "$SCRIPT_DIR/src/harness.cuh" \
      "$SCRIPT_DIR/src/profiler.cuh" \
      "${USER}@${HOST}:${REMOTE_PATH}"
}

echo "Copying files to remote host..."
copy_files_to_host
echo "Successfully copied all files to remote host."

echo "Compiling program..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-compile"
echo "Program compiled successfully."


# Benchmarking all kernels, sizes and precisions
if [[ -n $EXPERIMENTS ]]; then
  # Running benchmark
  echo "Running experiments benchmark..."
  sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-run"

  # Copying results
  echo "Copying results back to local..."
  mkdir -p $RESULTS_DIR
  sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/Results.csv" "$RESULTS_DIR/Results.csv" 

  # Summary
  python3 summary.py $RESULTS_DIR/Results.csv
fi


# Profiling kernels with NCU
if [[ -n $PROFILER ]]; then
  mkdir -p $RESULTS_DIR
  for k in ${kernel_list[@]}; do
    # Running profiler
    echo "Profiling $k..."
    printf '%s\n' "$PASSWD" | sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-profile KERNEL=$k TIMESTAMP=$timestamp"

    # Copying result NCU file
    echo "Copying results back to local..."
    sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/${timestamp}_${k}.ncu-rep" "$RESULTS_DIR/${timestamp}_${k}.ncu-rep"
  done
fi

# echo "Cleaning files..."
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-clean"
