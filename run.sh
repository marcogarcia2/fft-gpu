#!/bin/bash

set -e

# Timestamp of this run.sh execution
timestamp=$(date +%Y%m%d_%H%M%S)

# Profiler Variables List
kernel_list=("FFT_Naive" "FFT_Naive_Tabled" "FFT_Stockham_Tabled" "cuFFT")
nx_list=($((1<<10)) $((1<<14)) $((1<<18)) $((1<<22)) $((1<<24)))
precision_list=("float" "double")

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

# SSH connection multiplexing: authenticate once, reuse the connection
SSH_CTL="/tmp/ssh-ctl-$$"
sshpass -p "$PASSWD" ssh -fN -M -S "$SSH_CTL" -p "$PORT" "${USER}@${HOST}"
trap 'ssh -S "$SSH_CTL" -O exit -p "$PORT" "${USER}@${HOST}" &>/dev/null' EXIT

# Add ping test

# Function that copies all needed files to host
copy_files_to_host() {
    sshpass -p "$PASSWD" scp -o ControlPath="$SSH_CTL" -P "$PORT" \
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
sshpass -p "$PASSWD" ssh -o ControlPath="$SSH_CTL" -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-compile"
echo "Program compiled successfully."


# Benchmarking all kernels, sizes and precisions
if [[ -n $EXPERIMENTS ]]; then
  # Running benchmark
  echo "Running experiments benchmark..."
  sshpass -p "$PASSWD" ssh -o ControlPath="$SSH_CTL" -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-run"

  # Copying results
  echo "Copying results back to local..."
  mkdir -p $RESULTS_DIR
  sshpass -p "$PASSWD" scp -o ControlPath="$SSH_CTL" -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/Results.csv" "$RESULTS_DIR/Results.csv"

  # Summary
  python3 summary.py $RESULTS_DIR
fi


# Profiling kernels with NCU
if [[ -n $PROFILER ]]; then
  mkdir -p $RESULTS_DIR
  for k in "${kernel_list[@]}"; do
    for n in "${nx_list[@]}"; do
      for p in "${precision_list[@]}"; do
        # Running profiler
        echo "Profiling $k (N=$n, $p)..."
        printf '%s\n' "$PASSWD" | sshpass -p "$PASSWD" ssh -o ControlPath="$SSH_CTL" -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-profile KERNEL=\"$k\" NVALUE=$n PRECISION=$p TIMESTAMP=$timestamp && make cuda-profile-csv KERNEL=\"$k\" NVALUE=$n PRECISION=$p TIMESTAMP=$timestamp"
      done
    done
  done

  # Copying all profiling results back at once
  echo "Copying all profiling results back to local..."
  sshpass -p "$PASSWD" scp -o ControlPath="$SSH_CTL" -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/${timestamp}_*.ncu-rep" "$RESULTS_DIR/"
  sshpass -p "$PASSWD" scp -o ControlPath="$SSH_CTL" -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/${timestamp}_*.csv" "$RESULTS_DIR/"
fi

# echo "Cleaning files..."
sshpass -p "$PASSWD" ssh -o ControlPath="$SSH_CTL" -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make cuda-clean"

if [[ -n $EXPERIMENTS ]] && [[ -n $PROFILER ]]; then
  python3 plots.py $RESULTS_DIR
fi