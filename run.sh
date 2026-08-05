#!/bin/bash

set -e

# Usage: ./run.sh [-c] []
# -c: copy newer files

# Resolving flags
while getopts 'cf' OPTION; do
    case "$OPTION" in
        c) 
          COPY=1
          ;;
        f)
          FOO=1
          ;;
    esac
done

# Getting HOST and PASSWD variables
source .env

# Function that copies all needed files to host
copy_files_to_host() {
    sshpass -p "$PASSWD" scp -P "$PORT" \
      "main.cpp" \
      "Makefile" \
      "${USER}@${HOST}:${REMOTE_PATH}"
}

# Copy files if flag is passed
if [[ -n $COPY ]]; then
    echo "Copying files to host..."
    copy_files_to_host
    echo "Successfully copied all files to host."
fi

# Compiling on remote host
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make compile-on-host"
echo "Compiled"

# Run on host
sshpass -p "$PASSWD" ssh -p "$PORT" "${USER}@${HOST}" "cd $REMOTE_PATH && make run-on-host"

# Copy output from host to local
sshpass -p "$PASSWD" scp -P "$PORT" "${USER}@${HOST}:${REMOTE_PATH}/OUTPUT_SAMPLE.csv" "OUTPUT_SAMPLE.csv" 




