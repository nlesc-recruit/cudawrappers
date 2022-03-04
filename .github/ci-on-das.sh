#!/bin/bash

# Script configuration
logfile="ci4gpu.log"
function now() {
    echo "[$(date +"%Y-%m-%d_%H:%M:%S")]"
}
# End of script configuration

# DAS configuration
now >> $logfile
echo "CI-on-DAS - connection succesful" >> $logfile
echo "Preparing system" >> $logfile

module load cuda10.1
alias gpurun="srun -N 1 -C TitanX --gres=gpu:1"
# End of DAS configuration

# Test commands
now >> $logfile
cmake -S . -B build > $logfile 2>&1
make -C build > $logfile 2>&1
nvcc saxpy.cu -o saxpy > $logfile 2>&1

echo "Sending commands to gpu" >> $logfile
gpurun nvidia-smi > $logfile 2>&1
# End of test commands