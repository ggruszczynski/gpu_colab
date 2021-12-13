#!/bin/bash -l
#SBATCH -J gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1 
#SBATCH --account=ALLOCATION_NAME
#SBATCH --time=01:00:00 

#SBATCH --reservation=reservation_name_if_there_is_one_for_classes # optional
#SBATCH --qos=normal # optional

module load gpu/cuda/11.4  
cd /home/$USER/folder_with_your_cuda_exercises
nvcc ex1_hello_world.cu -o ex1_hello_world
srun ex1_hello_world
