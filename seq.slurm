#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=sequence_learning
  
module purge;
module load anaconda3/2020.07;

RUNDIR=$SCRATCH/VisualTransformers/run-${SLURM_JOB_ID/.*};
mkdir -p $RUNDIR;
DATADIR=$SCRATCH/omniglot_merge;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda config --add channels conda-forge;
conda config --add channels pytorch;
# conda env create -p ./penv --file environment.yml;
conda activate ./penv;
export PATH=./penv/bin:$PATH;
python ResViT_Seq_HPC.py

