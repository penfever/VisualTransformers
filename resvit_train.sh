#!/bin/bash

tmux;
conda activate resvit;
cd /home/bf996/VisualTransformers;
python ResViT_Seq_HPC_burst0.py;
tmux detach