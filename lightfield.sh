#!/bin/bash
module purge
module load CUDA/9.2.88-GCC-7.3.0-2.30
module load Python/3.7.2-GCCcore-8.2.0
python3 main.py
