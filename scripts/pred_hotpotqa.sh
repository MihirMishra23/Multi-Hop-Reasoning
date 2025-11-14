#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm

python scripts/run_agent.py --dataset hotpotqa --setting distractor --split validation --method icl --model llama-3.2-1b-instruct --batch-size 10 --batch-number 1 --num-batches 10

