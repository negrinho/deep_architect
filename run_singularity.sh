#!/bin/sh

cd /home/negrinho/workspace/darch_refactor

idx=0
export CUDA_VISIBLE_DEVICES=$idx 
/home/negrinho/anaconda2/bin/python -u examples/tensorflow/modified_examples/experiments/main.py \
    --gpu_id gpu$idx \
    --dataset_type mnist --searcher_type rand --num_samples 128 \
    --max_minutes_per_model 10000 > "$HOSTNAME.gpu$idx.txt" &
pid0=$!

sleep 3
idx=1
export CUDA_VISIBLE_DEVICES=$idx 
/home/negrinho/anaconda2/bin/python -u examples/tensorflow/modified_examples/experiments/main.py \
    --gpu_id gpu$idx \
    --dataset_type mnist --searcher_type rand --num_samples 128 \
    --max_minutes_per_model 10000 > "$HOSTNAME.gpu$idx.txt" & 
pid1=$!

sleep 3
idx=2
export CUDA_VISIBLE_DEVICES=$idx 
/home/negrinho/anaconda2/bin/python -u examples/tensorflow/modified_examples/experiments/main.py \
    --gpu_id gpu$idx \
    --dataset_type mnist --searcher_type smbo --num_samples 128 \
    --max_minutes_per_model 10000 > "$HOSTNAME.gpu$idx.txt" &
pid2=$!

sleep 3
idx=3
export CUDA_VISIBLE_DEVICES=$idx 
/home/negrinho/anaconda2/bin/python -u examples/tensorflow/modified_examples/experiments/main.py \
    --gpu_id gpu$idx \
    --dataset_type mnist --searcher_type smbo --num_samples 128 \
    --max_minutes_per_model 10000 > "$HOSTNAME.gpu$idx.txt" &
pid3=$!
wait $pid0 && wait $pid1 && wait $pid2 && wait $pid3
