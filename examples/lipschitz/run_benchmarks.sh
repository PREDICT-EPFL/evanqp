#!/bin/bash

CONCURRENT_PROCESSES=16
THREADS=8

for SAMPLES in 100 1000 10000; do
  for LAYERS in 1 2 3 4; do
    for HIDDEN_DIMENSION in 1 2 3 4 5 10 15 20 25 30; do
      python3 benchmark.py --samples $SAMPLES --hidden-layers $LAYERS --hidden-dimensions $HIDDEN_DIMENSION --threads $THREADS &> benchmark_data/out/${SAMPLES}_${LAYERS}_${HIDDEN_DIMENSION}.txt &

      # sync processes
      joblist=($(jobs -p))
      while (( ${#joblist[*]} >= CONCURRENT_PROCESSES )); do
          sleep 1
          joblist=($(jobs -p))
      done
    done
  done
done
