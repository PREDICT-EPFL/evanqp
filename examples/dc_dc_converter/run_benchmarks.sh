#!/bin/bash

CONCURRENT_PROCESSES=4
THREADS=32

for HORIZON in 15 20; do
  for LAYERS in 1 2 3 4; do
    for HIDDEN_DIMENSION in 10 20 30 40 50; do
      python3 benchmark.py --horizon $HORIZON --hidden-layers $LAYERS --hidden-dimensions $HIDDEN_DIMENSION --threads $THREADS &> benchmark_data/out/${HORIZON}_${LAYERS}_${HIDDEN_DIMENSION}.txt &

      # sync processes
      joblist=($(jobs -p))
      while (( ${#joblist[*]} >= CONCURRENT_PROCESSES )); do
          sleep 1
          joblist=($(jobs -p))
      done
    done
  done
done
