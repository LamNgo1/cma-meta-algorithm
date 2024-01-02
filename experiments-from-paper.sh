#!/bin/sh

conda init bash
conda activate cmabo-linux

for optimizer in 'bo' 'turbo' 'baxus'; do
   for seed in {1..10}; do
      python test-main.py --solver $optimizer -f alpine -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f levy -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f ellipsoid -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f rastrigin -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f shifted-alpine -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f shifted-levy -d 100 -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f schaffer100 -n 1000 --seed $seed
      python test-main.py --solver $optimizer -f branin500 -n 1000 --seed $seed
      python test-main.py --solver $optimizer -f rover100 -n 1000 --seed $seed
      python test-main.py --solver $optimizer -f half-cheetah -n 2000 --seed $seed
      python test-main.py --solver $optimizer -f lasso-dna -n 1000 --seed $seed
   done
done