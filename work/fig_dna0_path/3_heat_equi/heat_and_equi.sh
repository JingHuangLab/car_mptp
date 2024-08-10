#/bin/bash

mkdir -p ./logs_heat ./cors_heat

for i_replica in {0..79}
do {
  srun -N 1 -n 1 -c 1 --gres=gpu:1 python 0heat.py ${i_replica} & 
} done

wait

mkdir -p ./logs_equi ./cors_equi

for i_replica in {0..79}
do {
  srun -N 1 -n 1 -c 1 --gres=gpu:1 python 1equi.py ${i_replica} & 
} done

wait

python ./stat.py

python ./mkdcd.py