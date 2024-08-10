# Compute average box volumes.
# Zilin Song
# 

import numpy as np

if __name__ == '__main__':
  box_volumes = np.zeros((80, ))

  for sim in range(80):
    with open(f'./logs_equi/r{sim}.log', 'r') as f:
      line = f.readlines()[-1]

    box_volumes[sim] = line.split(',')[-3]
  
  print(average_box_dim:=np.power(box_volumes, 1./3.))
  print(np.mean(average_box_dim))
  print(np.max(average_box_dim))