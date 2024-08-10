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
  
# [6.24597381 6.2380309  6.24122112 6.24347593 6.24849496 6.2338674
#  6.24526253 6.25560695 6.23665265 6.25520483 6.25077333 6.24168171
#  6.24153762 6.24740028 6.23165376 6.25231864 6.22944135 6.24397956
#  6.25122598 6.23186737 6.23705723 6.23947643 6.2377862  6.24332896
#  6.23862429 6.24920984 6.24922707 6.23970436 6.24784234 6.23295767
#  6.23705984 6.23470927 6.24467605 6.24101615 6.24732235 6.24432269
#  6.24101011 6.24438702 6.22944252 6.2342225  6.23071162 6.24262087
#  6.24136657 6.24597093 6.25384864 6.23206401 6.24795486 6.24778291
#  6.24819246 6.23825328 6.23879065 6.24529817 6.24021668 6.24010137
#  6.24164946 6.24845078 6.24082082 6.23756499 6.24381177 6.25104041
#  6.24375623 6.23609922 6.23238188 6.24396062 6.2281533  6.2442887
#  6.24039799 6.24359726 6.24750437 6.23829554 6.24111293 6.24421499
#  6.23604032 6.24132703 6.23974959 6.2391724  6.23745853 6.24093662
#  6.24845003 6.2320818 ]
# 6.241681802180063
# 6.25560695026544