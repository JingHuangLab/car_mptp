# Transition PathCV optimization on Muller potential: Make initial guess.
# Zilin Song, 20231009
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

print('SHELL: Python Paths:')
for p in sys.path:
  print(p)
print()

import numpy as np, pickle as pkl
from pycospath.cos import CAR
from pycospath.comm.twod import TwoDMullerBrownSystem, TwoDLangevinIntegrator
from pycospath.rep import TwoDReplica
from pycospath.app.mep import MEP, StdGradientDescentPathOptimizer

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_output = str(sys.argv[2].split('dir_output:')[1])
  assert os.path.exists(dir_output), f'Invalid dir_output {dir_output}.'

  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Make initial guess. 
  context_coordinates_list = np.zeros((num_replicas, 2))
  context_coordinates_list[:, 1] = np.linspace(0., 2., num=num_replicas)
  context_coordinates_list = list(context_coordinates_list)
  mep = MEP(context_coordinates_list=context_coordinates_list,   
            replica_class=TwoDReplica, 
            replica_kwargs ={'fn_twod_system_init':    lambda: TwoDMullerBrownSystem(),
                            'fn_twod_integrator_init': lambda: TwoDLangevinIntegrator(), },
            cos_class=CAR,
            cos_kwargs={'cons_regulr_curv_thresh': 20., },
            optimizer_class=StdGradientDescentPathOptimizer,
            optimizer_kwargs={'config_path_fix_mode': 'none'}, # 'both'},
            method_alignment='noronotr', 
            method_path_tangent='cspline', )
  mep.execute(num_steps=1000)

  ## Linear interpolation between two minimized states.
  r = mep.get_replica_list()[ 0].obtain_context_coordinates()
  p = mep.get_replica_list()[-1].obtain_context_coordinates()
  context_coordinates_list = np.zeros((num_replicas, 2))
  for i in range(num_replicas):
    context_coordinates_list[i, :] = r+float(i)/float(num_replicas-1)*(p-r)
  context_coordinates_list = list(context_coordinates_list)
  
  # OUTPUTs. ---------------------------------------------------------------------------------------
  ## Sampler kwargs. 
  samplers_dict = {}
  for whoami in range(len(context_coordinates_list)):
    samplers_dict.update({f'replica{str(whoami)}_whoami':        whoami,
                          f'replica{str(whoami)}_context_coord': context_coordinates_list[whoami], })
  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_dict, fo)

  # Dump path_colvar.
  np.save(f'{dir_output}/path_colvar.npy', np.asarray(context_coordinates_list))

  # Dump exact path_colvar.
  np.save(f'{dir_output}/path_colvar_exact.npy', mep.get_path_colvar())
