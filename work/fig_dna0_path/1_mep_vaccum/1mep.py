# MEP in vaccume for DNA wc -> hg transition.
# Zilin Song, 20231204
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.path.join(os.getcwd(), '../../../'))

from pycospath.cos import CAR
from pycospath.rep import OpenMMReplica
from pycospath.app.mep import StdGradientDescentPathOptimizer, StdAdaptiveMomentumPathOptimizer, MEP

from helpers import read_pdb, read_top, write_dcd, write_pdb, get_openmm_system_and_integrator


if __name__ == '__main__':
  # Setups. ----------------------------------------------------------------------------------------
  ## Set of Context coordinates.
  context_coordinates_list = [read_pdb(file_dir=f'./intpol/r{_}.pdb') for _ in range(80)]

  ## weight_per_atom_dict: Exclude from path but still optimize all protons.
  weight_per_atom_dict = {}
  for atom in read_top(file_dir='../0_setups/guess/dna.psf').atoms():
    atom_dict = {int(atom.index): 'none'} if atom.name[0] != 'H' else {int(atom.index): 'excl'}
    weight_per_atom_dict.update(atom_dict)

  ## MEP task.
  mep = MEP(context_coordinates_list=context_coordinates_list, 
            replica_class=OpenMMReplica, 
            replica_kwargs={'weight_per_atom_dict':      weight_per_atom_dict, 
                            'fn_openmm_system_init':     lambda: get_openmm_system_and_integrator()[0], 
                            'fn_openmm_integrator_init': lambda: get_openmm_system_and_integrator()[1], },
            cos_class=CAR, 
            cos_kwargs={}, 
            optimizer_class=StdGradientDescentPathOptimizer, 
            optimizer_kwargs={'config_path_fix_mode': 'none'}, 
            method_alignment='kabsch', 
            method_path_tangent='cspline', )

  # Optimizations. ---------------------------------------------------------------------------------
  print('Start free GraD optimization.')

  for _ in range(100):
    print(f'  current step: {_}', flush=True)
    mep.execute(num_steps=1)
  
  print('Done free GraD optimization.\nStart free-end AdaM optimization', flush=True)

  mep._optimizer = StdAdaptiveMomentumPathOptimizer(config_path_fix_mode='none')

  for _ in range(200):
    print(f'  current step: {_}', flush=True)
    mep.execute(num_steps=1)
  
  print('Done free-end AdaM optimization.\nStart free-end GraD optimization', flush=True)

  mep._optimizer = StdGradientDescentPathOptimizer(config_path_fix_mode='none')

  for _ in range(100):
    print(f'  current step: {_}', flush=True)
    mep.execute(num_steps=1)
  
  print('Done free-end GraD optimization.\nStart fixed-end GraD optimization', flush=True)

  mep._optimizer = StdGradientDescentPathOptimizer(config_path_fix_mode='both')

  for _ in range(100):
    print(f'  current step: {_}', flush=True)
    mep.execute(num_steps=1)
  
  print('Done GraD optimization.\nStart fixed-end downscaling GraD optimization', flush=True)

  mep._optimizer._grad_lr_scaling=.98

  for _ in range(100):
    print(f'  current step: {_}', flush=True)
    mep.execute(num_steps=1)
  
  print('Done fixed-end downscaling GraD optimization.', flush=True)

  # Outputs. ---------------------------------------------------------------------------------------
  for ener in mep.compute_path_potential_energies_and_gradients()[0]:
    print(ener, flush=True)

  all_coordinates = [replica.obtain_context_coordinates() for replica in mep.get_replica_list()]
  write_dcd(to_file='./mep/mep.dcd', 
            topology=read_top('../0_setups/guess/dna.psf'), 
            coordinates=all_coordinates, )
  
  for i in range(80):
    write_pdb(to_file=f'./mep/r{i}.pdb', 
              topology=read_top('../0_setups/guess/dna.psf'), 
              coordinate=all_coordinates[i], )
