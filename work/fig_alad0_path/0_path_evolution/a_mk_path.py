# Transition PathCV optimization on Di-Alanine isomerization: Make initial guess.
# Zilin Song, 20230915
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

print('SHELL: Python Paths:')
for p in sys.path:
  print(p)

print()

import numpy as np, pickle as pkl
from helpers import get_openmm_system_and_integrator, get_openmm_topology, PDBFile, DCDFile
from pycospath.cos import CAR
from pycospath.rep import OpenMMReplica
from pycospath.app import Path
from pycospath.app.mep import MEP, StdGradientDescentPathOptimizer


if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_output = str(sys.argv[2].split('dir_output:')[1])
  method_alignment = 'kabsch'
  method_path_tangent = 'cspline'

  assert num_replicas == 13,         f'Invalid num_replicas not equals 13.'
  assert os.path.exists(dir_output), f'Invalid dir_output {dir_output}.'

  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Make initial guess. 
  pdb_files = [PDBFile(f'./examples/alad_c36m/intpol/r{_*2}.pdb') for _ in range(num_replicas)]
  context_coords = [pdb_file.positions for pdb_file in pdb_files]

  print([ atom.name for atom in list(pdb_files[0].getTopology().atoms()) if atom.index in range(len(pdb_files[0].positions)) ])

  ## Optimize to MEP.
  mep = MEP(context_coordinates_list=context_coords, 
            replica_class=OpenMMReplica, 
            replica_kwargs={'weight_per_atom_dict':      None, 
                            'fn_openmm_system_init':     lambda: get_openmm_system_and_integrator()[0], 
                            'fn_openmm_integrator_init': lambda: get_openmm_system_and_integrator()[1], }, 
            cos_class=CAR, 
            cos_kwargs={'cons_regulr_curv_thresh': 30., }, 
            optimizer_class=StdGradientDescentPathOptimizer, 
            optimizer_kwargs={'config_path_fix_mode': 'both', }, 
            method_alignment=method_alignment, 
            method_path_tangent=method_path_tangent, )
  mep.execute(num_steps=100)

  # OUTPUTs. ---------------------------------------------------------------------------------------
  ## Curate inputs for sampling.
  ## Theta:      0,  4,  6,  8,
  ## Phi:        4,  6,  8, 14,  Oxygen:  5
  ## Psi:        6,  8, 14, 16,  Oxygen: 15
  ## Alt-theta:  8, 14, 16, 18,  CB:     10
  weight_per_atom = { 4: 'none',  6: 'none',  8: 'none', 14: 'none', 16: 'none', 
                      0: 'none', 18: 'none', 
                      5: 'none', 15: 'none', 10: 'none', }
  
  samplers_all_dict = {'weight_per_atom':     weight_per_atom, 
                       'method_alignment':    method_alignment, 
                       'method_path_tangent': method_path_tangent, }
  
  context_coords = [r.obtain_context_coordinates() for r in mep.get_replica_list()]

  for whoami in range(num_replicas):
    samplers_all_dict.update({f'replica{str(whoami)}_whoami':        whoami,
                              f'replica{str(whoami)}_context_coord': context_coords[whoami], })
    
  tmp_path = Path(context_coordinates_list=context_coords, 
                  replica_class=OpenMMReplica, 
                  replica_kwargs={'weight_per_atom_dict': weight_per_atom, 
                                  'fn_openmm_system_init':     lambda: get_openmm_system_and_integrator()[0], 
                                  'fn_openmm_integrator_init': lambda: get_openmm_system_and_integrator()[1], }, 
                  method_alignment=method_alignment, 
                  method_path_tangent=method_path_tangent, )
  samplers_all_dict.update({'weight_per_dof': tmp_path.get_path_weight_per_dof(), })

  ## Sampler kwargs. 
  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_all_dict, fo)
  
  # Dump path_colvar.
  np.save(f'{dir_output}/path_colvar.npy', tmp_path.get_path_colvar())

  # Dump instantaneous Context DCD.
  with open(f'{dir_output}/path_traj.dcd', 'wb') as fo:
    dcd = DCDFile(file=fo, topology=get_openmm_topology(), dt=.001)
    for r in tmp_path.get_replica_list():
      dcd.writeModel(r.obtain_context_coordinates())
