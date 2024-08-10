# Transition PathCV optimization on DNA WC HG transition: Make initial guess.
# Zilin Song, 20231208
# 

import sys, os, time
sys.dont_write_bytecode=True
sys.path.insert(0, os.path.join(os.getcwd(), '../../..'))
sys.path.insert(0, os.getcwd())

print('SHELL: Python Paths:')
for p in sys.path:
  print(p)

print()

from typing import Callable

import numpy as np, pickle as pkl
from helpers import PDBFile, get_openmm_system_and_integrator, get_openmm_topology
from pycospath.rep import OpenMMReplica
from pycospath.app import Path
from pycospath.utils.pathtools import getfn_get_path_tangent, getfn_get_path_weighted_rms
from pycospath.cos import CAR


def get_car_external_functions(weight_per_dof: np.ndarray, 
                               method_path_tangent: str, 
                               ) -> tuple[Callable, Callable]:
  """Returns the external functions for ReaxpathVoronoiConfinedSampler."""
  fn_get_path_tangent = getfn_get_path_tangent(method_path_tangent=method_path_tangent)
  fn_get_path_weighted_rms = getfn_get_path_weighted_rms(weight_per_dof=weight_per_dof, 
                                                         num_dofs_per_atom=3, )
  return fn_get_path_tangent, fn_get_path_weighted_rms


if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_output = str(sys.argv[2].split('dir_output:')[1])
  method_alignment = 'kabsch'
  method_path_tangent = 'cspline'

  assert num_replicas == 80,         f'Invalid num_replicas not equals 48.'
  assert os.path.exists(dir_output), f'Invalid dir_output {dir_output}.'

  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Read coordinates from constrained optimization.
  pdb_files = [PDBFile(f'./ms_scalebxd/fig_dna0_path/3_heat_equi/cors_equi/r{_}.pdb') for _ in range(num_replicas)]

  ## Curate inputs for sampling.
  replica_atoms = []
  for atom in get_openmm_topology().atoms():
    if atom.residue.chain.id == 'A' and atom.residue.index in [7, 8, 9, 14, 15, 16, ]: #  7, 16, 9, 14, 
      if not (    atom.name[0] == 'H' 
              or  atom.name in ['OP1', 'OP2'] 
              or (atom.residue.index in [7, 14] and atom.name in ['P', 'O5\''])
              ): # 'O2\'', 
        replica_atoms.append(atom.index)

  weight_per_atom = dict(zip(replica_atoms, ['none' for _ in range(len(replica_atoms))]))
  print(weight_per_atom, flush=True)

  samplers_all_dict = {'weight_per_atom':     weight_per_atom, 
                       'method_alignment':    method_alignment, 
                       'method_path_tangent': method_path_tangent, }
  
  context_coords = [pdb_file.positions for pdb_file in pdb_files]

  for whoami in range(num_replicas):
    samplers_all_dict.update({f'replica{str(whoami)}_whoami':        whoami,       
                              f'replica{str(whoami)}_context_coord': context_coords[whoami], })
  
  print('\nMaking the temperal path.', flush=True)

  tmp_path = Path(context_coordinates_list=context_coords, 
                  replica_class=OpenMMReplica, 
                  replica_kwargs={'weight_per_atom_dict':      weight_per_atom, 
                                  'fn_openmm_system_init':     lambda: get_openmm_system_and_integrator()[0], 
                                  'fn_openmm_integrator_init': lambda: get_openmm_system_and_integrator()[1], }, 
                  method_alignment=method_alignment, 
                  method_path_tangent=method_path_tangent, )
  samplers_all_dict.update({'weight_per_dof': tmp_path.get_path_weight_per_dof(), })
  
  external_functions = get_car_external_functions(weight_per_dof=samplers_all_dict['weight_per_dof'], 
                                                  method_path_tangent=samplers_all_dict['method_path_tangent'], )
  
  car = CAR(cons_regulr_curv_thresh=50., cons_convergence_maxitr=2000, cons_regulr_grow_dscale=.99, cons_regulr_grow_thresh=1.01)
  car.implement(fn_get_path_tangent=external_functions[0], fn_get_path_weighted_rms=external_functions[1])
  tmp_path.set_path_colvar(path_colvar=car.apply_constraint(path_colvar=tmp_path.get_path_colvar(), path_energies=None))

  
  # OUTPUTs. ---------------------------------------------------------------------------------------
  print('\nOutputing...', flush=True)
  
  ## Sampler kwarg dict.
  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_all_dict, fo)

  # Path colvar npy.
  np.save(f'{dir_output}/path_colvar.npy', tmp_path.get_path_colvar())
  
  # Path colvar PDB.
  replica = OpenMMReplica(whoami=0, 
                          context_coordinates =PDBFile(f'./ms_scalebxd/fig_dna0_path/3_heat_equi/cors_equi/r0.pdb').positions, 
                          weight_per_atom_dict=samplers_all_dict['weight_per_atom'], 
                          fn_openmm_system_init    =lambda: get_openmm_system_and_integrator()[0], 
                          fn_openmm_integrator_init=lambda: get_openmm_system_and_integrator()[1], )
  
  for _ in range(num_replicas):
    replica.update_replica_coordinates(replica_coordinates=tmp_path.get_path_colvar()[_, :])
    replica_colvar_positions = replica.obtain_context_coordinates()

    with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'w') as fo_replica_colvar:
      PDBFile.writeModel(topology =get_openmm_topology(), 
                         positions=replica_colvar_positions, 
                         file=fo_replica_colvar, )
        
  with open(os.path.join(dir_output, f'0final_replica_colvar.pdb'), 'w') as fo_path_colvar:
    for _ in range(num_replicas):
      
      with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'r') as fi:
        fo_path_colvar.write(f'MODEL       {str(_):>2}                                                                  \n')
        for line in fi.readlines():
          if not (len(line.split()) < 2 or line.split()[0] != 'ATOM') and int(line.split()[1])-1 in samplers_all_dict['weight_per_atom'].keys():
            fo_path_colvar.write(line)
        fo_path_colvar.write(f'ENDMDL                                                                          \n')
