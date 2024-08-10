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
from helpers import (get_openmm_system_and_integrator, 
                     get_openmm_topology, 
                     getfn_get_weighted_aligned, 
                     PDBFile, )
from pycospath.rep import OpenMMReplica

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = 32
  dir_input  = str(sys.argv[1].split('dir_input:' )[1])
  dir_output = str(sys.argv[2].split('dir_output:')[1])

  path_colvar = np.load(f'{dir_input}/path_colvar.npy')
  samplers_dict = pkl.load(open(f'{dir_input}/samplers_all.pkl','rb'))

  print(samplers_dict.keys(), flush=True)

  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Curate inputs for sampling.
  # Framework atoms for restrained sampling.
  restrained_region = []
  for atom in get_openmm_topology().atoms():
    if atom.residue.chain.id == 'A' and atom.residue.index in [7, 8, 9, 14, 15, 16, ]: #  7, 16, 9, 14, 
      if not (    atom.name[0] == 'H' 
              or  atom.name in ['OP1', 'OP2'] 
              or (atom.residue.index in [7, 14] and atom.name in ['P', 'O5\''])
              or (atom.residue.index in [8, ] and atom.name in['C1\'', 'C2\'', 'O4\'', 
                                                               'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 
                                                               'C5', 'C6', 'C7', ])
              or (atom.residue.index in [15, ] and (atom.name not in ['O3\'', 'P', 'C5\'', 'O5\'', 'C4\'', 'C3\'']))
              ): # 'O2\'', 
        restrained_region.append(atom.index)
  print(restrained_region, flush=True)

  ordinary_replica = OpenMMReplica(whoami=0, 
                                   context_coordinates =PDBFile(f'./ms_scalebxd/fig_dna1_path/3_heat_equi/cors_equi/r0.pdb').positions, 
                                   weight_per_atom_dict=samplers_dict['weight_per_atom'],
                                   fn_openmm_system_init    =lambda: get_openmm_system_and_integrator()[0], 
                                   fn_openmm_integrator_init=lambda: get_openmm_system_and_integrator()[1], )

  restrained_replica = OpenMMReplica(whoami=0, 
                                     context_coordinates =PDBFile(f'./ms_scalebxd/fig_dna1_path/3_heat_equi/cors_equi/r0.pdb').positions, 
                                     weight_per_atom_dict=dict(zip(restrained_region, ['none' for _ in restrained_region])), 
                                     fn_openmm_system_init    =lambda: get_openmm_system_and_integrator()[0], 
                                     fn_openmm_integrator_init=lambda: get_openmm_system_and_integrator()[1], )
  
  restrained_path_colvar = np.zeros((num_replicas, len(restrained_region)*3))

  for _ in range(num_replicas):
    replica_coord = ordinary_replica.cast_to_context_coordinates(replica_coordinates=path_colvar[_, :], 
                                                                 context_coordinates=ordinary_replica.obtain_context_coordinates(), )
    restrained_replica.update_context_coordinates(context_coordinates=replica_coord)
    restrained_path_colvar[_, :] = np.copy(restrained_replica.obtain_replica_coordinates())

  ## Aligning all to first so that mean can be computed.
  get_weighted_aligned = getfn_get_weighted_aligned(method_alignment=samplers_dict['method_alignment'], 
                                                    weight_per_dof  =np.ones((restrained_path_colvar.shape[1], )), )
  for _ in range(1, num_replicas):
    restrained_path_colvar[_, :] =  np.copy(get_weighted_aligned(array_to_refer=restrained_path_colvar[0, :], 
                                                                 array_to_align=restrained_path_colvar[_, :], ))
  restrained_mean_colvar = np.mean(restrained_path_colvar, axis=0)

  # OUTPUTs. ---------------------------------------------------------------------------------------
  print('\nOutputing...', flush=True)
  print(restrained_region)
  
  # Path colvar PDBs.
  with open(os.path.join(dir_output, f'0final_context_restrained.pdb'), 'w') as fo_replica_colvar:
    restrained_replica.update_replica_coordinates(replica_coordinates=restrained_mean_colvar)
    replica_colvar_positions = restrained_replica.obtain_context_coordinates()
    
    PDBFile.writeModel(topology =get_openmm_topology(), 
                       positions=replica_colvar_positions, 
                       file=fo_replica_colvar, )
        
  with open(os.path.join(dir_output, f'0final_replica_restrained.pdb'), 'w') as fo_path_colvar:  
    with open(os.path.join(dir_output, f'0final_context_restrained.pdb'), 'r') as fi:
      fo_path_colvar.write(f'MODEL       {str(_):>2}                                                                  \n')
      for line in fi.readlines():
        if not (len(line.split()) < 2 or line.split()[0] != 'ATOM') and int(line.split()[1])-1 in restrained_region:
          fo_path_colvar.write(line)
      fo_path_colvar.write(f'ENDMDL                                                                          \n')