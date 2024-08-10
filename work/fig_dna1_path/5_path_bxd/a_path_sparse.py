# Downsampling the PathCV.
# Zilin Song, 20240316
# 

import sys, os, time, copy
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import numpy as np, pickle as pkl
from helpers import (get_openmm_system_and_integrator, 
                     get_openmm_topology, 
                     getfn_get_path_tangent, 
                     getfn_get_path_weighted_rms, 
                     getfn_get_weighted_aligned, 
                     PDBFile, )
from pycospath.cos import CAR
from pycospath.rep import OpenMMReplica
from pycospath.app import Path

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv and load data.
  num_replicas = 40
  dir_input    = str(sys.argv[1].split('dir_input:' )[1])
  dir_output   = str(sys.argv[2].split('dir_output:')[1])

  raw_path_colvar = np.load(f'{dir_input}/path_colvar.npy')
  raw_samplers_dict = pkl.load(open(f'{dir_input}/samplers_all.pkl','rb'))

  print(raw_samplers_dict.keys(), flush=True)
  
  # Down-Sampling the Path colvar. -----------------------------------------------------------------
  # Make new path colvar.
  ## Downsampling: first to 40 Replicas and CAR constraint...
  path_colvar_40 = np.zeros((40, raw_path_colvar.shape[1]))
  path_colvar_indices = [  0,  3,  5,  7,  9, 11, 
                          13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 
                          33, 35, 37, 39, 41, 43, 45, 47, 
                          49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 
                          69, 71, 73, 75, 77, 79, ]
  
  print(path_colvar_indices, len(path_colvar_indices))
  assert len(path_colvar_indices) == 40
  
  for i in range(40):
    path_colvar_40[i, :] = np.copy(raw_path_colvar[path_colvar_indices[i], :])

  print('DSampled:   ',        np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1)), flush=True)

  ## Aligning.
  get_weighted_aligned = getfn_get_weighted_aligned(method_alignment=raw_samplers_dict['method_alignment'], 
                                                    weight_per_dof  =raw_samplers_dict['weight_per_dof'], )
  
  for i in range(1, 40):
    path_colvar_40[i, :] = np.copy(get_weighted_aligned(array_to_refer=path_colvar_40[i-1, :], 
                                                        array_to_align=path_colvar_40[i  , :], ))
      
  print('Aligned:    ',        np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1)), flush=True)

  ## Constraining.
  car = CAR(cons_convergence_maxitr=2000   , 
            cons_regulr_curv_thresh= 180.  , 
            cons_regulr_grow_dscale=    .99, 
            cons_regulr_grow_thresh=   1.01, )
  car.implement(fn_get_path_tangent     =getfn_get_path_tangent(method_path_tangent=raw_samplers_dict['method_path_tangent']), 
                fn_get_path_weighted_rms=getfn_get_path_weighted_rms(weight_per_dof=raw_samplers_dict['weight_per_dof'], num_dofs_per_atom=3), )
  path_colvar_40 = car.apply_constraint(path_colvar=np.copy(path_colvar_40), path_energies=None)

  print('Constrained:',        np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar_40[1:  , :] - path_colvar_40[0:-1, :], axis=1)), flush=True)

  # path_colvar = path_colvar_40

  ## Downsampling: first to 32 Replicas and CAR constraint...
  num_replicas = 32
  path_colvar = np.zeros((num_replicas, raw_path_colvar.shape[1]))
  path_colvar_indices = [ 0,  1,  3,  4,  6,  7,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 
                         20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, ]
  
  print(path_colvar_indices, len(path_colvar_indices))
  assert len(path_colvar_indices) == num_replicas
  
  for i in range(num_replicas):
    path_colvar[i, :] = np.copy(path_colvar_40[path_colvar_indices[i], :])

  print('DSampled:   ',        np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1)), flush=True)

  ## Aligning.
  get_weighted_aligned = getfn_get_weighted_aligned(method_alignment=raw_samplers_dict['method_alignment'], 
                                                    weight_per_dof  =raw_samplers_dict['weight_per_dof'], )
  
  for i in range(1, num_replicas):
    path_colvar[i, :] = np.copy(get_weighted_aligned(array_to_refer=path_colvar[i-1, :], 
                                                     array_to_align=path_colvar[i  , :], ))
      
  print('Aligned:    ',        np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1)), flush=True)

  ## Constraining.
  car = CAR(cons_convergence_maxitr=1000   , 
            cons_regulr_curv_thresh= 180.  , 
            cons_regulr_grow_dscale=    .99, 
            cons_regulr_grow_thresh=   1.01, )
  car.implement(fn_get_path_tangent     =getfn_get_path_tangent(method_path_tangent=raw_samplers_dict['method_path_tangent']), 
                fn_get_path_weighted_rms=getfn_get_path_weighted_rms(weight_per_dof=raw_samplers_dict['weight_per_dof'], num_dofs_per_atom=3), )
  path_colvar = car.apply_constraint(path_colvar=np.copy(path_colvar), path_energies=None)

  print('Constrained:',        np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar[1:  , :] - path_colvar[0:-1, :], axis=1)), flush=True)

  # OUTPUTs for path colvar. -----------------------------------------------------------------------
  # Path colvar npys.
  np.save(f'{dir_output}/path_colvar.npy', path_colvar)

  replica = OpenMMReplica(whoami=0, 
                          context_coordinates=PDBFile(f'./ms_scalebxd/fig_dna1_path/3_heat_equi/cors_equi/r0.pdb').positions,
                          weight_per_atom_dict=raw_samplers_dict['weight_per_atom'], 
                          fn_openmm_system_init    =lambda: get_openmm_system_and_integrator()[0], 
                          fn_openmm_integrator_init=lambda: get_openmm_system_and_integrator()[1], )
  
  # Path colvar PDBs.
  for _ in range(num_replicas):

    with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'w') as fo_replica_colvar:
      replica.update_replica_coordinates(replica_coordinates=path_colvar[_, :])
      replica_colvar_positions = replica.obtain_context_coordinates()
      
      PDBFile.writeModel(topology =get_openmm_topology(), 
                         positions=replica_colvar_positions, 
                         file=fo_replica_colvar, )
        
  with open(os.path.join(dir_output, f'0final_replica_colvar.pdb'), 'w') as fo_path_colvar:
    for _ in range(num_replicas):
      
      with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'r') as fi:
        fo_path_colvar.write(f'MODEL       {str(_):>2}                                                                  \n')
        for line in fi.readlines():
          if not (len(line.split()) < 2 or line.split()[0] != 'ATOM') and int(line.split()[1])-1 in raw_samplers_dict['weight_per_atom'].keys():
            fo_path_colvar.write(line)
        fo_path_colvar.write(f'ENDMDL                                                                          \n')
  
  # Find best context coordiantes for the down-sampled path colvar. --------------------------------
  samplers_dict = {f'weight_per_atom':     raw_samplers_dict['weight_per_atom'], 
                   f'weight_per_dof' :     raw_samplers_dict['weight_per_dof'], 
                   f'method_alignment':    raw_samplers_dict['method_alignment'], 
                   f'method_path_tangent': raw_samplers_dict['method_path_tangent'], }
  
  print('Making the paths...', flush=True)
  
  tmp_path = Path(context_coordinates_list=[raw_samplers_dict[f'replica{str(_)}_context_coord'] for _ in range(80)], 
                  replica_class=OpenMMReplica, 
                  replica_kwargs={'weight_per_atom_dict':      raw_samplers_dict['weight_per_atom'], 
                                  'fn_openmm_system_init':     lambda: get_openmm_system_and_integrator()[0], 
                                  'fn_openmm_integrator_init': lambda: get_openmm_system_and_integrator()[1], }, 
                  method_alignment=raw_samplers_dict['method_alignment'], 
                  method_path_tangent=raw_samplers_dict['method_path_tangent'], )
  
  print('Making the paths... DONE.', flush=True)
  
  for i in range(num_replicas):
    aligned_colvars = np.zeros((tmp_path.get_num_path_replicas(), tmp_path.get_num_path_dofs()))
    
    for _ in range(tmp_path.get_num_path_replicas()):
      aligned_colvars[_, :] = np.copy(get_weighted_aligned(array_to_refer=np.copy(path_colvar[i, :]), 
                                                           array_to_align=tmp_path.get_replica_list()[_].obtain_replica_coordinates(), ))
    
    aligned_colvar_dists = np.linalg.norm(aligned_colvars-path_colvar[i, :], 2, axis=1)
    aligned_colvar_dists_min_index = np.argmin(aligned_colvar_dists)
    print(i, aligned_colvar_dists_min_index, aligned_colvar_dists.shape, flush=True)
    
    samplers_dict.update({
      f'replica{str(i)}_whoami': i, 
      f'replica{str(i)}_context_coord': tmp_path.get_replica_list()[aligned_colvar_dists_min_index]._openmm_context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True), })
    
  print(samplers_dict.keys(), flush=True)
  
  # OUTPUTs for context dict/PDBs. -----------------------------------------------------------------
  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_dict, fo)
    print(samplers_dict.keys(), flush=True)
  
  for i in range(num_replicas):
    with open(os.path.join(dir_output, f'chk_replica{str(i)}_coordinates.pdb'), 'w') as fo:
      PDBFile.writeModel(topology =get_openmm_topology(),
                         positions=samplers_dict[f'replica{str(i)}_context_coord'],
                         file=fo, )
  
  # Context PDB.
  print('Outputing... final_context_coordinates', flush=True)
  with open(os.path.join(dir_output, f'0final_context_coordinates.pdb'), 'w') as fo_context:
    for i in range(num_replicas):
      with open(os.path.join(dir_output, f'chk_replica{str(i)}_coordinates.pdb'), 'r') as fi:
        fo_context.write(f'MODEL       {str(_):>2}                                                                  \n')
        for line in fi.readlines():
          fo_context.write(line)
        fo_context.write(f'ENDMDL                                                                          \n')
