# Transition PathCV optimization on DNA WC HG transition: Evolves the path.
# Zilin Song, 20231213
# 

import sys, os, time
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import numpy as np, pickle as pkl
from typing import Callable
from helpers import PDBFile, get_openmm_system_and_integrator, get_openmm_topology
from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.pathtools import getfn_get_path_tangent, getfn_get_path_weighted_rms
from pycospath.io.tapes import ReplicaTrajectoryTape
from pycospath.cos import CAR
from pycospath.samp import VoronoiConfinedSampler
from pycospath.app.mptp import VoronoiConfinedPathEvolver
from pycospath.rep import OpenMMReplica


def get_external_functions(weight_per_dof: np.ndarray, 
                           method_alignment: str, 
                           method_path_tangent: str, 
                           ) -> tuple[Callable, Callable, Callable, Callable]:
  """Returns the external functions for ReaxpathVoronoiConfinedSampler."""
  fn_get_weighted_aligned = getfn_get_weighted_aligned(method_alignment=method_alignment, 
                                                       weight_per_dof=weight_per_dof, )
  fn_get_rowwise_weighted_rms = getfn_get_rowwise_weighted_rms(weight_per_dof=weight_per_dof, 
                                                               num_dofs_per_atom=3, )
  fn_get_path_tangent = getfn_get_path_tangent(method_path_tangent=method_path_tangent)
  fn_get_path_weighted_rms = getfn_get_path_weighted_rms(weight_per_dof=weight_per_dof, 
                                                         num_dofs_per_atom=3, )
  return (fn_get_weighted_aligned, 
          fn_get_rowwise_weighted_rms, 
          fn_get_path_tangent, 
          fn_get_path_weighted_rms, )

class PseudoSampler(VoronoiConfinedSampler):
  """A PseudoSampler to pass the serialized Datatape to VoronoiConfinedPathEvolver."""

  def __init__(self, 
               whoami:      int, 
               path_colvar: np.ndarray, 
               traj_coords: np.ndarray, 
               weight_per_dof: np.ndarray, 
               method_alignment: str, 
               method_path_tangent: str, 
               ) -> None:
    """Create a PseudoSampler to pass the serialized Datatape to VoronoiConfinedPathEvolver."""
    external_functions = get_external_functions(weight_per_dof=weight_per_dof, 
                                                method_alignment=method_alignment, 
                                                method_path_tangent=method_path_tangent, )
    VoronoiConfinedSampler.__init__(self, 
                                    fn_get_weighted_aligned    =external_functions[0], 
                                    fn_get_rowwise_weighted_rms=external_functions[1], 
                                    fn_get_voronoi_box_id      =external_functions[2], )
    self._whoami = whoami
    self._path_colvar = path_colvar
    self._replica_trajectory_tape = ReplicaTrajectoryTape()
    self._replica_trajectory_tape._data_replica_coordinates = traj_coords

  def get_whoami(self) -> int:
    return self._whoami


class PseudoEvolver(VoronoiConfinedPathEvolver):

  def get_evolved_replica_coordinates(self, sampler: VoronoiConfinedSampler) -> np.ndarray:
    array_to_refer = sampler.get_path_colvar()[sampler.get_whoami(), :]
    array_ensemble = sampler.get_replica_trajectory_tape()._data_replica_coordinates
    array_ensemble_average = self.get_ensemble_average(array_to_refer=array_to_refer, array_ensemble=array_ensemble)
    return array_to_refer + (array_ensemble_average-array_to_refer) * .5



if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_input    = str(sys.argv[2].split('dir_input:' )[1])
  dir_output   = str(sys.argv[3].split('dir_output:')[1])
  path_fix_mode = 'none' if     int(dir_output.split('_')[-1]) <= 5  else \
                  'head' if 5 < int(dir_output.split('_')[-1]) <= 10 else \
                  'both'
  print(f'Path fixing ... {path_fix_mode}', flush=True)
  
  ## Conclude Samplers statistics. 
  samplers_all_dict = {}
  for whoami in range(num_replicas):
    dir_sampler_dict = os.path.join(dir_output, f'sampler{str(whoami)}_chk.pkl')
    assert os.path.isfile(dir_sampler_dict), f'Invalide dir_sampelr_dict: {dir_sampler_dict}.'
    
    with open(dir_sampler_dict, 'rb') as fi:
      sampler_dict = pkl.load(fi)
      assert whoami == sampler_dict[f'replica{str(whoami)}_whoami'], f'Bad whoami match.'
      samplers_all_dict.update(sampler_dict)

  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_all_dict, fo)
    print(samplers_all_dict.keys(), flush=True)
  
  ## Path variables.
  path_colvar: np.ndarray = np.load(os.path.join(dir_input, f'path_colvar.npy'))
  traj_coords_list = [np.load(os.path.join(dir_output, f'sampler{str(_)}_traj.npz')) for _ in range(num_replicas)]

  # RUNTIME. ---------------------------------------------------------------------------------------
  # Path Components.
  external_functions = get_external_functions(weight_per_dof     =samplers_all_dict['weight_per_dof'],
                                              method_alignment   =samplers_all_dict['method_alignment'], 
                                              method_path_tangent=samplers_all_dict['method_path_tangent'], )
  
  evolver = PseudoEvolver(config_path_fix_mode=path_fix_mode)
  evolver.implement(fn_get_weighted_aligned=external_functions[0], 
                    fn_get_rowwise_weighted_rms=external_functions[1], )
  
  pseudo_samplers_list: list[PseudoSampler] = []
  for whoami in range(num_replicas):
    pseudo_samplers_list.append(PseudoSampler(whoami=samplers_all_dict[f'replica{str(whoami)}_whoami'], 
                                              path_colvar=path_colvar, 
                                              traj_coords=traj_coords_list[whoami]['replica_coords'], 
                                              weight_per_dof=samplers_all_dict['weight_per_dof'], 
                                              method_alignment=samplers_all_dict['method_alignment'], 
                                              method_path_tangent=samplers_all_dict['method_path_tangent'], ))
  
  # Path Evolutions.
  path_colvar_evolved = np.zeros(path_colvar.shape)
  for whoami in range(num_replicas):
    path_colvar_evolved[whoami, :] = evolver.get_evolved_replica_coordinates(sampler=pseudo_samplers_list[whoami])

  path_colvar_evolved = evolver.apply_path_fixing(path_colvar=path_colvar, 
                                                  path_colvar_evolved=path_colvar_evolved, )

  # Align path colvar.
  for whoami in range(1, num_replicas):
    path_colvar_evolved[whoami, :] = external_functions[0](array_to_refer=path_colvar_evolved[whoami-1, :], 
                                                           array_to_align=path_colvar_evolved[whoami  , :], )
  
  print('Aligned:    ',        np.linalg.norm(path_colvar_evolved[1:  , :] - path_colvar_evolved[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar_evolved[1:  , :] - path_colvar_evolved[0:-1, :], axis=1)), flush=True)
  

  # Apply CAR constraint.
  car = CAR(cons_regulr_curv_thresh=45., cons_convergence_maxitr=2000, cons_regulr_grow_dscale=.9, cons_regulr_grow_thresh=1.1)
  car.implement(fn_get_path_tangent=external_functions[2], fn_get_path_weighted_rms=external_functions[3])
  path_colvar_new = car.apply_constraint(path_colvar=np.copy(path_colvar_evolved), path_energies=None)

  print('Constrained:',        np.linalg.norm(path_colvar_new[1:  , :] - path_colvar_new[0:-1, :], axis=1) , flush=True)
  print('TotalSum:   ', np.sum(np.linalg.norm(path_colvar_new[1:  , :] - path_colvar_new[0:-1, :], axis=1)), flush=True)
  
  
  # OUTPUTs. ---------------------------------------------------------------------------------------
  print('Outputing...', flush=True)

  # Path colvar.
  np.save(os.path.join(dir_output, 'path_colvar_evolved.npy'), path_colvar_evolved)
  np.save(os.path.join(dir_output, 'path_colvar.npy'        ), path_colvar_new)

  # Context PDB.
  print('Outputing... final_context_coordinates', flush=True)
  with open(os.path.join(dir_output, f'0final_context_coordinates.pdb'), 'w') as fo_context:
    for _ in range(num_replicas):
      with open(os.path.join(dir_output, f'chk_replica{str(_)}_coordinates.pdb'), 'r') as fi:
        fo_context.write(f'MODEL       {str(_):>2}                                                                  \n')
        for line in fi.readlines():
          fo_context.write(line)
        fo_context.write(f'ENDMDL                                                                          \n')

  # Path colvar PDB.
  replica = OpenMMReplica(whoami=0, 
                          context_coordinates =PDBFile(os.path.join(dir_output, 'chk_replica0_coordinates.pdb')).positions, 
                          weight_per_atom_dict=samplers_all_dict['weight_per_atom'], 
                          fn_openmm_system_init    =lambda: get_openmm_system_and_integrator()[0], 
                          fn_openmm_integrator_init=lambda: get_openmm_system_and_integrator()[1], )
  
  for _ in range(num_replicas):
    print(f'Outputing... chk_replica{_}_colvar', flush=True)

    with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'w') as fo_replica_colvar_constrained:
      replica.update_replica_coordinates(replica_coordinates=path_colvar_new[_, :])
      replica_colvar_constrained = replica.obtain_context_coordinates()
      PDBFile.writeModel(topology =get_openmm_topology(), 
                         positions=replica_colvar_constrained, 
                         file=fo_replica_colvar_constrained, )

  print('Outputing... final_replica_colvar', flush=True)
  
  with open(os.path.join(dir_output, f'0final_replica_colvar.pdb'), 'w') as fo_path_colvar_constrained:
    for _ in range(num_replicas):
      
      with open(os.path.join(dir_output, f'chk_replica{_}_colvar.pdb'), 'r') as fi:
        fo_path_colvar_constrained.write(f'MODEL       {str(_):>2}                                                                  \n')
        for line in fi.readlines():
          if not (len(line.split()) < 2 or line.split()[0] != 'ATOM') and int(line.split()[1])-1 in samplers_all_dict['weight_per_atom'].keys():
            fo_path_colvar_constrained.write(line)
        fo_path_colvar_constrained.write(f'ENDMDL                                                                          \n')