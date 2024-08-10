# Transition PathCV optimization on Muller potential: Evolves the path.
# Zilin Song, 20231009
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import numpy as np, pickle as pkl
from typing import Tuple, Callable
from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.pathtools import getfn_get_path_tangent, getfn_get_path_weighted_rms
from pycospath.io.tapes import ReplicaTrajectoryTape
from pycospath.cos import CAR
from pycospath.samp import VoronoiConfinedSampler
from pycospath.app.mptp import VoronoiConfinedPathEvolver

def get_external_functions() -> Tuple[Callable, Callable, Callable, Callable]:
  """Returns the external functions for ReaxpathVoronoiConfinedSampler."""
  fn_get_weighted_aligned = getfn_get_weighted_aligned(method_alignment='noronotr', 
                                                       weight_per_dof=np.ones((2, )), )
  fn_get_rowwise_weighted_rms = getfn_get_rowwise_weighted_rms(weight_per_dof=np.ones((2, )), 
                                                               num_dofs_per_atom=2, )
  fn_get_path_tangent = getfn_get_path_tangent(method_path_tangent='cspline')
  fn_get_path_weighted_rms = getfn_get_path_weighted_rms(weight_per_dof=np.ones((2, )), 
                                                         num_dofs_per_atom=2, )
  return (fn_get_weighted_aligned, 
          fn_get_rowwise_weighted_rms, 
          fn_get_path_tangent, 
          fn_get_path_weighted_rms, )


class PseudoSampler(VoronoiConfinedSampler):
  """A PseudoSampler to pass the serialized Datatape to VoronoiConfinedPathEvolver."""

  def __init__(self, 
               whoami:      int, 
               path_colvar: np.ndarray, 
               traj_coords: np.ndarray, ) -> None:
    """Create a PseudoSampler to pass the serialized Datatape to VoronoiConfinedPathEvolver."""
    external_functions = get_external_functions()
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


if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_input  = str(sys.argv[2].split('dir_input:' )[1])
  dir_output = str(sys.argv[3].split('dir_output:')[1])
  
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
    print(samplers_all_dict)

  ## Path variables.
  path_colvar = np.load(os.path.join(dir_input, f'path_colvar.npy'))
  traj_coords_list = [np.load(os.path.join(dir_output, f'sampler{str(_)}_traj.npz')) for _ in range(num_replicas)]

  # RUNTIME. ---------------------------------------------------------------------------------------
  # Path Components.
  external_functions = get_external_functions()

  car = CAR(cons_regulr_curv_thresh=30., )
  car.implement(fn_get_path_tangent=external_functions[2], 
                fn_get_path_weighted_rms=external_functions[3], )

  evolver = VoronoiConfinedPathEvolver(config_path_fix_mode='both')
  evolver.implement(fn_get_weighted_aligned=external_functions[0], 
                    fn_get_rowwise_weighted_rms=external_functions[1], )

  pseudo_samplers_list = []

  for whoami in range(num_replicas):
    pseudo_samplers_list.append(PseudoSampler(whoami=samplers_all_dict[f'replica{str(whoami)}_whoami'], 
                                              path_colvar=path_colvar, 
                                              traj_coords=traj_coords_list[whoami]['replica_coords'], ))

  # Path Evolutions. 
  path_colvar_evolved = np.zeros(path_colvar.shape)

  for whoami in range(num_replicas):
    path_colvar_evolved[whoami, :] = evolver.get_evolved_replica_coordinates(sampler=pseudo_samplers_list[whoami])
  
  path_colvar_evolved = evolver.apply_path_fixing(path_colvar=path_colvar, 
                                                  path_colvar_evolved=path_colvar_evolved, )

  path_colvar_evolved_constrained = car.apply_constraint(path_colvar=path_colvar_evolved, 
                                                         path_energies=None, )
  
  # OUTPUTs. ---------------------------------------------------------------------------------------
  # Remove Cartesian redundancy.
  for whoami in range(1, num_replicas):
    path_colvar_evolved_constrained[whoami, :] = external_functions[0](array_to_refer=path_colvar_evolved_constrained[whoami-1, :], 
                                                                       array_to_align=path_colvar_evolved_constrained[whoami  , :], )

  np.save(os.path.join(dir_output, 'path_colvar_evolved.npy'), path_colvar_evolved)
  np.save(os.path.join(dir_output, 'path_colvar.npy'        ), path_colvar_evolved_constrained)
