# Transition PathCV optimization on Di-Alanine isomerization: Sample the boxes.
# Zilin Song, 20230915
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import Callable
import numpy as np, pickle as pkl
from helpers import get_openmm_system_and_integrator, OpenMMPlatform
from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.voronoi  import getfn_get_voronoi_box_id
from pycospath.io.tapes import ReplicaTrajectoryTape
from pycospath.rep import OpenMMReplica
from pycospath.samp import VoronoiConfinedSampler

def get_sampler_external_functions(weight_per_dof: np.ndarray, method_alignment: str) -> tuple[
                                                   Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                                   Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                                   Callable[[np.ndarray, np.ndarray], int], ]:
  """Returns the external functions for ReaxpathVoronoiConfinedSampler."""
  fn_get_weighted_aligned = getfn_get_weighted_aligned(method_alignment=method_alignment, 
                                                       weight_per_dof=weight_per_dof, )
  fn_get_rowwise_weighted_rms = getfn_get_rowwise_weighted_rms(weight_per_dof=weight_per_dof, 
                                                               num_dofs_per_atom=3, )
  fn_get_voronoi_box_id = getfn_get_voronoi_box_id(method_voronoi_boundary='hplane', 
                                                   fn_get_weighted_aligned=fn_get_weighted_aligned, 
                                                   fn_get_rowwise_weighted_rms=fn_get_rowwise_weighted_rms, )
  return fn_get_weighted_aligned, fn_get_rowwise_weighted_rms, fn_get_voronoi_box_id

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  whoami     = int(sys.argv[1].split('whoami:'    )[1])
  dir_input  = str(sys.argv[2].split('dir_input:' )[1])
  dir_output = str(sys.argv[3].split('dir_output:')[1])
  print(f'1_sampling.py INPUT: Replica {str(whoami)}: {sys.argv}', flush=True)

  ## Input file paths.
  dir_path_colvar  = os.path.join(dir_input, f'path_colvar.npy')
  dir_samplers_all = os.path.join(dir_input, f'samplers_all.pkl')
  
  ## Check DIRs
  assert os.path.exists(dir_output),       f'Invalid dir_output: {dir_output}.'
  assert os.path.isfile(dir_path_colvar),  f'Invalid dir_path_colvar: {dir_path_colvar}.'
  assert os.path.isfile(dir_samplers_all), f'Invalid dir_samplers_all: {dir_samplers_all}.'

  # Load checkpoint. 
  ## Load Path colvar
  path_colvar = np.load(dir_path_colvar)
  
  ## Unpack sampler kwargs.
  with open(dir_samplers_all, 'rb') as fi:
    samplers_all_dict = pkl.load(fi)
    assert whoami == (key_whoami:=samplers_all_dict[f'replica{str(whoami)}_whoami']), f'Bad whoami match, {whoami} / {key_whoami}.'
    whoami          = samplers_all_dict[f'replica{str(whoami)}_whoami']
    context_coord   = samplers_all_dict[f'replica{str(whoami)}_context_coord']
    weight_per_atom = samplers_all_dict[f'weight_per_atom']
    weight_per_dof  = samplers_all_dict[f'weight_per_dof']
    method_alignment = samplers_all_dict[f'method_alignment']
    method_path_tangent = samplers_all_dict[f'method_path_tangent']
  
  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Sampler - Make.
  openmm_system, openmm_integrator = get_openmm_system_and_integrator()

  replica = OpenMMReplica(whoami=whoami, 
                          context_coordinates=context_coord, 
                          weight_per_atom_dict=weight_per_atom, 
                          fn_openmm_system_init    =lambda: openmm_system, 
                          fn_openmm_integrator_init=lambda: openmm_integrator, 
                          openmm_platform_spec=OpenMMPlatform.getPlatformByName('CPU'), )
                          # openmm_properties_spec=dict(CudaPrecision='mixed'), )
  replica.initialize_context_velocities(temperature=350.)

  external_functions = get_sampler_external_functions(weight_per_dof=replica.get_replica_weight_per_dof(), 
                                                      method_alignment=method_alignment, )
  sampler = VoronoiConfinedSampler(fn_get_weighted_aligned    =external_functions[0], 
                                   fn_get_rowwise_weighted_rms=external_functions[1], 
                                   fn_get_voronoi_box_id      =external_functions[2], )
  sampler.extend(replica=replica)
  sampler.set_path_colvar(path_colvar=path_colvar)

  ## Sampler - Equilibration.
  print(f'1_sampling.py Equilibrating...: Replica {str(whoami)}', flush=True)
  sampler.set_sampler_strategy(strategy='equilibration', 
                               config_rmsd_force_constant       =1000., 
                               config_rmsd_force_cutoff_distance=0., )
  sampler.set_replica_trajectory_tape(replica_trajectory_tape=None)
  sampler.md_execute_sampling(num_batches=10_000, 
                              num_steps_per_batch=1, )
  
  ## Sampler - Production.
  print(f'1_sampling.py Production Samping...: Replica {str(whoami)}', flush=True)
  # replica.initialize_context_veloc(temperature=300.)
  sampler.set_sampler_strategy(strategy='production')
  sampler.set_replica_trajectory_tape(replica_trajectory_tape=ReplicaTrajectoryTape(), 
                                      replica_trajectory_tape_freq=10, )
  sampler.md_execute_sampling(num_batches=10_000, 
                              num_steps_per_batch=1, )
  # print(sampler._sampler_strategy._penetration_count)
  
  # OUTPUTs. ---------------------------------------------------------------------------------------
  ## Sampler_kwargs.
  with open(os.path.join(dir_output, f'sampler{str(whoami)}_chk.pkl'), 'wb') as fo:
    sampler_dict = {f'replica{str(whoami)}_whoami':        whoami, 
                    f'replica{str(whoami)}_context_coord': sampler.obtain_context_coordinates(), 
                    f'weight_per_atom':     samplers_all_dict['weight_per_atom'], 
                    f'weight_per_dof' :     samplers_all_dict['weight_per_dof'], 
                    f'method_alignment':    samplers_all_dict['method_alignment'], 
                    f'method_path_tangent': samplers_all_dict['method_path_tangent'], }
    pkl.dump(sampler_dict, fo)
  
  ## Taped data.
  sampler.get_replica_trajectory_tape().serialize(to_file=os.path.join(dir_output, f'sampler{str(whoami)}_traj'))
  
  print(f'1_sampling.py Done: Replica {str(whoami)}', flush=True)
