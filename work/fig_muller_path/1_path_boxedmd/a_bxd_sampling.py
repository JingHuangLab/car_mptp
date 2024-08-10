# Boxed Molecular Dynamics on Muller potential: no force scaling.
# Zilin Song, 20231009
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import Callable

import numpy as np, pickle as pkl

from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.voronoi  import getfn_get_voronoi_box_id
from pycospath.io.tapes import VoronoiReflectionTape
from pycospath.comm.twod import TwoDMullerBrownSystem, TwoDLangevinIntegrator
from pycospath.rep import TwoDReplica
from pycospath.samp import VoronoiConfinedSampler

def get_sampler_external_functions() -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                              Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                              Callable[[np.ndarray, np.ndarray], int], ]:
  """Returns the external functions for ReaxpathVoronoiConfinedSampler."""
  fn_get_weighted_aligned = getfn_get_weighted_aligned(method_alignment='noronotr', 
                                                       weight_per_dof=np.ones((2, )), )
  fn_get_rowwise_weighted_rms = getfn_get_rowwise_weighted_rms(weight_per_dof=np.ones((2, )), 
                                                               num_dofs_per_atom=2, )
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
  i_run      = str(sys.argv[4].split('i_run:')[1])
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
    samplers_dict = pkl.load(fi)
    assert whoami == samplers_dict[f'replica{str(whoami)}_whoami'], f'Bad whoami match.'
    whoami        = samplers_dict[f'replica{str(whoami)}_whoami']
    context_coord = samplers_dict[f'replica{str(whoami)}_context_coord']


  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Sampler - Make.
  replica = TwoDReplica(whoami                 =whoami, 
                        context_coordinates    =context_coord, 
                        fn_twod_system_init    =lambda: TwoDMullerBrownSystem(),
                        fn_twod_integrator_init=lambda: TwoDLangevinIntegrator(timestep_size=  .0001, 
                                                                               friction_coef= 5.    , 
                                                                               inverse_beta =10.    , ), )
  replica.initialize_context_velocities(temperature=10.)

  external_functions = get_sampler_external_functions()
  sampler = VoronoiConfinedSampler(fn_get_weighted_aligned    =external_functions[0], 
                                   fn_get_rowwise_weighted_rms=external_functions[1], 
                                   fn_get_voronoi_box_id      =external_functions[2], )
  sampler.extend(replica=replica)
  sampler.set_path_colvar(path_colvar=path_colvar)

  ## Sampler - Equilibration.
  print(f'a_bxd_sampling.py Equilibrating...: Replica {str(whoami)}', flush=True)
  sampler.set_sampler_strategy(strategy='equilibration', 
                               config_rmsd_force_constant       =1000., 
                               config_rmsd_force_cutoff_distance=   0., )
  sampler.set_replica_trajectory_tape(replica_trajectory_tape=None)
  sampler.md_execute_sampling(num_batches=20_000, 
                              num_steps_per_batch=1, )
  
  ## Production run.
  print(f'a_bxd_sampling.py Production Samping...: Run {str(i_run)} Replica {str(whoami)}', flush=True)

  sampler.set_sampler_strategy(strategy='production')
  # sampler.set_replica_trajectory_tape(replica_trajectory_tape=ReplicaTrajectoryTape(), 
  #                                     replica_trajectory_tape_freq=1_000, )
  sampler.set_voronoi_reflection_tape(voronoi_reflection_tape=VoronoiReflectionTape())

  sampler._sampler_strategy.md_execute_on_epoch_begin(sampler=sampler)
    
  for i_batch in range(10_000_000):
    if (i_batch+1)%10_000 == 0:
      print(f'Run {str(i_run)} Replica {str(whoami)}: {i_batch+1}/1,000,000. Paths: {sampler._voronoi_reflection_tape._data_step_index.shape[0]}.', flush=True)

    sampler._sampler_strategy.md_execute_on_batch_begin(sampler=sampler)
    sampler.md_execute_steps(num_steps=1)
    sampler._sampler_strategy._runtime_md_batch_counter += 1
    sampler._sampler_strategy.md_execute_on_batch_end(sampler=sampler)
    
  sampler._sampler_strategy.md_execute_on_epoch_end(sampler=sampler)

  # OUTPUTs. ---------------------------------------------------------------------------------------
  with open(os.path.join(dir_output, f'sampler{str(whoami)}_chk.pkl'), 'wb') as fo:
    sampler_dict = {f'replica{str(whoami)}_whoami':        whoami, 
                    f'replica{str(whoami)}_context_coord': sampler.obtain_context_coordinates(), }
    pkl.dump(sampler_dict, fo)

  ## Taped data.
  sampler.get_voronoi_reflection_tape ().serialize(to_file=os.path.join(dir_output, f'run{i_run}_sampler{str(whoami)}_voronoi_reflection'))

  print(f'a_bxd_sampling.py Done: Replica {str(whoami)}', flush=True)
