# Brutal force MD for computing MFPTs from Muller upper product to lower reactant.
# Zilin Song, 20231219
# 

import sys, os

from numpy import ndarray
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

from typing import  Callable

import numpy as np, pickle as pkl

from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.voronoi  import getfn_get_voronoi_box_id
from pycospath.comm.twod import TwoDMullerBrownSystem, TwoDLangevinIntegrator
from pycospath.rep import TwoDReplica
from pycospath.samp import Sampler

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

class MullerBrownReferenceSampler(Sampler):
  """Sampler for Muller Brown potential reference MFPTs."""
  
  def __init__(self, 
               fn_get_weighted_aligned:     Callable[[ndarray, ndarray], ndarray], 
               fn_get_rowwise_weighted_rms: Callable[[ndarray, ndarray], ndarray],  
               fn_get_voronoi_box_id:       Callable[[np.ndarray, np.ndarray], int],
               ) -> None:
    Sampler.__init__(self, 
                     fn_get_weighted_aligned    =fn_get_weighted_aligned, 
                     fn_get_rowwise_weighted_rms=fn_get_rowwise_weighted_rms, )
    self.get_voronoi_box_id = fn_get_voronoi_box_id

  def get_voronoi_box_id(self,
                         voronoi_anchors:     np.ndarray, 
                         replica_coordinates: np.ndarray, 
                         ) -> np.ndarray:
    """Prompt: Get the Voronoi box ID under the Voronoi boundary conditions."""
    raise RuntimeError("Prompt method not realized in Sampler.__init__().")
  
  def set_sampler_strategy(self, strategy: str, **kwargs) -> None:
    return None
    
  def md_execute_sampling(self, 
                          target_whoami:       int, 
                          fo, 
                          num_batches:         int,
                          num_steps_per_batch: int = 1, 
                          ) -> None:
    """Execute one MD sampling Epoch. One MD Epoch consists of num_batches MD batches. Each MD batch
      consists of num_steps_per_batch steps. 
    """
    # Sanity checks.
    assert isinstance(num_batches, int),          "Illegal num_batches type."
    assert num_batches > 0,                       "Illegal num_batches spec."
    assert isinstance(num_steps_per_batch, int),  "Illegal num_steps_per_batch type."
    assert num_steps_per_batch > 0,               "Illegal num_steps_per_batch spec."
    
    # Begin epoch, check if inside of cell sampler_whoami.
    assert self.get_voronoi_box_id(voronoi_anchors    =self.get_path_colvar(), 
                                   replica_coordinates=self.obtain_replica_coordinates(), ) == self.get_whoami()
    
    self._runtime_md_batch_counter = 0

    for i_batch in range(num_batches):

      self.md_execute_steps(num_steps=num_steps_per_batch)
      self._runtime_md_batch_counter += 1
      
      # After batch, check for in cell target_whoami.
      current_whoami = self.get_voronoi_box_id(voronoi_anchors=self.get_path_colvar(), 
                                               replica_coordinates=self.obtain_replica_coordinates(), )
      
      if (i_batch+1)%20_0000 == 0:
        fo.writelines(f'Run {str(i_run):>5}: {int(i_batch+1):>14} / 200_(0000_0000) at: {current_whoami:>2}\n')
        fo.flush()

      if current_whoami == target_whoami:
        fo.writelines('In target state.\n')
        fo.flush()
        break

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  i_stage    = str(sys.argv[1].split('i_stage:'   )[1])
  i_run      = str(sys.argv[2].split('i_run:'     )[1])
  dir_output = str(sys.argv[3].split('dir_output:')[1])

  fo = open(f'./{dir_output}/stage_{i_stage}_run_{i_run}_results.log', 'w')
  fo.writelines(f'mfpt_ref.py INPUT: Replica {str(i_run)}: {sys.argv}\n')
  fo.flush()

  path_colvar_input = f'./ms_scalebxd/fig_muller_path/0_path_evolution/pathopt_7/path_colvar.npy'
  path_colvar = np.load(path_colvar_input)

  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Sampler - Make.
  whoami = 17                                                  ### Here sets the initial state. ###
  context_coord = np.copy(path_colvar[whoami, :])

  replica = TwoDReplica(whoami=whoami, 
                        context_coordinates=context_coord, 
                        fn_twod_system_init=lambda: TwoDMullerBrownSystem(), 
                        fn_twod_integrator_init=lambda: TwoDLangevinIntegrator(timestep_size=  .0001, 
                                                                               friction_coef= 5.    , 
                                                                               inverse_beta =10.    , ), )
  replica.initialize_context_velocities(temperature=10.)


  external_functions = get_sampler_external_functions()
  sampler = MullerBrownReferenceSampler(fn_get_weighted_aligned    =external_functions[0], 
                                        fn_get_rowwise_weighted_rms=external_functions[1], 
                                        fn_get_voronoi_box_id      =external_functions[2], )
  sampler.extend(replica=replica)
  sampler.set_path_colvar(path_colvar=path_colvar)

  initial_state = sampler.get_voronoi_box_id(voronoi_anchors    =sampler.get_path_colvar(), 
                                             replica_coordinates=sampler.obtain_replica_coordinates(), )
  fo.writelines(f'initial_voro_box_id: {initial_state}\n')
  fo.writelines(f'initial_context_coord: {sampler.obtain_replica_coordinates()}\n')
  fo.writelines(f'initial_context_veloc: {sampler.obtain_replica_velocities()}\n' )
  fo.writelines(f'mfpt_ref.py Production Samping...: Run {str(i_run)}\n')
  fo.flush()

  sampler.md_execute_sampling(target_whoami=0,            ### Here sets the target state. ###
                              fo=fo, 
                              num_batches=200_0000_0000,  # 10e10
                              num_steps_per_batch=1, )
  

  # OUTPUTs. ---------------------------------------------------------------------------------------
  final_state = sampler.get_voronoi_box_id(voronoi_anchors    =sampler.get_path_colvar(), 
                                           replica_coordinates=sampler.obtain_replica_coordinates(), )
  fo.write(f'final_state: {final_state}.\tnum_of_steps: {sampler._runtime_md_batch_counter} ')
  if final_state == 0: fo.write(':##@@##\n')
  fo.flush()