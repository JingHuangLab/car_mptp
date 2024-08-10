# Transition PathCV optimization on Di-Alanine isomerization: Sample the boxes.
# Zilin Song, 20230915
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from typing import Tuple, Callable
import numpy as np, pickle as pkl
from openmm import unit, Platform as OpenMMPlatform, LangevinIntegrator as OpenMMLangevinIntegrator
from openmm.app import CharmmParameterSet, CharmmPsfFile, NoCutoff, HBonds, DCDFile
from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.voronoi  import getfn_get_voronoi_box_id
from pycospath.io.tapes import VoronoiReflectionTape
from pycospath.rep import OpenMMReplica
from pycospath.samp import VoronoiConfinedSampler


def get_openmm_topology() -> object:
  """Get the OpenMM Topology."""
  return CharmmPsfFile('./examples/alad_c36m/alad.psf').topology


def get_sampler_external_functions(weight_per_dof: np.ndarray, method_alignment: str) -> Tuple[
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
  i_run      = str(sys.argv[4].split('i_run:'     )[1])
  print(f'a_bxd_sampling.py INPUT: Replica {str(whoami)}: {sys.argv}', flush=True)

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
    assert whoami == samplers_all_dict[f'replica{str(whoami)}_whoami'], f'Bad whoami match.'
    whoami              = samplers_all_dict[f'replica{str(whoami)}_whoami'       ]
    context_coord       = samplers_all_dict[f'replica{str(whoami)}_context_coord']
    weight_per_atom     = samplers_all_dict[f'weight_per_atom'    ]
    weight_per_dof      = samplers_all_dict[f'weight_per_dof'     ]
    method_alignment    = samplers_all_dict[f'method_alignment'   ]
    method_path_tangent = samplers_all_dict[f'method_path_tangent']
    
  # RUNTIME. ---------------------------------------------------------------------------------------
  ## Sampler - Make.
  replica = OpenMMReplica(whoami=whoami, 
                          context_coordinates=context_coord, 
                          weight_per_atom_dict=weight_per_atom, 
                          fn_openmm_system_init    =lambda: CharmmPsfFile('./examples/alad_c36m/alad.psf').createSystem(CharmmParameterSet('./examples/alad_c36m/toppar/par_all36m_prot.prm', 
                                                                                                                                           './examples/alad_c36m/toppar/top_all36_prot.rtf', ), 
                                                                                                                        nonbondedCutoff=NoCutoff, 
                                                                                                                        constraints=HBonds, ), 
                          fn_openmm_integrator_init=lambda: OpenMMLangevinIntegrator(350.   *unit.kelvin, 
                                                                                      30.   *unit.picosecond**-1, 
                                                                                        .001*unit.picosecond, ), 
                          openmm_platform_spec=OpenMMPlatform.getPlatformByName('CPU'), )
  replica.initialize_context_velocities(temperature=350.)

  external_functions = get_sampler_external_functions(weight_per_dof=replica.get_replica_weight_per_dof(), 
                                                      method_alignment=method_alignment, )
  sampler = VoronoiConfinedSampler(fn_get_weighted_aligned    =external_functions[0], 
                                   fn_get_rowwise_weighted_rms=external_functions[1], 
                                   fn_get_voronoi_box_id      =external_functions[2], )
  sampler.extend(replica=replica)
  sampler.set_path_colvar(path_colvar=path_colvar)

  ## Sampler - Equilibration.
  print(f'a_bxd_sampling.py Equilibrating...: Replica {str(whoami)}', flush=True)
  sampler.set_sampler_strategy(strategy='equilibration', 
                               config_rmsd_force_constant       =1000., 
                               config_rmsd_force_cutoff_distance=0., )
  sampler.md_execute_sampling(num_batches=10_000, 
                              num_steps_per_batch=1, )
  
  ## Sampler - Production.
  print(f'a_bxd_sampling.py Production Samping...: Run {str(i_run)} Replica {str(whoami)}', flush=True)
  sampler.set_sampler_strategy(strategy='production')
  sampler.set_voronoi_reflection_tape(voronoi_reflection_tape=VoronoiReflectionTape())

  dcd = DCDFile(open(os.path.join(dir_output, f'run{str(i_run)}_sampler{str(whoami)}_prod.dcd'), 'wb'), topology=CharmmPsfFile('./examples/alad_c36m/alad.psf').topology, dt=.001)
  dcd.writeModel(replica.get_openmm_context().getState(getPositions=True, enforcePeriodicBox=True).getPositions())

  sampler._sampler_strategy.md_execute_on_batch_begin(sampler=sampler)

  for i_batch in range(1_000_000):
    if (i_batch+1)%1_000 == 0 and (not sampler._voronoi_reflection_tape._data_cell_index is None):
      dcd.writeModel(replica.get_openmm_context().getState(getPositions=True, enforcePeriodicBox=True).getPositions())

    if (i_batch+1)%10_000 == 0 and (not sampler._voronoi_reflection_tape._data_cell_index is None):
      n_paths = sampler._voronoi_reflection_tape._data_cell_index.shape[0]
      print(f'Run {str(i_run)} Replica {str(whoami):>2}: {i_batch+1:>10}/1,000,000. Paths: {n_paths:<10}.', flush=True)

    sampler._sampler_strategy.md_execute_on_batch_begin(sampler=sampler)
    sampler.md_execute_steps(num_steps=1)
    sampler._sampler_strategy._runtime_md_batch_counter += 1
    sampler._sampler_strategy.md_execute_on_batch_end(sampler=sampler)
    
  sampler._sampler_strategy.md_execute_on_epoch_end(sampler=sampler)
    
  # OUTPUTs. ---------------------------------------------------------------------------------------

  with open(os.path.join(dir_output, f'run{str(i_run)}_sampler{str(whoami)}_chk.pkl'), 'wb') as fo:
    sampler_dict = {f'replica{str(whoami)}_whoami':        whoami, 
                    f'replica{str(whoami)}_context_coord': sampler.obtain_context_coordinates(), }
    pkl.dump(sampler_dict, fo)

  ## Taped data.
  sampler.get_voronoi_reflection_tape().serialize(to_file=os.path.join(dir_output, f'run{i_run}_sampler{str(whoami)}_voronoi_reflection'))

  print(f'a_bxd_sampling.py Done: Replica {str(whoami)}', flush=True)
