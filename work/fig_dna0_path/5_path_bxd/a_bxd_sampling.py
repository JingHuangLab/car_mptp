# Transition PathCV optimization on Di-Alanine isomerization: Sample the boxes.
# Zilin Song, 20240116
#

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np, pickle as pkl
from typing import Callable
from openmm import Platform as OpenMMPlatform, XmlSerializer, LocalEnergyMinimizer
from helpers import (get_openmm_system_and_integrator,
                     get_openmm_topology,
                     getfn_get_weighted_aligned, 
                     getfn_get_rowwise_weighted_rms, 
                     getfn_get_voronoi_box_id, DCDFile, PDBFile, OpenMMRmsdCVForceX)
from pycospath.io.tapes import VoronoiReflectionTape
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
  i_run      = str(sys.argv[2].split('i_run:'     )[1])
  dir_input  = str(sys.argv[3].split('dir_input:' )[1])
  dir_error  = str(sys.argv[4].split('dir_error:' )[1])
  dir_output = str(sys.argv[5].split('dir_output:')[1])
  print(f'a_bxd_sampling.py INPUT: Replica {str(whoami)}: {sys.argv}', flush=True)

  ## Check input paths.
  dir_path_colvar  = os.path.join(dir_input, f'path_colvar.npy')
  dir_samplers_all = os.path.join(dir_input, f'samplers_all.pkl')
  assert os.path.exists(dir_output),       f'Invalid dir_output: {dir_output}.'
  assert os.path.isfile(dir_path_colvar),  f'Invalid dir_path_colvar: {dir_path_colvar}.'
  assert os.path.isfile(dir_samplers_all), f'Invalid dir_samplers_all: {dir_samplers_all}.'

  # Load checkpoint.
  ## Load Path colvar
  path_colvar = np.load(dir_path_colvar)

  ## Unpack sampler kwargs.   samplers_all_dict[f'replica{str(whoami)}_context_coord']#
  with open(dir_samplers_all, 'rb') as fi:
    samplers_all_dict = pkl.load(fi)
    assert whoami == samplers_all_dict[f'replica{str(whoami)}_whoami'], f'Bad whoami match.'
    whoami              = samplers_all_dict[f'replica{str(whoami)}_whoami']    
    context_coordinates = samplers_all_dict[f'replica{str(whoami)}_context_coord']
    weight_per_atom     = samplers_all_dict[f'weight_per_atom'    ]
    weight_per_dof      = samplers_all_dict[f'weight_per_dof'     ]
    method_alignment    = samplers_all_dict[f'method_alignment'   ]
    method_path_tangent = samplers_all_dict[f'method_path_tangent']

  ## Output: free coordinates.
  with open(os.path.join(dir_output, f'sampler{str(whoami)}_init.pdb'), 'w') as f:
    PDBFile.writeFile(topology=get_openmm_topology(),
                      positions=context_coordinates,
                      file=f, )

  # RUNTIME. ---------------------------------------------------------------------------------------
  # Replica: initial minimization/equilibration with replica region fixed.
  openmm_system, openmm_integrator = get_openmm_system_and_integrator(constraints=False)
  for i_atom in weight_per_atom.keys():
    openmm_system.setParticleMass(index=i_atom, mass=0.)
  replica = OpenMMReplica(whoami=whoami,
                        context_coordinates=context_coordinates,
                        weight_per_atom_dict=weight_per_atom,
                        fn_openmm_system_init    = lambda: openmm_system,
                        fn_openmm_integrator_init= lambda: openmm_integrator,
                        openmm_platform_spec=OpenMMPlatform.getPlatformByName('CUDA'),
                        openmm_properties_spec=dict(CudaPrecision='mixed'), )
  
  get_weighted_aligned = getfn_get_weighted_aligned(method_alignment='kabsch', weight_per_dof=replica.get_replica_weight_per_dof())
  replica.update_replica_coordinates(replica_coordinates=get_weighted_aligned(array_to_refer=replica.obtain_replica_coordinates(), array_to_align=path_colvar[whoami, :]))
  ## brief minimization.
  LocalEnergyMinimizer.minimize(context=replica.get_openmm_context(), maxIterations=1000)
  replica.initialize_context_velocities(300.)
  replica.md_execute_steps(num_steps=50_000)

  ## output: minimized/equilibrated coordinates.
  with open(os.path.join(dir_output, f'sampler{str(whoami)}_cons.pdb'), 'w') as f:
    PDBFile.writeFile(topology=get_openmm_topology(),
                      positions=replica.get_openmm_context().getState(getPositions=True).getPositions(),
                      file=f, )
  context = replica.get_openmm_context()
  del replica; del context

  # Replica: equilibration/production with Voronoi PBCs.
  try: # To capture production problems.
    # Sampler - Equilibration.
    print(f'a_bxd_sampling.py EquiVoro...: Run {str(i_run)} Replica {str(whoami)}', flush=True)
    openmm_system, openmm_integrator = get_openmm_system_and_integrator()
    replica = OpenMMReplica(whoami=whoami,
                            context_coordinates=PDBFile(os.path.join(dir_output, f'sampler{str(whoami)}_cons.pdb')).positions,
                            weight_per_atom_dict=weight_per_atom,
                            fn_openmm_system_init    = lambda: openmm_system,
                            fn_openmm_integrator_init= lambda: openmm_integrator,
                            openmm_platform_spec=OpenMMPlatform.getPlatformByName('CUDA'),
                            openmm_properties_spec=dict(CudaPrecision='mixed'), )

    ## Add a restraining force in selected replica atoms to prevent far-away off path sampling.
    refcoords = replica.cast_to_context_coordinates(replica_coordinates=np.copy(path_colvar[whoami, :]), 
                                                    context_coordinates=replica.obtain_context_coordinates(asarray=False), )
  
    restrained_atoms = [225, 228, 230, 231, 233, 234, 236, 237, 238, 240, 241, 242, 243, 247,
                        250, 252, 253, 256, 257, 260, 282, 284, 285, 288, 289, 292, 294, 295, 
                        297, 298, 299, 302, 303, 304, 306, 307, 308, 309, 310, 312, 315, 317, 
                        447, 450, 452, 453, 455, 456, 458, 460, 461, 462, 463, 464, 467, 470, 
                        472, 473, 476, 477, 480, 502, 504, 505, 508, 509, 512, 514, 515, 517, 
                        518, 519, 520, 522, 523, 525, 526, 527, 528, 531, 534, 536, ]

    rmsdforce = OpenMMRmsdCVForceX(force_constant=10., 
                                   force_cutoff_distance=1., 
                                   reference_coordinates=PDBFile(os.path.join(dir_input, '0final_context_restrained.pdb')).positions, 
                                   replica_atom_indices=restrained_atoms, ) 
    replica.get_openmm_system().addForce(force=rmsdforce)
    replica.get_openmm_context().reinitialize(preserveState=True)

    ## Sampler: initial equilibration from path_colvar.
    external_functions = get_sampler_external_functions(weight_per_dof=replica.get_replica_weight_per_dof(),
                                                        method_alignment=method_alignment, )
    sampler = VoronoiConfinedSampler(fn_get_weighted_aligned    =external_functions[0],
                                     fn_get_rowwise_weighted_rms=external_functions[1],
                                     fn_get_voronoi_box_id      =external_functions[2], )
    sampler.extend(replica=replica)
    sampler.set_path_colvar(path_colvar=path_colvar)
    sampler.initialize_context_velocities(temperature=300.)

    i_batch=0

    sampler.set_sampler_strategy(strategy='equilibration',
                                 config_rmsd_force_constant       =1000.,
                                 config_rmsd_force_cutoff_distance=0., )
    sampler.md_execute_sampling(num_batches=20_000,
                                num_steps_per_batch=1, )

    ## Output: equilibrated coordinates.
    with open(os.path.join(dir_output, f'sampler{str(whoami)}_equi.pdb'), 'w') as f:
      PDBFile.writeFile(topology=get_openmm_topology(),
                        positions=replica.get_openmm_context().getState(getPositions=True).getPositions(),
                        file=f, )

    ## Sampler - Production.
    print(f'a_bxd_sampling.py ProdVoro...: Run {str(i_run)} Replica {str(whoami)}', flush=True)
    sampler.set_sampler_strategy(strategy='production')
    sampler.set_voronoi_reflection_tape(voronoi_reflection_tape=VoronoiReflectionTape())

    curr_n_paths = 0
    prev_n_paths = 0

    dcd = DCDFile(open(os.path.join(dir_output, f'dcd_sampler{str(whoami)}_prod.dcd'), 'wb'), topology=get_openmm_topology(), dt=.001)
    dcd.writeModel(replica.get_openmm_context().getState(getPositions=True, enforcePeriodicBox=True).getPositions())

    sampler._sampler_strategy.md_execute_on_batch_begin(sampler=sampler)

    for i_batch in range(2_000_000):

      if (i_batch+1)%10_000 == 0:
        dcd.writeModel(replica.get_openmm_context().getState(getPositions=True, enforcePeriodicBox=True).getPositions())
        curr_n_paths = sampler._voronoi_reflection_tape._data_cell_index.shape[0] if not sampler._voronoi_reflection_tape._data_cell_index is None else 0
        print(f'Run {str(i_run):<2} Replica {str(whoami):>2}: {str(curr_n_paths):<10} incr: {str(curr_n_paths-prev_n_paths):<5} Paths. {str(i_batch+1):>10}/2,000,000.', flush=True)

        prev_n_paths = curr_n_paths

      sampler._sampler_strategy.md_execute_on_batch_begin(sampler=sampler)

      sampler.md_execute_steps(num_steps=1)

      sampler._sampler_strategy._runtime_md_batch_counter += 1
      sampler._sampler_strategy.md_execute_on_batch_end(sampler=sampler)

    sampler._sampler_strategy.md_execute_on_epoch_end(sampler=sampler)

  except Exception as e:
    with open(os.path.join(dir_error, f'err_run{i_run}_replica{whoami}.log'), 'w') as f_err:
      workdir = './ms_scalebxd/fig_dna0_path/5_path_bxd'
      f_err.write(f'\n\nsrun -N1 -n1 -c1 --gres=gpu:1 --mem=10GB python {workdir}/a_bxd_sampling.py whoami:{whoami} i_run:{i_run} dir_input:{dir_input} dir_error:{dir_error} dir_output:{dir_output} &\n')
      f_err.write(f'\n\n## whoami:{str(whoami):<2} i_run:{str(i_run):<2} i_batch:{str(i_batch)}.\n')
      f_err.write(str(e)+'\n')
      f_err.write(str(e.__str__))
    raise e

  # OUTPUTs. ---------------------------------------------------------------------------------------
  ## Samplers dict.
  with open(os.path.join(dir_output, f'sampler{str(whoami)}_chk.pkl'), 'wb') as fo:
    sampler_dict = {f'replica{str(whoami)}_whoami':        whoami,
                    f'replica{str(whoami)}_context_coord': sampler.obtain_context_coordinates(), }
    pkl.dump(sampler_dict, fo)

  with open(os.path.join(dir_output, f'chk_sampler{whoami}.xml'), 'w') as f_xml:
    state_xml=replica.get_openmm_context().getState(getPositions=True, getVelocities=True)
    f_xml.write(XmlSerializer.serialize(state_xml))

  ## Taped data.
  sampler.get_voronoi_reflection_tape().serialize(to_file=os.path.join(dir_output, f'sampler{str(whoami)}_voronoi_reflection'))

  print(f'a_bxd_sampling.py Done: Replica {str(whoami)}', flush=True)
