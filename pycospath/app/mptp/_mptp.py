#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The Serial Most Probable Transition Path (MPTP) calculation."""

from typing import Type

import numpy as np

from pycospath.cos import CoS

from pycospath.rep import Replica, OpenMMReplica

from pycospath.samp import VoronoiConfinedSampler

from pycospath.io.tapes import ReplicaTrajectoryTape

from pycospath.app import Path

from pycospath.app.mptp import PathEvolver

class MPTP(Path):
  """The Serial Most Probable Transition Path (MPTP) calculation based on average path."""

  def __init__(self, 
               context_coordinates_list: list, 
               replica_class:  Type[Replica], 
               replica_kwargs: dict,
               cos_class:  Type[CoS], 
               cos_kwargs: dict, 
               evolver_class:  Type[PathEvolver], 
               evolver_kwargs: dict, 
               method_alignment:    str = 'kabsch', 
               method_path_tangent: str = 'context', 
               ) -> None:
    """Initialize a Serial Most Probable Transition Path (MPTP) calculation based on average path.
    
      Args:
        context_coordinates_list (list):
          The Python list of context Coordinates objects in Replica;
        replica_class (Type[Replica]):
          The class (not instances) of the Replica;
        replica_kwargs (dict):
          Additional keywords passed to the constructor of the Replica.
        cos_class (Type[CoS]):
          The class (not instances) of the CoS algorihtm;
        cos_kwargs (dict):
          Additional keywords passed to the constructor of the CoS.
        evolver_class (TypeVoroCons):
          The class (not instances) of the PathEvolver;
        evolver_kwargs (dict):
          Additional keywords passed to the constructor of the PathEvolver.
        method_alignment (str, optional):
          The method used for molecular alignment for the Chain-of-Replicas. 
          Default: 'kabsch'. 
        method_path_tangent (str, optional):
          The method used for Path tangent vector evaluation for the CoS. This option does not 
          impact the String Method as it uses its reparametrized Path function method for tangent 
          estimation.
          Default: 'context'. 
    """
    # Temporal MPTP kwarg rules:
    #   0. Must align with unit weight_per_atom with OpenMMReplica: 
    #      see pycospath.comm.openmm.OpenMMRmsdCVForce().
    if issubclass(replica_class, OpenMMReplica):
      method_alignment = 'kabsch'

      for k in replica_kwargs['weight_per_atom'].keys():
        replica_kwargs['weight_per_atom'][k] = 1.

    Path.__init__(self, 
                  context_coordinates_list=context_coordinates_list, 
                  replica_class=replica_class, 
                  replica_kwargs=replica_kwargs, 
                  method_alignment=method_alignment, 
                  method_path_tangent=method_path_tangent, )
    
    # Temporal check for MPTP rule 1.
    assert (   self._path_weight_per_dof \
            == np.ones((self.get_num_path_replicas(), self.get_num_path_dofs()))).all()
    
    # Initialize CoS. ------------------------------------------------------------------------------
    assert issubclass(cos_class,  CoS ), "Illegal cos_class type."
    assert isinstance(cos_kwargs, dict), "Illegal cos_kwargs type."
    assert cos_class.is_constraint_based(), "Illegal non-constraint-based cos_class."

    self._cos = cos_class(**cos_kwargs)
    self._cos.implement(fn_get_path_tangent     =self.get_path_tangent, 
                        fn_get_path_weighted_rms=self.get_path_weighted_rms, )

    # Initialize PathEvolver and create Sampler for all Replicas. ----------------------------------
    assert issubclass(evolver_class,  PathEvolver), "Illegal evolver_class type."
    assert isinstance(evolver_kwargs, dict       ), "Illegal evolver_kwargs type."

    self._evolver = evolver_class(**evolver_kwargs)
    self._evolver.implement(fn_get_weighted_aligned    =self.get_weighted_aligned, 
                            fn_get_rowwise_weighted_rms=self.get_rowwise_weighted_rms, )
    
    self._sampler_list: list[VoronoiConfinedSampler] = []

    for replica in self.get_replica_list():
      sampler = self._evolver.create_sampler()
      sampler.extend(replica=replica)
      self._sampler_list.append(sampler)

    self._num_path_samplers = len(self._sampler_list)

  def get_cos(self) -> CoS:
    """Get the CoS instance hosted by this MPTP calculation."""
    return self._cos
  
  def get_evolver(self) -> PathEvolver:
    """Get the Path dynamical drift Evolver instance hosted by this MPTP calculation."""
    return self._evolver
  
  def get_sampler_list(self) -> list[VoronoiConfinedSampler]:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    """Get the list of ReaxpathVoronoiConfinedSampler on this Path."""
    return self._sampler_list
  
  def get_num_path_samplers(self) -> int:
    """Get the number of Samplers on this Path."""
    return self._num_path_samplers

  def execute(self, 
              num_epochs: int = 1,
              equilibration_num_batches_per_epoch: int = 5000, 
              equilibration_num_steps_per_batch:   int = 1, 
              equilibration_strategy_kwargs:      dict = {'config_rmsd_force_constant': 1000.,
                                                          'config_rmsd_force_cutoff_distance': 0.,}, 
              production_num_batches_per_epoch:    int = 10000, 
              production_num_steps_per_batch:      int = 1,
              production_strategy_kwargs:         dict = {}, 
              ) -> None:
    """Execute the MPTP calculation for n_epochs evolution steps.
    
      Args: 
        num_epochs (int, optional):
          The number of Path evolution steps.
          Default: 1.
        equilibration_num_batches_per_epoch (int, optional):
          The number of equilibration MD batches per epoch.
          Default: 5000.
        equilibration_num_steps_per_batch (int, optional):
          The number of equilibration MD steps per MD batch.
          Default: 1.
        equilibration_strategy_kwargs (dict, optional):
          Additional keywords passed to the constructor of the equilibration SamplerStrategy.
        production_num_batches_per_epoch (int, optional):
          The number of production MD batches per epoch.
          Default: 5000.
        production_num_steps_per_batch (int, optional):
          The number of production MD steps per MD batch.
          Default: 1.
        production_strategy_kwargs (dict, optional):
          Additional keywords passed to the constructor of the production SamplerStrategy.
    """
    cos, evolver, samplers = self.get_cos(), self.get_evolver(), self.get_sampler_list()

    for _ in range(num_epochs):
      # Get the Path colvar.
      path_colvar = self.get_path_colvar()
      
      # The Average Path colvar placeholder.
      path_colvar_evolved = np.zeros(path_colvar.shape)

      # Apply Path evolution.
      # Do sampling with each Sampler.
      for i in range(self.get_num_path_samplers()):
        sampler = samplers[i]

        # Set Path colvar.
        sampler.set_path_colvar(path_colvar=path_colvar)

        # Equilibration phase.
        sampler.set_sampler_strategy('equilibration', **equilibration_strategy_kwargs)
        
        # TODO add more equilibration configs here.
        sampler.set_replica_trajectory_tape(replica_trajectory_tape=None)
        sampler.md_execute_sampling(num_batches=equilibration_num_batches_per_epoch, 
                                    num_steps_per_batch=equilibration_num_steps_per_batch, )
        
        # Production phase.
        sampler.set_sampler_strategy('production', **production_strategy_kwargs)
        
        # TODO add more production configs here.
        sampler.set_replica_trajectory_tape(replica_trajectory_tape=ReplicaTrajectoryTape(), 
                                            replica_trajectory_tape_freq=1, )
        sampler.md_execute_sampling(num_batches=production_num_batches_per_epoch, 
                                    num_steps_per_batch=production_num_steps_per_batch, )
        
        # Retrieve the average Path from the sampler.
        path_colvar_evolved[i, :] = evolver.get_evolved_replica_coordinates(sampler=sampler)

      # Apply Path fixing
      path_colvar_evolved = evolver.apply_path_fixing(path_colvar=path_colvar, 
                                                      path_colvar_evolved=path_colvar_evolved, )
      
      # After Path evolution: if Path constraints are to be applied.
      if cos.is_constraint_based() == True:
        path_colvar_evolved_constrained = cos.apply_constraint(path_colvar=path_colvar_evolved, 
                                                               path_energies=None, )
      
      self.set_path_colvar(path_colvar=path_colvar_evolved_constrained)


