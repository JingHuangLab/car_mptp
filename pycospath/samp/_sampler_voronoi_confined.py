#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Samplers using Hardwall-constrained Voronoi boundary condition."""

from typing import Callable, TypeVar

import numpy as np

from pycospath.rep import Replica

from pycospath.samp import Sampler, SamplerStrategy

from pycospath.io.tapes import (VoronoiReflectionTape, 
                                OnsagerPathActionTape, 
                                ReplicaTrajectoryTape, )

# VoronoiConfinedSampler. ==========================================================================

class VoronoiConfinedSampler(Sampler):
  """The Hardwall Voronoi boundary confined Sampler."""

  def __init__(self, 
               fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
               fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
               fn_get_voronoi_box_id:       Callable[[np.ndarray, np.ndarray], int], 
               ) -> None:
    """Create a Hardwall Voronoi boundary confined Sampler.
    
      Args:
        fn_get_weighted_aligned (Callable[[np.ndarray, np.ndarray], np.ndarray]):
          The get_weighted_aligned callable function. 
        fn_get_rowwise_weighted_rms (Callable[[np.ndarray, np.ndarray], np.ndarray]):
          The get_rowwise_weighted_rms callable function.
        fn_get_voronoi_box_id (Callable[[np.ndarray, np.ndarray], int]):
          The get_voronoi_box_id callable function.
    """
    Sampler.__init__(self, 
                     fn_get_weighted_aligned=fn_get_weighted_aligned, 
                     fn_get_rowwise_weighted_rms=fn_get_rowwise_weighted_rms, )

    # Realize constructor prompt methods.
    assert callable(fn_get_voronoi_box_id), "Illegal non-callable fn_get_voronoi_box_id."
    self._get_voronoi_box_id = fn_get_voronoi_box_id

    # Data Tapes and configs.
    self._replica_trajectory_tape      = None
    self._replica_trajectory_tape_freq = 100
    self._voronoi_reflection_tape      = None
    self._onsager_path_action_tape     = None

  # Constructor prompt methods. --------------------------------------------------------------------

  def get_voronoi_box_id(self,
                         voronoi_anchors:     np.ndarray, 
                         replica_coordinates: np.ndarray, 
                         ) -> np.ndarray:
    """Prompt: Get the Voronoi box ID under the Voronoi boundary conditions."""
    # Temporal fix: If SVD goes wrong, skip this step, let penetrated by default.
    return self._get_voronoi_box_id(voronoi_anchors    =voronoi_anchors, 
                                    replica_coordinates=replica_coordinates, )
  
  # Data Tapes. ------------------------------------------------------------------------------------

  def set_replica_trajectory_tape(self, 
                                  replica_trajectory_tape:      ReplicaTrajectoryTape | None, 
                                  replica_trajectory_tape_freq: int = 100, 
                                  ) -> None:
    """Set the ReplicaTrajectoryTape.
    
      Args:
        replica_trajectory_tape (ReplicaTrajectoryTape | None):
          The ReplicaTrajectoryTape. 
        replica_trajectory_tape_freq (int):
          The logging frequency of the ReplicaTrajectoryTape instance.
    """
    if replica_trajectory_tape is None:
      self._replica_trajectory_tape = None
      return
    
    self._replica_trajectory_tape = ReplicaTrajectoryTape.erase(instance=replica_trajectory_tape)
    
    assert isinstance(replica_trajectory_tape_freq,int),"Illegal replica_trajectory_tape_freq type."
    assert replica_trajectory_tape_freq > 0,            "Illegal replica_trajectory_tape_freq spec."
    self._replica_trajectory_tape_freq = replica_trajectory_tape_freq

  def get_replica_trajectory_tape(self) -> ReplicaTrajectoryTape | None: 
    """Get the ReplicaTrajectoryTape.
    
      Returns:
        replica_trajectory_tape (ReplicaTrajectoryTape | None)
        The ReplicaTrajectoryTape.
    """
    return self._replica_trajectory_tape
  
  def set_voronoi_reflection_tape(self, 
                                  voronoi_reflection_tape: VoronoiReflectionTape | None, 
                                  ) -> None:
    """Set the VoronoiReflectionTape.
    
      Args: 
        voronoi_reflection_tape (VoronoiReflectionTape | None):
          The VoronoiReflectionTape.
    """
    if voronoi_reflection_tape is None:
      self._voronoi_reflection_tape = None
      return
    
    self._voronoi_reflection_tape = VoronoiReflectionTape.erase(instance=voronoi_reflection_tape)
  
  def get_voronoi_reflection_tape(self) -> VoronoiReflectionTape | None:
    """Get the VoronoiReflectionTape.
    
      Returns:
        voronoi_reflection_tape (VoronoiReflectionTape | None):
          The VoronoiReflectionTape. 
    """
    return self._voronoi_reflection_tape
  
  def set_onsager_path_action_tape(self, 
                                   onsager_path_action_tape: OnsagerPathActionTape | None, 
                                   ) -> None:
    """Set the OnsagerPathActionTape.
    
      Args: 
        onsager_path_action_tape (OnsagerPathActionTape | None):
          The OnsagerPathActionTape.
    """
    if onsager_path_action_tape is None:
      self._onsager_path_action_tape = None
    
    tape_cls = type(onsager_path_action_tape)
    self._onsager_path_action_tape = tape_cls.erase(instance=onsager_path_action_tape)

  def get_onsager_path_action_tape(self) -> OnsagerPathActionTape | None:
    """Get the OnsagerPathActionTape.
    
      Returns:
        onsager_path_action_tape (OnsagerPathActionTape | None):
          The OnsagerPathActionTape.
    """
    return self._onsager_path_action_tape
  
  # External interfaces. ---------------------------------------------------------------------------
  
  def extend(self, replica: Replica) -> None:
    """To realize Sampler communication interfaces with Replica."""
    Sampler.extend(self, replica=replica)

    # Additional prompts. 
    self.append_rmsd_restraint_force = replica.append_rmsd_restraint_force
    self.remove_rmsd_restraint_force = replica.remove_rmsd_restraint_force

  # External interface prompts - Replica MD runtime.

  def append_rmsd_restraint_force(self, 
                                  force_constant:        float, 
                                  force_cutoff_distance: float, 
                                  reference_coordinates: np.ndarray, 
                                  ) -> object:
    """Prompt: Add a RMSD restraining force to the MD Context."""
    raise RuntimeError("Prompt method not realized in VoronoiConfinedSampler.extend().")

  def remove_rmsd_restraint_force(self, force_identity: object) -> None:
    """Prompt: Remove the RMSD restraining force from the MD context using the force_identity."""
    raise RuntimeError("Prompt method not realized in VoronoiConfinedSampler.extend().")
  
  # VoronoiConfinedSampler Strategies. -------------------------------------------------------------

  def set_sampler_strategy(self, strategy: str, **kwargs) -> None:
    """Set the SamplerStrategy adopted by this Sampler."""
    sampler_strategy_cls = VORONOI_CONFINED_SAMPLER_STRATEGIES[strategy]
    self._sampler_strategy: TVoronoiConfinedSamplerStrategy = sampler_strategy_cls(**kwargs)

  # VoronoiConfinedStrategy MD callbacks. ----------------------------------------------------------

  def md_callback_on_batch_begin(self,
                                 md_batch_count:      int, 
                                 ) -> None:
    """VoronoiConfinedSampler callback performed after the beginning of the MD batch.

      Args:
        md_batch_count (int):
          The batch count at the time of batch beginning.
    """
    # If ReplicaTrajectoryTape should be logged.
    log_replica_trajectory = (    (not self._replica_trajectory_tape is None) 
                              and (md_batch_count % self._replica_trajectory_tape_freq == 0) )

    if log_replica_trajectory == True:
      self._replica_trajectory_tape.write( (self.obtain_replica_coordinates(), ) )
    
    # If OnsagerPathActionTape should be logged.
    if not self._onsager_path_action_tape is None:
      # Coordinates at full step.
      context_coordinates_vec = self.obtain_context_coordinates(asarray=True)
      # Velocities at half step.
      context_velocities_vec  = self.obtain_context_velocities (asarray=True)
      # Gradients at full step.
      context_potential_gradients_vec = self.compute_context_potential_gradients()
      
      # Velocities at full step. 
      context_velocities_vec = self.cast_to_fullstep_velocities(
                                                      velocities=context_velocities_vec, 
                                                      mass_per_dof=self.get_context_mass_per_dof(), 
                                                      potential_gradients=context_velocities_vec, )
      
      self._onsager_path_action_tape.write( (context_coordinates_vec, 
                                             context_velocities_vec, 
                                             context_potential_gradients_vec, ) )
    
  def md_callback_on_batch_end_on_penetrated(self, 
                                             md_batch_count: int, 
                                             voronoi_box_id: int, 
                                             ) -> None:
    """VoronoiConfinedSampler callback performed at the end of the MD batch, after checking the 
      Voronoi boundary penetration and before inverting the Replica region velocities.

      Args: 
        md_batch_count (int):
          The batches count at the time of Voronoi penetration.
        voronoi_box_id (int):
          The ID of the Voronoi box that the Replica region coordinates locate immediately after the
          penetration.
    """
    # If VoronoiReflectionTape should be logged.
    if not self._voronoi_reflection_tape is None:
      self._voronoi_reflection_tape.write( (md_batch_count, 
                                            voronoi_box_id, 
                                            self.obtain_replica_coordinates(), ) )

    # If OnsagerPathActionTape should be logged.
    if not self._onsager_path_action_tape is None:
      context_coordinates_vec = self.obtain_context_coordinates(asarray=True)
      context_velocities_vec  = self.obtain_context_velocities (asarray=True)
      context_potential_gradients_vec = self.compute_context_potential_gradients()
      self._onsager_path_action_tape.write( (context_coordinates_vec, 
                                             context_velocities_vec, 
                                             context_potential_gradients_vec, ) )
      self._onsager_path_action_tape.integrate()



# VoronoiConfinedSamplerStrategy. ==================================================================

def _biased_momentum_inversion(p_replica: np.ndarray, 
                               p_anchor:  np.ndarray, 
                               p_outbox:  np.ndarray = None, ) -> tuple[np.ndarray, float]:
  """The biased momentum inversion.
  
    p_replica: replica momentum.
    p_anchor: unnormalized anchor momentum (q_anchor - q_replica).
    p_outbox: unnormalized outbox momentum (q_replica - q_outbox).
  """
  p_anchor_norm = p_anchor / np.linalg.norm(p_anchor)
  p_outbox_norm = p_outbox / np.linalg.norm(p_outbox) if not p_outbox is None else 0.
  p_replica_bias = (p_anchor_norm+p_outbox_norm) * np.linalg.norm(p_replica) - p_replica
  p_replica_bias = p_replica_bias * np.linalg.norm(p_replica) / np.linalg.norm(p_replica_bias)
  
  rand = .1 * np.random.random() # in [0., .1)
  p_biased = p_replica +  rand * p_replica_bias  
  p_biased *= (np.linalg.norm(p_replica) / np.linalg.norm(p_biased))

  return p_biased, rand


class VoronoiConfinedEquilibrationStrategy(SamplerStrategy):
  """The MD Equilibration SamplerStrategy for Hardwall Voronoi boundary confined Sampler."""

  def __init__(self, 
               config_rmsd_force_constant:        float = 100., 
               config_rmsd_force_cutoff_distance: float = 0., 
               ) -> None:
    """Create a MD Equilibration SamplerStrategy for Hardwall Voronoi boundary confined Sampler."""
    SamplerStrategy.__init__(self)

    # Configs for the RMSD bias forces. 
    self._config_rmsd_force_constant        = config_rmsd_force_constant
    self._config_rmsd_force_cutoff_distance = config_rmsd_force_cutoff_distance
  
  def md_execute_on_epoch_begin(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedEquilibrationStrategy actions performed at the begining of the MD epoch:
        0. Computes the Context temperature for (re)initializing the Replica region velocities; 
        1. Sets the runtime flag for indicating if the Replica region coordinates have entered the 
           Voronoi cell; 
        2. Sets the runtime flag for recording phase checkpoint.
        3. Adds an RMSD restraining force to pull the Replica region coordinates to the Voronoi 
           anchors, and sets the runtime flag for RMSD force identity;
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0. The Context temperature for (re)initializing the Replica region velocities
    self._runtime_initial_temperature = sampler.compute_context_temperature()

    # 1. Sets the runtime flag for indicating if the Replica region coordinates have entered the 
    #    Voronoi cell; 
    self._runtime_inside_voronoi_cell = False

    # 2. Sets the runtime flag for recording the phase checkpoint.
    self._runtime_context_phase_chkpt = None, None # (coordinates, velocities)

    # 3. Adds the RMSD restraining force.
    voronoi_anchor = sampler.get_path_colvar()[sampler.get_whoami(), :]
    self._runtime_rmsd_force_identity = sampler.append_rmsd_restraint_force(
                                      force_constant=self._config_rmsd_force_constant, 
                                      force_cutoff_distance=self._config_rmsd_force_cutoff_distance, 
                                      reference_coordinates=voronoi_anchor, )

  def md_execute_on_batch_begin(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedEquilibrationStrategy actions performed at the begining of the MD batch:
      0. Checks if the runtime flag indicates the Voronoi cell entering has took place before:
        0.a. If Yes, records the phase checkpoint;
        0.b. If  No, checks if the Replica region coordinates have entered its Voronoi cell:
          0.a.i.  If NOT entered, continue to run MD with the RMSD restraint; 
          0.b.ii. If     entered, sets the runtime flag indicating that Voronoi cell entering has 
                  taken place; removes the RMSD force and sets the runtime flag for the RMSD force
                  to None; reinitializes the Context velocities; and records the phase checkpoint.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0.a. The runtime flag indicates the Voronoi cell entering has took place before:
    #      Records the phase checkpoint.
    if self._runtime_inside_voronoi_cell == True:
      self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
                                           sampler.obtain_context_velocities (asarray=False), )
      # Callbacks.
      sampler.md_callback_on_batch_begin(md_batch_count=self.get_num_md_batches())
      return
    
    # 0.b. The runtime flag indicates the Voronoi cell entering has NOT took place before:
    #      Checks if the Replica region coordinates have entered its Voronoi cell.
    box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
    # 0.a.i.  If NOT entered:
    #         Continues to run MD with the RMSD restraint.
    if box_id != sampler.get_whoami():
      self._runtime_context_phase_chkpt = None, None # (coordinates, velocities)
      sampler.md_callback_on_batch_begin(md_batch_count=self.get_num_md_batches())
      return
      
    # 0.b.ii. If     entered:
    assert not self._runtime_initial_temperature is None, "Illegal None initial temperature."
    #         Sets the runtime flag indicating that Voronoi cell entering has taken place.
    self._runtime_inside_voronoi_cell = True
    #         Removes the RMSD force and sets the runtime flag for the RMSD force to None.
    sampler.remove_rmsd_restraint_force(force_identity=self._runtime_rmsd_force_identity)
    self._runtime_rmsd_force_identity = None
    #         Reinitializes the Context velocities.
    sampler.initialize_context_velocities(temperature=self._runtime_initial_temperature)
    #         Records the phase checkpoint.
    self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
                                         sampler.obtain_context_velocities (asarray=False), )
    #         Callbacks.
    sampler.md_callback_on_batch_begin(md_batch_count=self.get_num_md_batches())

  def md_execute_on_batch_end(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedEquilibrationStrategy actions performed at the end of the MD batch:
      0. Checks if the runtime flag indicates the Voronoi cell entering has took place before:
        0.a. If  No, does nothing and return. 
        0.b. If Yes, checks if the Replica region coordinates are inside of its Voronoi cell:
          0.b.i.  If  No, reverts to the coordinates and velocities checkpoint and inverts the 
                          Replica region velocities.
          0.b.ii. If Yes, does nothing and return.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0.a. The runtime flag indicates the Voronoi cell entering has NOT took place before:
    if self._runtime_inside_voronoi_cell == False:
      # Do nothing.
      return
    
    # 0.b. The runtime flag indicates the Voronoi cell entering has took place before:
    #      Checks if the Replica region coordinates are inside of its Voronoi cell.
    box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
    
    # 0.b.i.  If  Yes:
    if box_id == sampler.get_whoami():
      return
    
    # 0.b.ii. If No:
    #       Callbacks. 
    sampler.md_callback_on_batch_end_on_penetrated(md_batch_count=self.get_num_md_batches(), 
                                                   voronoi_box_id=box_id, )
    # Try plain inversion --------------------------------------------------------------------------
    ##      Reverts to the coordinates checkpoint.
    sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    ##      Reverts to the velocities checkpoint and inverts the Replica region velocities.
    # bonding_velocities = sampler.cast_context_to_bonding_velocities(
    #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    # inverted_context_velocities = sampler.cast_bonding_to_context_velocities(
    #                                     bonding_velocities=-1.*bonding_velocities, 
    #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    replica_velocities = sampler.cast_to_replica_velocities(
                                        context_velocities=self._runtime_context_phase_chkpt[1], )
    inverted_context_velocities = sampler.cast_to_context_velocities(
                                        replica_velocities=-1.*replica_velocities, 
                                        context_velocities=self._runtime_context_phase_chkpt[1], )
    # inverted_context_velocities = self._runtime_context_phase_chkpt[1] * -1.
    sampler.update_context_velocities(context_velocities=inverted_context_velocities)
    
    for _ in range(5):
      sampler.md_execute_steps(num_steps=1)

      box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                          replica_coordinates=sampler.obtain_replica_coordinates(), )
      
      # Plain inversion successed.
      if box_id == sampler.get_whoami():
        return
    
    # Plain inversion failed: revert back to checkpoint with inverted velocities.
    sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    sampler.update_context_velocities (context_velocities =inverted_context_velocities)

    # Try adaptive inversion. ----------------------------------------------------------------------
    # rand_prev = 0.

    for i_trial in range(500):
      self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
                                           sampler.obtain_context_velocities (asarray=False), )
      
      p_inverted = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
      p_inverted = p_inverted * sampler.get_replica_mass_per_dof()

      # Generate (biased/zig-zagging) momentums.
      q_replica = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
      q_anchor = sampler.get_weighted_aligned(array_to_refer=q_replica, 
                                              array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :], )
      p_anchor = (q_anchor - q_replica) * sampler.get_replica_mass_per_dof()

      # -------------------------------------- scratched -------------------------------------------
      # p_anchor *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_anchor))

      # if i_trial % 2 == 0:
      #   rand_curr = ((1-rand_prev) * np.random.random() + rand_prev) *.5
      #   p_corrected = p_inverted + p_anchor * rand_curr
      
      # else:
      #   q_outbox = sampler.get_weighted_aligned(array_to_refer=q_replica, 
      #                                           array_to_align=sampler.get_path_colvar()[box_id, :], )
      #   p_outbox = (q_replica - q_outbox) * sampler.get_replica_mass_per_dof()
      #   p_outbox *= (np.linalg.norm(p_anchor) / np.linalg.norm(p_outbox))
      #   p_outbox = p_anchor + p_outbox
      #   p_outbox *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_outbox))

      #   rand_curr = rand_prev
      #   p_corrected = p_inverted + p_outbox * rand_curr
        
      # p_corrected *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_corrected))
      # -------------------------------------- scratched -------------------------------------------

      if i_trial % 2 == 0:
        p_corrected, rand_curr = _biased_momentum_inversion(p_replica=p_inverted, 
                                                            p_anchor =p_anchor)

      else:
        q_outbox = sampler.get_weighted_aligned(array_to_refer=q_replica, 
                                                array_to_align=sampler.get_path_colvar()[box_id, :], )
        p_outbox = (q_replica - q_outbox) * sampler.get_replica_mass_per_dof()
        p_corrected, rand_curr = _biased_momentum_inversion(p_replica=p_inverted, 
                                                            p_anchor=p_anchor, 
                                                            p_outbox=p_outbox, )

      v_corrected = p_corrected / sampler.get_replica_mass_per_dof()
      
      # Integrate for one step with the corrected momentum.
      v_context_corrected = sampler.cast_to_context_velocities(replica_velocities=v_corrected, 
                                                               context_velocities=self._runtime_context_phase_chkpt[1], )
      sampler.update_context_velocities(context_velocities=v_context_corrected)
      sampler.md_execute_steps(num_steps=1)

      box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                          replica_coordinates=sampler.obtain_replica_coordinates(), )
      
      print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_curr} inbox {box_id}', flush=True)
      
      # x_replica = sampler.obtain_replica_coordinates()
      # dist1 = np.linalg.norm(x_replica - sampler.get_weighted_aligned(array_to_refer=x_replica, 
      #                                                                 array_to_align=sampler.get_path_colvar()[box_id, :]))
      # dist0 = np.linalg.norm(x_replica - sampler.get_weighted_aligned(array_to_refer=x_replica, 
      #                                                                 array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :]))
      # print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_curr} inbox {box_id}, danchor {dist0} / doutbox {dist1}', flush=True)
      
      # Accepted step.
      if box_id == sampler.get_whoami():
        return
      
      # # Rejected step, do another trial at the integrated phase state.
      # rand_prev = rand_curr

    # In no case should velocity inverted integration penetrates the Voronoi boundary. 
    assert False, f'Exceeded max velocity inversion trials.'

    # rand_previous = 0.

    # for i_trial in range(20):
    #   self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
    #                                        sampler.obtain_context_velocities (asarray=False), )

    #   v_replica = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
    #   p_inverted = sampler.get_replica_mass_per_dof() * v_replica

    #   q_replica = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #   q_colvar  = sampler.get_weighted_aligned(array_to_refer=q_replica, 
    #                                            array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :], ) 
    #   p_bias = sampler.get_replica_mass_per_dof() * (q_colvar - q_replica)
    #   p_bias *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_bias))

    #   rand_current = (1-rand_previous) * np.random.random() + rand_previous
    #   p_corrected = p_inverted  + p_bias * rand_current
    #   p_corrected *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_corrected))

    #   v_corrected = p_corrected / sampler.get_replica_mass_per_dof()

    #   # Integrate for one step with the corrected momentum.
    #   v_context_corrected = sampler.cast_to_context_velocities(replica_velocities=v_corrected, 
    #                                                            context_velocities=self._runtime_context_phase_chkpt[1], )
    #   sampler.update_context_velocities(context_velocities=v_context_corrected)
    #   sampler.md_execute_steps(num_steps=1)

    #   box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
    #                                       replica_coordinates=sampler.obtain_replica_coordinates(), )
      
    #   print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_current} inbox {box_id}', flush=True)
      
    #   # Accepted step.
    #   if box_id == sampler.get_whoami():
    #     return
      
    #   # Rejected step, do another trial at the integrated phase state.
    #   rand_previous = rand_current

    # # In no case should velocity inverted integration penetrates the Voronoi boundary. 
    # assert False, f'Exceeded max velocity inversion trials.'
    
    # ----------------------------------------------------------------------
    # replica_velocities = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
    # p_replica = sampler.get_replica_mass_per_dof() * -1.*replica_velocities

    # replica_coordinates = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    # replica_colvar = sampler.get_path_colvar()[sampler.get_whoami(), :]
    # replica_colvar = sampler.get_weighted_aligned(array_to_refer=replica_coordinates, array_to_align=replica_colvar)
    # p_correct = sampler.get_replica_mass_per_dof() * (replica_colvar - replica_coordinates)
    # p_correct *= (np.linalg.norm(p_replica) / np.linalg.norm(p_correct))
      
    # rand_previous = 0.
    # p_inverted = np.copy(p_replica)

    # for _ in range(21):
    #     rand_current = 0. if _ == 0 else (1-rand_previous) * np.random.random() + rand_previous
    #     p_inverted = p_replica + p_correct*rand_current
    #     p_inverted *= (np.linalg.norm(p_replica) / np.linalg.norm(p_inverted))

    #     v_inverted = p_inverted / sampler.get_replica_mass_per_dof()
    #     v_context_inverted = sampler.cast_to_context_velocities(replica_velocities=v_inverted, 
    #                                                             context_velocities=self._runtime_context_phase_chkpt[1], )
        
    #     sampler.update_context_velocities(context_velocities=v_context_inverted)
    #     sampler.md_execute_steps(num_steps=1)

    #     boxid = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
    #                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
        
    #     if boxid == sampler.get_whoami():
    #       break
        
    #     print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {_} comp: {rand_current}.', flush=True)
    #     if _ == 20:
    #       assert False
        
    #     rand_previous = rand_current
    #     sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #     sampler.update_context_velocities (context_velocities =self._runtime_context_phase_chkpt[1])

      
  def md_execute_on_epoch_end(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedEquilibrationStrategy actions performed at the end of the MD epoch:
      0. Checks if the runtime flag indicates the Voronoi cell entering has took place before:
        0.a. If  No, raises a RuntimeError;
        0.b. If Yes, does nothing.
      1. Checks if the runtime flag indicates the RMSD force has been removed:
        1.a. If  No, raises a RuntimeError;
        1.b. If Yes, does nothing.
      2. Checks if the Replica region coordinates are inside of its Voronoi cell:
        2.a. If  No, raises a RuntimeError;
        2.b. If Yes, does nothing.
      3. Resets all runtime flags.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0. The runtime flag indicates the Voronoi cell entering has NOT took place before:
    if self._runtime_inside_voronoi_cell == False:
      raise RuntimeError("The instantaneous Replica region coordinates are NOT inside of its "
                         "designated Voronoi cell.")

    # 1. The runtime flag indicates the RMSD force has NOT been removed:
    if not self._runtime_rmsd_force_identity is None:
      raise RuntimeError("The RMSD restraining force has not been removed.")
    
    # 2. The Replica region coordinates are inside of its Voronoi cell:
    box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
    if box_id != sampler.get_whoami():
      raise RuntimeError("The instantaneous Replica region coordinates are NOT inside of its "
                         "designated Voronoi cell.")
    
    # 3. Resets all runtime flags.
    self._runtime_inside_voronoi_cell = None
    self._runtime_context_phase_chkpt = None, None
    self._runtime_rmsd_force_identity = False



class VoronoiConfinedProductionStrategy(SamplerStrategy):
  """The MD Production SamplerStrategy for Hardwall Voronoi boundary confined Sampler."""

  def md_execute_on_epoch_begin(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedProductionStrategy actions performed at the begining of the MD epoch:
      0. Checks if the Replica region coordinates are inside of its Voronoi cell.
        0.a. If  No, raises a RuntimeError;
        0.b. If Yes, does nothing.
      1. Sets the runtime flag for recording phase checkpoint.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0.a. The Replica region coordinates are NOT inside of its Voronoi cell:
    box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                        replica_coordinates=sampler.obtain_replica_coordinates(), )

    if box_id != sampler.get_whoami():
      raise RuntimeError("The instantaneous Replica region coordinates are NOT inside of its "
                         "designated Voronoi cell.")
    
    # 1. Sets the runtime flag for recording phase checkpoint.
    self._runtime_context_phase_chkpt = None, None # (coordinates, velocities)
  
  def md_execute_on_batch_begin(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedProductionStrategy actions performed at the begining of the MD batch:
      0. Records the phase checkpoint.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0. Records the phase checkpoint.
    self._runtime_context_phase_chkpt=(sampler.obtain_context_coordinates(asarray=False), 
                                       sampler.obtain_context_velocities (asarray=False), )
    # Callbacks.
    sampler.md_callback_on_batch_begin(md_batch_count=self.get_num_md_batches())
  
  def md_execute_on_batch_end(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedProductionStrategy actions performed at the end of the MD batch:
      0. Checks if the Replica region coordinates are inside of its Voronoi cell:
          0.a. If  No, reverts to the coordinates and velocities checkpoint and inverts the Replica 
                       region velocities.
          0.b. If Yes, does nothing and return.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0. Checks if the Replica region coordinates are inside of its Voronoi cell.
    box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
    
    # 0.b.i.  If  Yes:
    if box_id == sampler.get_whoami():
      return
    
    # 0.b.ii. If No:
    #       Callbacks. 
    sampler.md_callback_on_batch_end_on_penetrated(md_batch_count=self.get_num_md_batches(), 
                                                   voronoi_box_id=box_id, )
    # Try plain inversion --------------------------------------------------------------------------
    ##      Reverts to the coordinates checkpoint.
    sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    ##      Reverts to the velocities checkpoint and inverts the Replica region velocities.
    # bonding_velocities = sampler.cast_context_to_bonding_velocities(
    #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    # inverted_context_velocities = sampler.cast_bonding_to_context_velocities(
    #                                     bonding_velocities=-1.*bonding_velocities, 
    #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    replica_velocities = sampler.cast_to_replica_velocities(
                                        context_velocities=self._runtime_context_phase_chkpt[1], )
    inverted_context_velocities = sampler.cast_to_context_velocities(
                                        replica_velocities=-1.*replica_velocities, 
                                        context_velocities=self._runtime_context_phase_chkpt[1], )
    # inverted_context_velocities = self._runtime_context_phase_chkpt[1] * -1.
    sampler.update_context_velocities(context_velocities=inverted_context_velocities)
    
    for _ in range(5):
      sampler.md_execute_steps(num_steps=1)

      box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                          replica_coordinates=sampler.obtain_replica_coordinates(), )
      
      # Plain inversion successed.
      if box_id == sampler.get_whoami():
        return
    
    # revert back to checkpoint.
    sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    sampler.update_context_velocities (context_velocities =inverted_context_velocities)

    # Try adaptive inversion. ----------------------------------------------------------------------
    # rand_prev = 0.

    for i_trial in range(500):
      self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
                                           sampler.obtain_context_velocities (asarray=False), )
      
      p_inverted = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
      p_inverted = p_inverted * sampler.get_replica_mass_per_dof()

      # Generate (biased/zig-zagging) momentums.
      q_replica = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
      q_anchor = sampler.get_weighted_aligned(array_to_refer=q_replica, 
                                              array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :], )
      p_anchor = (q_anchor - q_replica) * sampler.get_replica_mass_per_dof()

      # -------------------------------------- scratched -------------------------------------------
      # p_anchor *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_anchor))

      # if i_trial % 2 == 0:
      #   rand_curr = ((1-rand_prev) * np.random.random() + rand_prev) *.5
      #   p_corrected = p_inverted + p_anchor * rand_curr
      
      # else:
      #   q_outbox = sampler.get_weighted_aligned(array_to_refer=q_replica, 
      #                                           array_to_align=sampler.get_path_colvar()[box_id, :], )
      #   p_outbox = (q_replica - q_outbox) * sampler.get_replica_mass_per_dof()
      #   p_outbox *= (np.linalg.norm(p_anchor) / np.linalg.norm(p_outbox))
      #   p_outbox = p_anchor + p_outbox
      #   p_outbox *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_outbox))

      #   rand_curr = rand_prev
      #   p_corrected = p_inverted + p_outbox * rand_curr
        
      # p_corrected *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_corrected))
      # -------------------------------------- scratched -------------------------------------------

      if i_trial % 2 == 0:
        p_corrected, rand_curr = _biased_momentum_inversion(p_replica=p_inverted, 
                                                            p_anchor =p_anchor)

      else:
        q_outbox = sampler.get_weighted_aligned(array_to_refer=q_replica, 
                                                array_to_align=sampler.get_path_colvar()[box_id, :], )
        p_outbox = (q_replica - q_outbox) * sampler.get_replica_mass_per_dof()
        p_corrected, rand_curr = _biased_momentum_inversion(p_replica=p_inverted, 
                                                            p_anchor=p_anchor, 
                                                            p_outbox=p_outbox, )
      
      v_corrected = p_corrected / sampler.get_replica_mass_per_dof()
      
      # Integrate for one step with the corrected momentum.
      v_context_corrected = sampler.cast_to_context_velocities(replica_velocities=v_corrected, 
                                                               context_velocities=self._runtime_context_phase_chkpt[1], )
      sampler.update_context_velocities(context_velocities=v_context_corrected)
      sampler.md_execute_steps(num_steps=1)

      box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
                                          replica_coordinates=sampler.obtain_replica_coordinates(), )
      
      print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_curr} inbox {box_id}', flush=True)
      
      # x_replica = sampler.obtain_replica_coordinates()
      # dist1 = np.linalg.norm(x_replica - sampler.get_weighted_aligned(array_to_refer=x_replica, 
      #                                                                 array_to_align=sampler.get_path_colvar()[box_id, :]))
      # dist0 = np.linalg.norm(x_replica - sampler.get_weighted_aligned(array_to_refer=x_replica, 
      #                                                                 array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :]))
      # print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_curr} inbox {box_id}, danchor {dist0} / doutbox {dist1}', flush=True)
      
      # Accepted step.
      if box_id == sampler.get_whoami():
        return
      
      # # Rejected step, do another trial at the integrated phase state.
      # rand_prev = rand_curr

    # In no case should velocity inverted integration penetrates the Voronoi boundary. 
    assert False, f'Exceeded max velocity inversion trials.'



    # # Try stochastic inversion. --------------------------------------------------------------------
    # rand_previous = 0.

    # for i_trial in range(20):
    #   self._runtime_context_phase_chkpt = (sampler.obtain_context_coordinates(asarray=False), 
    #                                        sampler.obtain_context_velocities (asarray=False), )

    #   v_replica = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
    #   p_inverted = sampler.get_replica_mass_per_dof() * v_replica

    #   q_replica = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #   q_colvar  = sampler.get_weighted_aligned(array_to_refer=q_replica, 
    #                                            array_to_align=sampler.get_path_colvar()[sampler.get_whoami(), :], ) 
    #   p_bias = sampler.get_replica_mass_per_dof() * (q_colvar - q_replica)
    #   p_bias *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_bias))

    #   rand_current = (1-rand_previous) * np.random.random() + rand_previous
    #   p_corrected = p_inverted  + p_bias * rand_current
    #   p_corrected *= (np.linalg.norm(p_inverted) / np.linalg.norm(p_corrected))

    #   v_corrected = p_corrected / sampler.get_replica_mass_per_dof()

    #   # Integrate for one step with the corrected momentum.
    #   v_context_corrected = sampler.cast_to_context_velocities(replica_velocities=v_corrected, 
    #                                                            context_velocities=self._runtime_context_phase_chkpt[1], )
    #   sampler.update_context_velocities(context_velocities=v_context_corrected)
    #   sampler.md_execute_steps(num_steps=1)

    #   box_id = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
    #                                       replica_coordinates=sampler.obtain_replica_coordinates(), )
      
    #   print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {i_trial} comp: {rand_current} inbox {box_id}', flush=True)
      
    #   # Accepted step.
    #   if box_id == sampler.get_whoami():
    #     return
      
    #   # Rejected step, do another trial at the integrated phase state.
    #   rand_previous = rand_current

    # # In no case should velocity inverted integration penetrates the Voronoi boundary. 
    # assert False, f'Exceeded max velocity inversion trials.'

    # # 0.a. If  No:
    # if boxid != sampler.get_whoami():
    #   #    Callbacks.
    #   sampler.md_callback_on_batch_end_on_penetrated(md_batch_count=self.get_num_md_batches(), 
    #                                                  voronoi_box_id=boxid, )
    #   #    Reverts to the coordinates checkpoint.
    #   sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #   # #       Reverts to the velocities checkpoint and inverts the Replica region velocities.
    #   # replica_velocities = sampler.cast_to_replica_velocities(
    #   #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    #   # inverted_context_velocities = sampler.cast_to_context_velocities(
    #   #                                     replica_velocities=-1.*replica_velocities, 
    #   #                                     context_velocities=self._runtime_context_phase_chkpt[1], )
    #   # sampler.update_context_velocities(context_velocities=inverted_context_velocities)
    #   replica_velocities = sampler.cast_to_replica_velocities(context_velocities=self._runtime_context_phase_chkpt[1])
    #   p_replica = sampler.get_replica_mass_per_dof() * -1.*replica_velocities

    #   replica_coordinates = sampler.cast_to_replica_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #   replica_colvar = sampler.get_path_colvar()[sampler.get_whoami(), :]
    #   replica_colvar = sampler.get_weighted_aligned(array_to_refer=replica_coordinates, array_to_align=replica_colvar)
    #   p_correct = sampler.get_replica_mass_per_dof() * (replica_colvar - replica_coordinates)
    #   p_correct *= (np.linalg.norm(p_replica) / np.linalg.norm(p_correct))
      
    #   rand_previous = 0.
    #   p_inverted = np.copy(p_replica)

    #   for _ in range(21):
    #     rand_current = 0. if _ == 0 else (1-rand_previous) * np.random.random() + rand_previous
    #     p_inverted = p_replica + p_correct*rand_current
    #     p_inverted *= (np.linalg.norm(p_replica) / np.linalg.norm(p_inverted))

    #     v_inverted = p_inverted / sampler.get_replica_mass_per_dof()
    #     v_context_inverted = sampler.cast_to_context_velocities(replica_velocities=v_inverted, 
    #                                                             context_velocities=self._runtime_context_phase_chkpt[1], )
        
    #     sampler.update_context_velocities(context_velocities=v_context_inverted)
    #     sampler.md_execute_steps(num_steps=1)

    #     boxid = sampler.get_voronoi_box_id(voronoi_anchors=sampler.get_path_colvar(), 
    #                                        replica_coordinates=sampler.obtain_replica_coordinates(), )
        
    #     if boxid == sampler.get_whoami():
    #       break
        
    #     print(f'Current trial: Replica {sampler.get_whoami()}, v_invert trials: {_} comp: {rand_current}.', flush=True)
    #     if _ == 20:
    #       assert False, f'Exceeded max velocity inversion trials. {np.linalg.norm(v_inverted)}'
        
    #     rand_previous = rand_current
    #     sampler.update_context_coordinates(context_coordinates=self._runtime_context_phase_chkpt[0])
    #     sampler.update_context_velocities (context_velocities =self._runtime_context_phase_chkpt[1])


  def md_execute_on_epoch_end(self, sampler: VoronoiConfinedSampler) -> None:
    """VoronoiConfinedProductionStrategy actions performed at the end of the MD epoch:
      0. Resets all runtime flags.
      
      Args:
        sampler (VoronoiConfinedSampler):
          The VoronoiConfinedSampler to which the VoronoiConfinedEquilibrationStrategy is bounded.
    """
    # 0. Resets all runtime flags.
    self._runtime_context_phase_chkpt = None, None





TVoronoiConfinedSamplerStrategy = TypeVar("TVoronoiConfinedSamplerStrategy", 
                                          VoronoiConfinedEquilibrationStrategy, 
                                          VoronoiConfinedProductionStrategy, )

VORONOI_CONFINED_SAMPLER_STRATEGIES = {"equilibration": VoronoiConfinedEquilibrationStrategy, 
                                       "production":    VoronoiConfinedProductionStrategy, }
