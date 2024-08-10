#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""2D Replica."""

from typing import Callable

import numpy as np

from pycospath.rep import Replica

from pycospath.comm.twod import (TwoDRmsdCVForce, 
                                 TwoDSystem, 
                                 TwoDIntegrator, 
                                 TwoDLangevinIntegrator,        # Thermostated Integrator.
                                 TwoDScaledLangevinIntegrator,  # Thermostated Integrator.
                                 TwoDContext, )

class TwoDReplica(Replica):
  """The class of Replicas on TwoD potential."""

  # TwoDContext initializer. -----------------------------------------------------------------------

  def _before__init__(self,
                      fn_twod_system_init:     Callable[[], TwoDSystem    ],
                      fn_twod_integrator_init: Callable[[], TwoDIntegrator], 
                      ) -> None:
    """To realize the 2D Context. Autoexecuted as the first method in Replica.__init__()."""
    # Build the 2D system from the corresponding init_fn.
    twod_system: TwoDSystem = fn_twod_system_init()
    assert isinstance(twod_system, TwoDSystem), "Illegal fn_twod_system_init return type."
    
    # Build the 2D integrator from the corresponding init_fn.
    twod_integrator: TwoDIntegrator = fn_twod_integrator_init()
    assert isinstance(twod_integrator,TwoDIntegrator),"Illegal fn_twod_integrator_init return type."
    
    # TwoDReplica specific members.
    self._twod_context = TwoDContext(twod_system    =twod_system, 
                                     twod_integrator=twod_integrator, )
    
    self._context_mass_per_atom = np.asarray( [twod_system.get_particle_mass()] )
    self._context_mass_per_dof  = np.repeat(self.get_context_mass_per_atom()[:, np.newaxis], 
                                           repeats=2, 
                                           axis=1, )
    self._context_num_atoms = 1
    self._context_num_dofs_per_atom = 2
    self._context_velocities_time_offset = twod_integrator.get_velocities_time_offset()

  def get_context_num_atoms(self) -> int:
    """Get the number of atoms in the 2D Context."""
    return self._context_num_atoms
  
  def get_context_mass_per_atom(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the atom masses in the 2D Context."""
    return np.copy(self._context_mass_per_atom)
  
  def get_context_mass_per_dof(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-DOF masses in the simulation Context, shape
      (num_context_atoms, num_dofs_per_atom).
    """
    return np.copy(self._context_mass_per_dof)

  def get_context_num_dofs_per_atom(self) -> int:
    """Get the number of DOFs on each atom in the 2D Context."""
    return self._context_num_dofs_per_atom
  
  def get_integrator_velocities_time_offset(self) -> float:
    """Get the time interval by which the velocities are offset from the coordinates in the 
      TwoDIntegrator.
    """
    return self._context_velocities_time_offset
  
  def get_twod_context(self) -> TwoDContext:
    """Get the 2D Context."""
    return self._twod_context
  
  def get_twod_system(self) -> TwoDSystem:
    """Get the 2D System."""
    return self.get_twod_context().get_twod_system()
  
  def get_twod_integrator(self) -> TwoDIntegrator:
    """Get the 2D Integrator."""
    return self.get_twod_context().get_twod_integrator()

  # ------------------------------------------------------------------------------------------------

  def __init__(self, 
               whoami:              int,
               context_coordinates: np.ndarray, 
               fn_twod_system_init:     Callable[[], TwoDSystem    ] = None, 
               fn_twod_integrator_init: Callable[[], TwoDIntegrator] = None, 
               **kwargs) -> None:
    """Initialize a 2D Replica. 
      
      Args:
        whoami (int):
          The identity index of the Replica.
        context_coordinates (np.ndarray):
          The initial TwoDContext coordinates. 
        fn_twod_system_init (Callable):
          The callable function used to create the TwoDSystem object. Raises an AssertationError if 
          it is unspecified or returns the wrong object. No kwarg is accepted in this function.
        fn_twod_integrator_init (Callable):
          The callable function used to create the TwoDIntegrator object. Raises an AssertationError 
          if it is unspecified or returns the wrong object. No kwarg is accepted in this function.
        Additional kwargs are ignored. 
    """
    # Sanity checks.
    assert not fn_twod_system_init is None,     "Illegal None fn_twod_system_init."
    assert not fn_twod_integrator_init is None, "Illegal None fn_twod_integrator_init."
    assert callable(fn_twod_system_init),       "Illegal non-callable fn_twod_system_init."
    assert callable(fn_twod_integrator_init),   "Illegal non-callable fn_twod_integrator_init."
    
    Replica.__init__(self, 
                     whoami=whoami,
                     context_coordinates =context_coordinates, 
                     weight_per_atom_dict={0: 'none', }, # Must not change for TwoD.
                     fn_twod_system_init    =fn_twod_system_init,
                     fn_twod_integrator_init=fn_twod_integrator_init, )
  
  # Replica interfaces. ----------------------------------------------------------------------------
  # Replica interfaces - Coordinates. 

  def obtain_context_coordinates(self, asarray: bool = False) -> np.ndarray:
    """Obtain the 2D Context coordinates.
    
      Args:
        asarray (bool, optional):
          If cast the returned 2D Context coordinates to np.ndarray.
          Default: False.

      Returns:
        context_coordinates (np.ndarray):
          The 2D Context coordinates. 
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns a np.ndarray, shape (num_context_dofs, )
    """
    if asarray == False:
      return self.get_twod_context().get_coordinates()
    
    return np.expand_dims(self.get_twod_context().get_coordinates(), axis=0)
    
  def update_context_coordinates(self, context_coordinates: np.ndarray) -> None:
    """Update the 2D Context coordinates.
    
      Args:
        context_coordinates (np.ndarray):
          The 2D Context coordinates, shape (num_context_dofs, ).
    """
    self.get_twod_context().set_coordinates(coordinates=context_coordinates)
  
  def cast_to_replica_coordinates(self, context_coordinates: np.ndarray) -> np.ndarray:
    """Cast the 2D Context coordinates to Replica region coordinates.
    
      Args:
        context_coordinates (np.ndarray):
          The 2D Context coordinates, shape (num_context_dofs, ).

      Returns:
        casted_replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    return np.copy(context_coordinates)
  
  def cast_to_context_coordinates(self,
                                  replica_coordinates: np.ndarray, 
                                  context_coordinates: np.ndarray, 
                                  ) -> np.ndarray:
    """Cast the Replica region coordinates to 2D Context coordinates. 

      Args:
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
        context_coordinates (np.ndarray):
          The 2D Context coordinates, shape (num_context_dofs, ).
      
      Returns:
        casted_context_coordinates (np.ndarray):
          The 2D Context coordinates with the Replica region coordinates overridden, shape 
          (num_context_dofs, ).
    """
    return np.copy(replica_coordinates)
  
  # Replica interfaces - Velocities. 

  def initialize_context_velocities(self, temperature: float) -> None:
    """Initialize the 2D Context velocities at designated temperature.
    
      Args:
        temperature (float):
          The temperature at which the 2D Context velocities are initialized, unit: k_{B}T.
    """
    self.get_twod_integrator().initialize_velocities(inverse_beta=temperature)

  def obtain_context_velocities(self, asarray: bool = False) -> np.ndarray:
    """Obtain the 2D Context velocities.
    
      Args:
        asarray (bool, optional):
          If cast the returned 2D Context velocities to np.ndarray.
          Default: False.

      Returns:
        context_velocities (np.ndarray):
          The 2D Context velocities. 
          If asarray==True , returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns a np.ndarray, shape (num_context_dofs, )
    """
    if asarray == False:
      return self.get_twod_context().get_velocities()
    
    return np.expand_dims(self.get_twod_context().get_velocities(), axis=0)

  def update_context_velocities(self, context_velocities: np.ndarray) -> None:
    """Update the 2D Context velocities.
    
      Args:
        context_velocities (np.ndarray):
          The 2D Context velocities, shape (num_context_dofs, ).
    """
    self.get_twod_context().set_velocities(velocities=context_velocities)

  def cast_to_replica_velocities(self, context_velocities: np.ndarray) -> np.ndarray:
    """Cast the 2D Context velocities to Replica region velocities.
    
      Args:
        context_velocities (np.ndarray):
          The 2D Context velocities, shape (num_context_dofs, ).

      Returns:
        casted_replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    return np.copy(context_velocities)

  def cast_to_context_velocities(self, 
                                 replica_velocities: np.ndarray, 
                                 context_velocities: np.ndarray, 
                                 ) -> np.ndarray:
    """Cast the Replica region velocities to 2D Context velocities.

      Args:
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
        context_velocities (np.ndarray):
          The 2D Context velocities, shape (num_context_dofs, ).
      
      Returns:
        casted_context_velocities (np.ndarray):
          The 2D Context velocities with the Replica region velocities overridden.
    """
    return np.copy(replica_velocities)

  # Replica interfaces - Computables. 

  def compute_context_temperature(self) -> float:
    """Get the instantaneous temperature of the Context at its current phase state.
      NOTE: this method returns always the default temperature maintained by the LangevinIntegrator.
    """
    integrator = self.get_twod_integrator()

    if not isinstance(integrator, (TwoDLangevinIntegrator, TwoDScaledLangevinIntegrator)):
      raise RuntimeError("Illegal non-thermostated TwoDIntegrator.")
    
    return integrator._inverse_beta
  
  def compute_context_potential_gradients(self) -> np.ndarray:
    """Get the potential gradients on the 2D Context coordinates. 
    
      Returns:
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom).
    """
    return np.expand_dims(self.get_twod_context().get_potential_gradients(), axis=0)
  
  def compute_context_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradients on the 2D Context coordinates.
    
      Returns:
        context_energy (float):
          The energy.
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom).
    """
    twod_context = self.get_twod_context()
    return (twod_context.get_potential_energy(), 
            np.expand_dims(twod_context.get_potential_gradients(), axis=0), )

  def compute_replica_potential_gradients(self) -> np.ndarray:
    """Get the potential gradient on the Replica region coordinates.

      Returns:
        gradient (np.ndarray):
          The gradient, shape (num_replica_dofs, ).
    """
    return self.get_twod_context().get_potential_gradients()

  def compute_replica_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradient on the Replica region coordinates.
      
      Returns:
        energy (float): 
          The energy;
        gradients (np.ndarray):
          The gradients, shape (num_replica_dofs, ).
    """
    twod_context = self.get_twod_context()
    return twod_context.get_potential_energy(), twod_context.get_potential_gradients()

  # Replica interfaces - MD runtime. 
  
  def append_rmsd_restraint_force(self, 
                                  force_constant:        float, 
                                  force_cutoff_distance: float, 
                                  reference_coordinates: np.ndarray, 
                                  ) -> int:
    """Append an RMSD bias force to the MD Context.
    
      Args:
        force_constant (float):
          The force constant.
        force_cutoff_distance (float):
          The distance beyond which the RMSD bias force is active. 
        reference_coordinates (np.ndarray):
          The reference Replica region coordinates for computing the RMSD distances. 

      Returns:
        force_identity (int):
          The ID used to retrieve the added RMSD bias force from the MD context.
    """
    ref_context_coordinates = self.cast_to_context_coordinates(
                              replica_coordinates=reference_coordinates, 
                              context_coordinates=self.obtain_context_coordinates(asarray=False), )
    twod_force_rmsd = TwoDRmsdCVForce(force_constant=force_constant, 
                                      force_cutoff_distance=force_cutoff_distance, 
                                      reference_coordinates=ref_context_coordinates, )
    return self.get_twod_system().append_force(twod_force=twod_force_rmsd)

  def remove_rmsd_restraint_force(self, force_identity: int) -> None:
    """Remove the RMSD bias force from the MD context using the force_identity.

      Args: 
        force_identity (int):
          The ID used to obtain the to-be-removed RMSD bias force from the MD context.
    """
    self.get_twod_system().remove_force(twod_force_id=force_identity)

  def md_execute_steps(self, num_steps: int) -> None:
    """Execute MD integration for num_steps steps.

      Args:
        num_steps (int):
          The number of MD steps to integrate.
    """
    self.get_twod_integrator().step(num_steps=num_steps)

  # def cast_bonding_to_context_velocities(self, 
  #                                         bonding_velocities: np.ndarray, 
  #                                         context_velocities: np.ndarray, 
  #                                         ) -> np.ndarray:
  #   return self.cast_to_context_velocities(replica_velocities=bonding_velocities, 
  #                                          context_velocities=context_velocities, )
  
  # def cast_context_to_bonding_velocities(self, context_velocities: object) -> np.ndarray:
  #   return self.cast_to_replica_velocities(context_velocities=context_velocities)
