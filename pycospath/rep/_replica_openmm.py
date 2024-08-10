#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""OpenMM Replica."""

from typing import Callable

import numpy as np

from openmm import (State      as OpenMMState, 
                    System     as OpenMMSystem, 
                    Context    as OpenMMContext,
                    Integrator as OpenMMIntegrator, unit)

from pycospath.utils.constants import MOLAR_GAS_CONSTANT

from pycospath.comm.openmm import (OpenMMQuantity, 
                                   openmm_mass_to_akma, 
                                   openmm_energy_to_akma, 
                                   openmm_forces_to_akma,
                                   openmm_velocities_to_akma, 
                                   openmm_coordinates_to_akma, 
                                   openmm_temperature_to_akma,
                                   
                                   openmm_energy_to_np_array,
                                   openmm_forces_to_np_array,
                                   openmm_velocities_to_np_array,
                                   openmm_coordinates_to_np_array, 

                                   akma_velocities_to_openmm, 
                                   akma_coordinates_to_openmm, 

                                   OpenMMRmsdCVForce, 
                                   get_velocities_time_offset, )

from pycospath.rep import Replica

class OpenMMReplica(Replica):
  """The class of Replicas on OpenMM potential."""

  # OpenMM Context initializer. --------------------------------------------------------------------

  def _before__init__(self, 
                      fn_openmm_system_init:     Callable[[], OpenMMSystem    ],
                      fn_openmm_integrator_init: Callable[[], OpenMMIntegrator],
                      openmm_platform_spec:      str  = None,
                      openmm_properties_spec:    dict = None, 
                      ) -> None:
    """To realize the simulation Context. Autoexecuted as the first method in Replica.__init__()."""
    # Build the system from the corresponding init_fn.
    openmm_system: OpenMMSystem = fn_openmm_system_init()
    assert isinstance(openmm_system, OpenMMSystem), \
           "Illegal fn_openmm_system_init return type."
    
    # Build the integrator from the corresponding init_fn
    openmm_integrator = fn_openmm_integrator_init()
    assert isinstance(openmm_integrator, OpenMMIntegrator), \
           "Illegal fn_openmm_integrator_init return type."

    # Build the OpenMM Context object. 
    self._openmm_context = None

    ## With platform and properties spec.
    if not (openmm_platform_spec is None or openmm_properties_spec is None):
      self._openmm_context = OpenMMContext(openmm_system, 
                                           openmm_integrator, 
                                           openmm_platform_spec,
                                           openmm_properties_spec, )
    
    ## With Platform spec only.
    elif (not openmm_platform_spec is None) and openmm_properties_spec is None:
      self._openmm_context = OpenMMContext(openmm_system, 
                                           openmm_integrator, 
                                           openmm_platform_spec, )

    ## No platform spec (thus properties are ignored).
    else:
      self._openmm_context = OpenMMContext(openmm_system, 
                                           openmm_integrator, )

    # RepPotOpenMM specific private members.
    context_num_atoms, context_mass_per_atom = openmm_system.getNumParticles(), []

    for i in range(context_num_atoms):
      m = openmm_system.getParticleMass(i)
      context_mass_per_atom.append(openmm_mass_to_akma(openmm_mass=m))

    self._context_mass_per_atom = np.asarray(context_mass_per_atom)
    self._context_mass_per_dof = np.repeat(self.get_context_mass_per_atom()[:, np.newaxis], 
                                           repeats=3, 
                                           axis=1, )
    self._context_num_atoms = context_num_atoms
    self._context_num_dofs_per_atom = 3
    self._context_velocities_time_offset = get_velocities_time_offset(integrator=openmm_integrator)

  def get_context_num_atoms(self) -> int:
    """Get the number of atoms in the OpenMM Context."""
    return self._context_num_atoms

  def get_context_mass_per_atom(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the atom masses in the OpenMM Context."""
    return np.copy(self._context_mass_per_atom)

  def get_context_mass_per_dof(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-DOF masses in the simulation Context."""
    return np.copy(self._context_mass_per_dof)

  def get_context_num_dofs_per_atom(self) -> int:
    """Get the number of DOFs on each atom in the OpenMM Context."""
    return self._context_num_dofs_per_atom
  
  def get_integrator_velocities_time_offset(self) -> float:
    """Get the time interval by which the velocities are offset from the coordinates in the 
      OpenMMIntegrator.
    """
    return self._context_velocities_time_offset

  def get_openmm_context(self) -> OpenMMContext:
    """Get the OpenMM Context object."""
    return self._openmm_context
  
  def get_openmm_system(self) -> OpenMMSystem:
    """Get the OpenMM System object."""
    return self.get_openmm_context().getSystem()
  
  def get_openmm_integrator(self) -> OpenMMIntegrator:
    """Get the OpenMM Integrator object."""
    return self.get_openmm_context().getIntegrator()

  # ------------------------------------------------------------------------------------------------

  def __init__(self,
               whoami:               int,
               context_coordinates:  OpenMMQuantity,
               weight_per_atom_dict: dict[int, float | str] = None, 
               fn_openmm_system_init:     Callable[[], OpenMMSystem    ] = None,
               fn_openmm_integrator_init: Callable[[], OpenMMIntegrator] = None,
               openmm_platform_spec:      str  = None,
               openmm_properties_spec:    dict = None,
               ) -> None:
    """Initialize a OpenMM Replica.

      Args:
        whoami (int):
          The identity index of the Replica.
        context_coordinates (OpenMMQuantity):
          The initial Context coordinates. 
        weight_per_atom_dict (dict[int, float | str], optional):
          The Python dict for defining the Replica region. Its keys specify the (zero-indexed) 
          indices of the atoms included in the Replica region while its values specify the weights 
          on this atom for computing the weighted inter-Replica RMS distances. For Minimum Energy 
          Path (MEP) tasks, atoms not present in this dict will not be optimized (fixed in place). 
          To include an atom during the optimizations but exclude from the RMS distance computations 
          (e.g., for MEP using heavy-atom weighted RMS, we would exclude the Hydrogen atoms from the 
          Replica region but still optimize their positions rather than fix them in place), include 
          the atom indices as the Key in this dict but set the Value to zero or 'excl'. 
          Some examples: 
          ```
            atom_weights = {
              0: 'mass', # The _1st_ atom is scaled by its atomic unit mass;
              2: 'none', # The _3rd_ atom is not scaled (or scaled by 1.);
              4: 'excl', # The _5th_ is ignored by weight scaling (or scaled by 0.); 
              8: 10.   , # The _9th_ atom is scaled by 10.
            }
          ```
          Default: None, all atoms in the system will be used with unit weights.
        fn_openmm_system_init (Callable):
          The callable function used to create the OpenMM System object. Raises an AssertationError 
          if is unspecified or returns the wrong object. No kwarg is accepted in this function.
        fn_openmm_integrator_init (Callable):
          The callable function used to create the OpenMM Integrator object. Raises an 
          AssertationError if is unspecified or returns the wrong object. No kwarg is accepted in 
          this function.
        openmm_platform_spec (openmm.Platform, optional):
          The OpenMM Platform object used for the calculation. No sanity check is performed on this
          kwarg. This kwarg and 'openmm_plotformprop_spec' initializes the OpenMM Context object. 
          Default: None.
        openmm_properties_spec (dict, optional):
          The set of values used to specify OpenMM Platform specific properties. No sanity check is 
          performed on this kwarg. This kwarg and 'openmm_platform_spec' initialize the OpenMM 
          Context object. 
          Default: None. 
    """
    # Sanity checks.
    assert not fn_openmm_system_init is None,     "Illegal None fn_openmm_system_init."
    assert not fn_openmm_integrator_init is None, "Illegal None fn_openmm_integrator_init."
    assert callable(fn_openmm_system_init),       "Illegal non-callable fn_openmm_system_init."
    assert callable(fn_openmm_integrator_init),   "Illegal non-callable fn_openmm_integrator_init."

    # Initialize the OpenMMReplica.
    Replica.__init__(self,
                     whoami=whoami,
                     context_coordinates =context_coordinates,
                     weight_per_atom_dict=weight_per_atom_dict, 
                     fn_openmm_system_init    =fn_openmm_system_init, 
                     fn_openmm_integrator_init=fn_openmm_integrator_init, 
                     openmm_platform_spec     =openmm_platform_spec, 
                     openmm_properties_spec   =openmm_properties_spec, )
  
  # Replica interfaces. ----------------------------------------------------------------------------
  # Replica interfaces - Coordinates. 
  
  def obtain_context_coordinates(self, asarray: bool = False) -> np.ndarray | OpenMMQuantity:
    """Obtain the OpenMM Context coordinates.
    
      Args:
        asarray (bool, optional):
          If cast the returned OpenMM Context coordinates to np.ndarray.
          Default: False.

      Returns:
        context_coordinates (np.ndarray | OpenMMQuantity):
          The OpenMM Context coordinates. 
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom), 
                             unit: nm; 
          If asarray==False, returns an OpenMMQuantity.
    """
    openmm_state: OpenMMState = self.get_openmm_context().getState(getPositions=True)
    openmm_coordinates = openmm_state.getPositions(asNumpy=True)

    if asarray == False:
      return openmm_coordinates

    context_coordinates = openmm_coordinates_to_np_array(openmm_coordinates=openmm_coordinates)
    context_coordinates = context_coordinates.reshape(self.get_context_num_atoms(), 
                                                      self.get_context_num_dofs_per_atom(), )
    return context_coordinates
  
  def update_context_coordinates(self, context_coordinates: OpenMMQuantity) -> None:
    """Update the OpenMM Context coordinates.
    
      Args:
        context_coordinates (OpenMMQuantity):
          The OpenMM Context coordinates.
    """
    assert isinstance(context_coordinates, OpenMMQuantity), "Illegal context_coordinates type."
    self.get_openmm_context().setPositions(positions=context_coordinates)

  def cast_to_replica_coordinates(self, context_coordinates: OpenMMQuantity) -> np.ndarray:
    """Cast the OpenMM Context coordinates to Replica region coordinates.

      Args:
        context_coordinates (OpenMMQuantity):
          The OpenMM Context coordinates.

      Returns:
        casted_replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    tmp_coordinates = openmm_coordinates_to_akma(openmm_coordinates=context_coordinates)
    return tmp_coordinates[self.get_replica_atom_indices(), :].flatten()
  
  def cast_to_context_coordinates(self,
                                  replica_coordinates: np.ndarray, 
                                  context_coordinates: OpenMMQuantity, 
                                  ) -> OpenMMQuantity:
    """Cast the Replica region coordinates to OpenMM Context coordinates. 

      Args:
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
        context_coordinates (OpenMMQuantity):
          The OpenMM Context coordinates.
      
      Returns:
        casted_context_coordinates (OpenMMQuantity):
          The OpenMM Context coordinates with the Replica region coordinates overridden.
    """
    # Cast OpenMM Context coordinates to AKMA np.ndarray.
    tmp_coordinates = openmm_coordinates_to_akma(openmm_coordinates=context_coordinates)
    replica_coordinates_vec3 = replica_coordinates.reshape(self.get_replica_num_atoms(), 
                                                           self.get_context_num_dofs_per_atom(), )
    # Overwrite the Replica region coordinates.
    tmp_coordinates[self.get_replica_atom_indices(), :] = replica_coordinates_vec3
    # Rebuild the OpenMM Context coordinates and return.
    return akma_coordinates_to_openmm(akma_coordinates=tmp_coordinates)

  # Replica interfaces - Velocities. 

  def initialize_context_velocities(self, temperature: float) -> None:
    """Initialize the OpenMM Context velocities at designated temperature.
    
      Args:
        temperature (float):
          The temperature at which the OpenMM Context velocities are initialized, unit: kelvin.
    """
    self.get_openmm_context().setVelocitiesToTemperature(temperature)

  def obtain_context_velocities(self, asarray: bool = False) -> np.ndarray | OpenMMQuantity:
    """Obtain the OpenMM Context velocities.
    
      Args:
        asarray (bool, optional):
          If cast the returned OpenMM Context velocities to np.ndarray.
          Default: False.

      Returns:
        context_velocities (np.ndarray | OpenMMQuantity):
          The OpenMM Context velocities. 
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom), 
                             unit: nm/ps; 
          If asarray==False, returns an OpenMMQuantity.
    """
    openmm_state: OpenMMState = self.get_openmm_context().getState(getVelocities=True)
    openmm_velocities = openmm_state.getVelocities(asNumpy=True)

    if asarray == False:
      return openmm_velocities

    context_velocities = openmm_velocities_to_np_array(openmm_velocities=openmm_velocities)
    context_velocities = context_velocities.reshape(self.get_context_num_atoms(), 
                                                    self.get_context_num_dofs_per_atom(), )
    return context_velocities

  def update_context_velocities(self, context_velocities: OpenMMQuantity) -> None:
    """Update the OpenMM Context velocities.
    
      Args:
        context_velocities (OpenMMQuantity):
          The OpenMM Context velocities.
    """
    assert isinstance(context_velocities, OpenMMQuantity), "Illegal context_velocities type."
    self.get_openmm_context().setVelocities(velocities=context_velocities)

  def cast_to_replica_velocities(self, context_velocities: OpenMMQuantity) -> np.ndarray:
    """Cast the OpenMM Context velocities to Replica region velocities.
    
      Args:
        context_velocities (OpenMMQuantity):
          The OpenMM Context velocities.

      Returns:
        casted_replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    tmp_velocities = openmm_velocities_to_akma(openmm_velocities=context_velocities)
    return tmp_velocities[self.get_replica_atom_indices(), :].flatten()

  def cast_to_context_velocities(self, 
                                 replica_velocities: np.ndarray, 
                                 context_velocities: OpenMMQuantity, 
                                 ) -> OpenMMQuantity:
    """Cast the Replica region velocities to OpenMM Context velocities.

      Args:
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
        context_velocities (OpenMMQuantity):
          The 2D Context velocities.
      
      Returns:
        casted_context_velocities (OpenMMQuantity):
          The 2D Context velocities with the Replica region velocities overridden.
    """
    # Cast OpenMM Context velocities to AKMA np.ndarray.
    tmp_velocities = openmm_velocities_to_akma(openmm_velocities=context_velocities)
    replica_velocities_vec3 = replica_velocities.reshape(self.get_replica_num_atoms(), 
                                                         self.get_context_num_dofs_per_atom(), )
    # Override the Replica region velocities.
    tmp_velocities[self.get_replica_atom_indices(), :] = replica_velocities_vec3
    # Rebuild the OpenMM Context velocities and return.
    return akma_velocities_to_openmm(akma_velocities=tmp_velocities)
  
  # def cast_context_to_bonding_velocities(self, context_velocities: OpenMMQuantity) -> np.ndarray:
  #   tmp_velocities = openmm_velocities_to_akma(openmm_velocities=context_velocities)
  #   return tmp_velocities[self.get_bonding_atom_indices(), :].flatten()

  # def cast_bonding_to_context_velocities(self, 
  #                                        bonding_velocities: np.ndarray, 
  #                                        context_velocities: OpenMMQuantity, 
  #                                        ) -> OpenMMQuantity:
  #   tmp_velocities = openmm_velocities_to_akma(openmm_velocities=context_velocities)
  #   bonding_velocities_vec3 = bonding_velocities.reshape(-1, 3)
  #   tmp_velocities[self.get_bonding_atom_indices(), :] = bonding_velocities_vec3
  #   return akma_velocities_to_openmm(akma_velocities=tmp_velocities)

  # Replica interfaces - Computables. 

  def compute_context_temperature(self) -> float:
    """Get the instantaneous temperature of the Context at its current phase state."""
    # https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/statedatareporter.py
    # If thermostated integrator.
    integrator = self.get_openmm_integrator()
    if hasattr(integrator, 'computeSystemTemperature'):
      return openmm_temperature_to_akma(openmm_temperature=integrator.computeSystemTemperature())
    
    # If non-thermostated integrator.
    state: OpenMMState = self.get_openmm_context().getState(getEnergy=True)
    kinetic_energy = openmm_energy_to_akma(openmm_energy=state.getKineticEnergy()) # Full step v.
    num_context_dofs = self.get_context_num_atoms() * self.get_context_num_dofs_per_atom()
    return 2. * kinetic_energy / (num_context_dofs * MOLAR_GAS_CONSTANT)

  def compute_context_potential_gradients(self) -> np.ndarray:
    """Get the potential gradients on the OpenMM Context coordinates. 
    
      Returns:
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom), unit: kJ/mol/nm.
    """
    openmm_state: OpenMMState = self.get_openmm_context().getState(getForces=True)
    # grad = -force.
    context_gradients = openmm_state.getForces(asNumpy=True)
    context_gradients = -1.*openmm_forces_to_np_array(openmm_forces=context_gradients)
    context_gradients = context_gradients.reshape(self.get_context_num_atoms(), 
                                                  self.get_context_num_dofs_per_atom(), )

    return context_gradients 
  
  def compute_context_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradients on the OpenMM Context coordinates.
    
      Returns:
        context_energy (float):
          The energy, unit: kJ/mol.
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom), unit: kJ/mol/nm.
    """
    openmm_state: OpenMMState = self.get_openmm_context().getState(getEnergy=True, getForces=True)
    context_energy = openmm_energy_to_np_array(openmm_energy=openmm_state.getPotentialEnergy())
    # grad = -force.
    context_gradients = openmm_state.getForces(asNumpy=True)
    context_gradients = -1.*openmm_forces_to_np_array(openmm_forces=context_gradients)
    context_gradients = context_gradients.reshape(self.get_context_num_atoms(), 
                                                  self.get_context_num_dofs_per_atom(), )
    return context_energy, context_gradients

  def compute_replica_potential_gradients(self) -> np.ndarray:
    """Get the potential gradient on the Replica region coordinates.

      Returns:
        gradients (np.ndarray):
          The gradient, shape (num_replica_dofs, ), unit: kcal/mol/Angstrom.
    """
    # Note that once an array is indexed by another array, the returned array is a copy instead of a
    # memory reference. If indexed by a set of numbers, then the returned array will share the same 
    # memory allocation. Either way it does not cause problems, since grad_all is temporary in this 
    # def. context_gradient is a shape (num_replica_atoms, 3) array.
    openmm_state: OpenMMState = self.get_openmm_context().getState(getForces=True)
    context_gradients = openmm_state.getForces(asNumpy=True)
    context_gradients = -1.*openmm_forces_to_akma(openmm_forces=context_gradients)
    context_gradients = context_gradients[self.get_replica_atom_indices(), :].flatten()
    return context_gradients

  def compute_replica_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradient on the Replica region coordinates.
      
      Returns:
        energy (float): 
          The energy, unit: kcal/mol;
        gradients (np.ndarray):
          The gradient, shape (num_replica_dofs, ), unit: kcal/mol/Angstrom.
    """
    # Note that once an array is indexed by another array, the returned array is a copy instead of a
    # memory reference. If indexed by a set of numbers, then the returned array will share the same 
    # memory allocation. Either way it does not cause problems, since grad_all is temporary in this 
    # def. context_forces is a shape (num_replica_atoms, 3) array.
    openmm_state: OpenMMState = self.get_openmm_context().getState(getEnergy=True, getForces=True)
    context_energy = openmm_energy_to_akma(openmm_energy=openmm_state.getPotentialEnergy())
    # grad = -force.
    context_gradients = openmm_state.getForces(asNumpy=True)
    context_gradients = -1.*openmm_forces_to_akma(openmm_forces=context_gradients)
    context_gradients = context_gradients[self.get_replica_atom_indices(), :].flatten()
    return context_energy, context_gradients
  
  # Replica interfaces - MD runtime. 
  
  def append_rmsd_restraint_force(self, 
                                  force_constant:        float, 
                                  force_cutoff_distance: float, 
                                  reference_coordinates: np.ndarray, 
                                  ) -> int:
    """Append a RMSD bias force to the MD Context.
    
      Args:
        force_constant (float):
          The force constant, unit: kcal/mol/Angstrom**2.
        force_cutoff_distance (float):
          The distance beyond which the RMSD bias force is active, unit: Angstrom. 
        reference_coordinates (np.ndarray):
          The Replica region coordinates to be used as the reference to compute RMSD. 

      Returns:
        force_identity (int):
          The ID used to obtain the added RMSD bias force from the MD context.
    """
    ref_context_coordinates = self.cast_to_context_coordinates(
                              replica_coordinates=reference_coordinates, 
                              context_coordinates=self.obtain_context_coordinates(asarray=False), )
    openmm_force_rmsd = OpenMMRmsdCVForce(force_constant=force_constant, 
                                          force_cutoff_distance=force_cutoff_distance, 
                                          reference_coordinates=ref_context_coordinates, 
                                          replica_atom_indices=self.get_replica_atom_indices(), )
    force_identity = self.get_openmm_system().addForce(force=openmm_force_rmsd)
    
    # Refresh the Context cache.
    self.get_openmm_context().reinitialize(preserveState=True)

    return force_identity
  
  def remove_rmsd_restraint_force(self, force_identity: int) -> None:
    """Remove the RMSD bias force identified by `force_identity` from the MD context.

      Args: 
        force_identity (int):
          The ID used to obtain the RMSD bias force to be removed.
    """
    self.get_openmm_system().removeForce(index=force_identity)

    # Refresh the Context cache.
    self.get_openmm_context().reinitialize(preserveState=True)
    
  def md_execute_steps(self, num_steps: int) -> None:
    """Execute MD integration for num_steps steps.

      Args:
        num_steps (int):
          The number of MD steps to integrate.
    """
    self.get_openmm_integrator().step(steps=num_steps)


