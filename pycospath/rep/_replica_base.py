#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of Replicas."""

from abc import ABC, abstractmethod

import numpy as np

class Replica(ABC):
  """The abstract class of Replica."""

  KW_WEIGHT_PER_ATOM = ['mass', 'none', 'excl', ]

  # Context initializer. ---------------------------------------------------------------------------
  
  @abstractmethod
  def _before__init__(self, **kwargs) -> None:
    """To realize the simulation Context. Autoexecuted as the first method in Replica.__init__()."""

  @abstractmethod
  def get_context_num_atoms(self) -> int:
    """Get the number of atoms in the simulation Context."""

  @abstractmethod
  def get_context_mass_per_atom(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-atom masses in the simulation Context."""
  
  @abstractmethod
  def get_context_mass_per_dof(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-DOF masses in the simulation Context."""

  @abstractmethod
  def get_context_num_dofs_per_atom(self) -> int:
    """Get the number of DOFs on each atom in the simulation Context."""
  
  @abstractmethod
  def get_integrator_velocities_time_offset(self) -> float:
    """Get the time interval by which the velocities are offset from the coordinates."""
  
  # ------------------------------------------------------------------------------------------------

  def __init__(self,
               whoami:               int,
               context_coordinates:  object,
               weight_per_atom_dict: dict[int, float | str] = None, 
               **kwargs) -> None: 
    """Initialize a Replica instance.
      
      Args: 
        whoami (int):
          The identity index of the Replica.
        context_coordinates (object):
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
        Additional kwargs are passed to Replica._before__init__().
    """
    # Initialize simulation Context. 
    self._before__init__(**kwargs)

    # Make identity index. 
    assert isinstance(whoami, (int, np.int_)), "Illegal whoami type."
    self._whoami = int(whoami)

    # Make Replica atom selection. 
    num_context_atoms = self.get_context_num_atoms()

    # atom_weights is None: all atoms are included in Replica with unit weights.
    if weight_per_atom_dict is None:
      weight_per_atom_dict = dict(zip(list( np.arange(  num_context_atoms    ) ), 
                                 list( np.ones  ( (num_context_atoms, ) ) ), ) )
    
    assert isinstance(weight_per_atom_dict, dict), "Illegal weight_per_atom type."
    assert len(weight_per_atom_dict) > 0,          "Illegal weight_per_atom spec."

    num_replica_atoms = 0
    replica_atom_selection = np.zeros((num_context_atoms, )) # all unselected.

    for k in weight_per_atom_dict.keys():           # Accounting with no duplication.
      assert isinstance(k, (int, np.int_)),  "Illegal weight_per_atom.keys type."
      assert num_context_atoms > k and k >= 0, "Illegal weight_per_atom.keys spec."

      num_replica_atoms += 1
      replica_atom_selection[ int(k) ] = 1

    self._replica_num_atoms = num_replica_atoms
    self._replica_atom_onehots = np.asarray(replica_atom_selection, dtype=int)

    # Make Replica atom weights and atom indices in context. 
    replica_atom_indices = []
    replica_weight_per_atom = np.zeros( (num_replica_atoms, ) )

    i_key = 0
    context_mass_per_atom = self.get_context_mass_per_atom()

    for k in weight_per_atom_dict.keys():
      replica_atom_indices.append( int(k) )
      
      # Parse atom_weights entries by type.
      v = weight_per_atom_dict.get(k)

      if isinstance(v, str):
        v = v.lower()

        assert v in self.KW_WEIGHT_PER_ATOM, "Illegal weight_per_atom.values spec."
        replica_weight_per_atom[ i_key ] = 0. if v in ['excl'] else        \
                                           1. if v in ['none'] else        \
                                           context_mass_per_atom[k] # if v in ['mass'].
      
      elif isinstance(v, (int, float)):
        assert v >= 0., "Illegal weight_per_atom.values spec."
        replica_weight_per_atom[ i_key ] = float(v)

      else:
        raise TypeError("Illegal weight_per_atom.values type.")
      
      i_key += 1

    self._replica_atom_indices = np.asarray(replica_atom_indices, dtype=int)
    self._replica_weight_per_atom = replica_weight_per_atom
    
    # Make Replica DOF weights. 
    self._replica_weight_per_dof = np.repeat(self._replica_weight_per_atom, 
                                             self.get_context_num_dofs_per_atom(),
                                             axis=0, )
    
    # Make Replica atom/DOF masses. 
    self._replica_mass_per_atom: np.ndarray = np.take(context_mass_per_atom, 
                                                      self._replica_atom_indices, )
    self._replica_mass_per_dof = np.repeat(self._replica_mass_per_atom, 
                                           self.get_context_num_dofs_per_atom(), 
                                           axis=0, )
    
    # Initialize atomic coordinates in Context. 
    self.update_context_coordinates(context_coordinates=context_coordinates)
    
    # self._bonding_atom_indices = None

  # Replica interfaces. ----------------------------------------------------------------------------
  # Replica interfaces - General info. 

  def get_whoami(self) -> int:
    """Get the identity index of the Replica."""
    return self._whoami
  
  def get_replica_atom_indices(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the indices of the Replica region atoms, shape 
      (num_replica_atoms, ).
    """
    return np.copy(self._replica_atom_indices)
  
  def get_replica_atom_onehots(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the flags that distinguishes Replica region atoms (1) 
      or other atoms (0), shape (num_context_atoms, ).
    """
    return np.copy(self._replica_atom_onehots)
  
  def get_replica_num_atoms(self) -> int:
    """Get the number of atoms in the Replica region."""
    return self._replica_num_atoms
  
  def get_replica_weight_per_atom(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-atom weighting factors on the Replica region
      atoms, shape (num_replica_atoms, ).
    """
    return np.copy(self._replica_weight_per_atom)
    
  def get_replica_mass_per_atom(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-atom masses on the Replica region atoms, 
      shape (num_replica_atoms, ).
    """
    return np.copy(self._replica_mass_per_atom)

  def get_replica_num_dofs(self) -> int:
    """Get the number of DOFs on the Replica region atoms."""
    return self.get_replica_num_atoms() * self.get_context_num_dofs_per_atom()
  
  def get_replica_weight_per_dof(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-DOF weighting factors on the Replica region
      atoms, shape (num_replica_dofs, ).
    """
    return np.copy(self._replica_weight_per_dof)

  def get_replica_mass_per_dof(self) -> np.ndarray:
    """Get a copy of the np.ndarray that holds the per-DOF masses on the Replica region atoms, shape
      (num_replica_dofs, ).
    """
    return np.copy(self._replica_mass_per_dof)
  
  # Replica interfaces - Coordinates. 

  ## TODO: change of APIs:
  ##       obtain_coordinates()
  ##       obtain_coordinates_as_replica()
  ##       obtain_coordinates_as_asnumpy()
  ##       update_coordinates(coordinates: object)
  ##       update_coordinates_as_replica(replica_coordinates: np.ndarray)
  ##       update_coordinates_as_asnumpy(asnumpy_coordinates: np.ndarray)
  ##       coordinates_to_replica(coordinates: object)
  ##       coordinates_to_asnumpy(coordinates: object)
  ## Remove all 'context' namings.
  
  @abstractmethod
  def obtain_context_coordinates(self, asarray: bool = False) -> object | np.ndarray:
    """Obtain the simulation Context coordinates.
    
      Args:
        asarray (bool, optional):
          If cast the returned simulation Context coordinates to np.ndarray.
          Default: False.

      Returns:
        context_coordinates (object | np.ndarray):
          The simulation Context coordinates. 
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns the communicator-specific data structure.
    """

  @abstractmethod
  def update_context_coordinates(self, context_coordinates: object) -> None:
    """Update the simulation Context coordinates.
    
      Args:
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.
    """

  @abstractmethod
  def cast_to_replica_coordinates(self, context_coordinates: object) -> np.ndarray:
    """Cast the simulation Context coordinates to Replica region coordinates.
    
      Args:
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.

      Returns:
        casted_replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
  
  @abstractmethod
  def cast_to_context_coordinates(self,
                                  replica_coordinates: np.ndarray, 
                                  context_coordinates: object, 
                                  ) -> object:
    """Cast the Replica region coordinates to simulation Context coordinates. 

      Args:
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.
      
      Returns:
        casted_context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure with the 
          Replica region coordinates overridden.
    """

  def obtain_replica_coordinates(self) -> np.ndarray:
    """Obtain the Replica region coordinates.

      Returns: 
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    context_coordinates = self.obtain_context_coordinates(asarray=False)
    return self.cast_to_replica_coordinates(context_coordinates=context_coordinates)

  def update_replica_coordinates(self, replica_coordinates: np.ndarray) -> None:
    """Update the Replica region coordinates.

      Args:
        replica_coordinates: 
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    context_coordinates = self.obtain_context_coordinates(asarray=False)
    context_coordinates = self.cast_to_context_coordinates(replica_coordinates=replica_coordinates, 
                                                           context_coordinates=context_coordinates,)
    self.update_context_coordinates(context_coordinates=context_coordinates)

  # Replica interfaces - Velocities. 

  @abstractmethod
  def initialize_context_velocities(self, temperature: float) -> None:
    """Initialize the simulation Context velocities at designated temperature.
    
      Args:
        temperature (float):
          The temperature at which the simulation Context velocities are initialized, unit: kelvin.
    """

  @abstractmethod
  def obtain_context_velocities(self, asarray: bool = False) -> object | np.ndarray:
    """Obtain the simulation Context velocities.
    
      Args:
        asarray (bool, optional):
          If cast the returned simulation Context velocities to np.ndarray.
          Default: False.

      Returns:
        context_velocities (object | np.ndarray):
          The simulation Context velocities.
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns the communicator-specific data structure.
    """

  @abstractmethod
  def update_context_velocities(self, context_velocities: object) -> None:
    """Update the simulation Context velocities.
    
      Args:
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.
    """

  def cast_to_fullstep_velocities(self, 
                                  velocities:          np.ndarray, 
                                  mass_per_dof:        np.ndarray, 
                                  potential_gradients: np.ndarray, 
                                  ) -> np.ndarray:
    """Cast the velocities to full timestep by time offseting.
    
      Args:
        velocities (np.ndarray):
          The velocities to be offset in time.
        mass_per_dof (np.ndarray):
          The per-DOF masses. 
        potential_gradients (np.ndarray):
          The potential gradients. 
      
      Returns:
        velocities_offset (np.ndarray):
          The velocities offset to full timestep. 
    """
    velocities = np.copy(velocities)
    
    if (offset:=self.get_integrator_velocities_time_offset()) != 0.:
      # V(dt) = V(.5dt) - .5dt * \frac{\partial U}{\partial x(dt)} / m
      velocities -= offset*potential_gradients*mass_per_dof**-1

    return velocities

  @abstractmethod
  def cast_to_replica_velocities(self, context_velocities: object) -> np.ndarray:
    """Cast the simulation Context velocities to Replica region velocities.
    
      Args:
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.

      Returns:
        casted_replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
  
  @abstractmethod
  def cast_to_context_velocities(self, 
                                 replica_velocities: np.ndarray, 
                                 context_velocities: np.ndarray, 
                                 ) -> object:
    """Cast the Replica region velocities to simulation Context velocities.

      Args:
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.
      
      Returns:
        casted_context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure with the 
          Replica region velocities overridden.
    """

  def obtain_replica_velocities(self) -> np.ndarray:
    """Obtain the Replica region velocities.

      Returns: 
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    context_velocities = self.obtain_context_velocities(asarray=False)
    return self.cast_to_replica_velocities(context_velocities=context_velocities)

  def update_replica_velocities(self, replica_velocities: np.ndarray) -> None:
    """Update the Replica region velocities.

      Args:
        replica_velocities: 
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    context_velocities = self.obtain_context_velocities(asarray=False)
    context_velocities = self.cast_to_context_velocities(replica_velocities=replica_velocities, 
                                                         context_velocities=context_velocities, )
    self.update_context_velocities(context_velocities=context_velocities)
  
  # Replica interfaces - Computables. 
  
  @abstractmethod
  def compute_context_temperature(self) -> float:
    """Get the instantaneous temperature of the simulation Context at its current phase state."""

  @abstractmethod
  def compute_context_potential_gradients(self) -> np.ndarray:
    """Get the potential gradients on the simulation Context coordinates. 
    
      Returns:
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom).
    """

  @abstractmethod
  def compute_context_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradients on the simulation Context coordinates.
    
      Returns:
        context_energy (float):
          The energy, unit: kcal/mol.
        context_gradients (np.ndarray):
          The gradients, shape (num_context_atoms, num_dofs_per_atom).
    """

  @abstractmethod
  def compute_replica_potential_gradients(self) -> np.ndarray:
    """Get the potential gradients on the Replica region coordinates.

      Returns:
        replica_gradients (np.ndarray):
          The gradient, shape (num_replica_dofs, ), unit: kcal/mol/Angstrom.
    """

  @abstractmethod
  def compute_replica_potential_energy_and_gradients(self) -> tuple[float, np.ndarray]:
    """Get the potential energy and gradients on the Replica region coordinates.
      
      Returns:
        replica_energy (float): 
          The energy, unit: kcal/mol;
        replica_gradients (np.ndarray):
          The gradient, shape (num_replica_dofs, ), unit: kcal/mol/Angstrom.
    """

  # Replica interfaces - Append / Remove forces.

  @abstractmethod
  def append_rmsd_restraint_force(self, 
                                  force_constant:        float, 
                                  force_cutoff_distance: float, 
                                  reference_coordinates: np.ndarray, 
                                  ) -> object:
    """Append an RMSD bias force to the MD Context.
    
      Args:
        force_constant (float):
          The force constant, unit: kcal/mol/Angstrom**2.
        force_cutoff_distance (float):
          The distance beyond which the RMSD bias force is active, unit: Angstrom. 
        reference_coordinates (np.ndarray):
          The reference Replica region coordinates for computing the RMSD distances, unit: Angstrom. 

      Returns:
        force_identity (object):
          The ID used to obtain the appended RMSD bias force from the MD context.
    """

  @abstractmethod
  def remove_rmsd_restraint_force(self, force_identity: object) -> None:
    """Remove the RMSD bias force from the MD context using the force_identity.

      Args: 
        force_identity (object):
          The ID used to obtain the to-be-removed RMSD bias force from the MD context.
    """

  # Replica interfaces - MD runtime. 

  @abstractmethod
  def md_execute_steps(self, num_steps: int) -> None:
    """Execute MD integration for num_steps steps.

      Args:
        num_steps (int):
          The number of MD steps to integrate.
    """
  

  # def set_bonding_atom_indices(self, indices) -> None:
  #   self._bonding_atom_indices = np.copy(indices)

  # def get_bonding_atom_indices(self) -> np.ndarray:
  #   if self._bonding_atom_indices is None:
  #     return np.copy(self._replica_atom_indices)
  #   return np.copy(self._bonding_atom_indices)
  
  # @abstractmethod
  # def cast_bonding_to_context_velocities(self, 
  #                                        bonding_velocities: np.ndarray, 
  #                                        context_velocities: object, ) -> object: ...
  
  # @abstractmethod
  # def cast_context_to_bonding_velocities(self, context_velocities: object) -> np.ndarray: ...


