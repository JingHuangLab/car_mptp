#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The Serial Minimum Energy Path (MEP) calculation."""

from typing import Type

import numpy as np

from pycospath.cos import CoS

from pycospath.rep import Replica

from pycospath.app import Path

from pycospath.app.mep import PathOptimizer

class MEP(Path):
  """The Serial Minimum Energy Path (MEP) calculation."""

  def __init__(self, 
               context_coordinates_list: list, 
               replica_class:  Type[Replica], 
               replica_kwargs: dict, 
               cos_class:  Type[CoS], 
               cos_kwargs: dict, 
               optimizer_class:  Type[PathOptimizer], 
               optimizer_kwargs: dict, 
               method_alignment:    str = 'kabsch', 
               method_path_tangent: str = 'context', 
               ) -> None:
    """Initialize a Serial Minimum Energy Path (MEP) calculation.
    
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
        optimizer_class (Type[Optimizer]):
          The class (not instances) of the Path Optimizer;
        optimizer_kwargs (dict):
          Additional keywords passed to the constructor of the Path Optimizer;
        method_alignment (str, optional):
          The method used for molecular alignment for the Chain-of-Replicas. 
          Default: 'kabsch'. 
        method_path_tangent (str, optional):
          The method used for Path tangent vector evaluation for the CoS. This option does not 
          impact the String Method as it uses its reparametrized Path function method for tangent 
          estimation.
          Default: 'context'. 
    """
    Path.__init__(self, 
                  context_coordinates_list=context_coordinates_list, 
                  replica_class=replica_class, 
                  replica_kwargs=replica_kwargs, 
                  method_alignment=method_alignment, 
                  method_path_tangent=method_path_tangent, )
    
    # Init. CoS and realize CoS prompt methods.. 
    assert issubclass(cos_class,  CoS ), "Illegal cos_class type."
    assert isinstance(cos_kwargs, dict), "Illegal cos_kwargs type."
    
    self._cos = cos_class(**cos_kwargs)
    self._cos.implement(fn_get_path_tangent     =self.get_path_tangent, 
                        fn_get_path_weighted_rms=self.get_path_weighted_rms, )

    # Init. PathOptimizer. 
    assert issubclass(optimizer_class,  PathOptimizer), "Illegal optimizer_class type."
    assert isinstance(optimizer_kwargs, dict         ), "Illegal optimizer_kwargs type."

    self._optimizer = optimizer_class(**optimizer_kwargs)
  
  def get_cos(self) -> CoS:
    """Get the CoS instance hosted by this MEP calculation."""
    return self._cos
  
  def get_optimizer(self) -> PathOptimizer:
    """Get the Path Optimizer instance hosted by this MEP calculation."""
    return self._optimizer

  def compute_path_potential_energies_and_gradients(self) -> tuple[np.ndarray, np.ndarray]:
    """Get the Path potential energies and gradients on the current Path colvar.
    
      Returns:
        path_energies (np.ndarray):
          path_energies:  The Path potential energies, shape (num_replicas, ).
        path_gradients (np.ndarray):
          path_gradients: The Path potential gradients, shape (num_replicas, num_replica_dofs).
    """
    replica_list = self.get_replica_list()
    path_energies  = np.zeros((self.get_num_path_replicas(), ))
    path_gradients = np.zeros((self.get_num_path_replicas(), self.get_num_path_dofs()))

    for i_replica in range(self.get_num_path_replicas()):
      # The energy and the unrotated gradient on Replica coordinates.
      replica = replica_list[i_replica]
      ener, grad = replica.compute_replica_potential_energy_and_gradients()

      # Potential gradients on the instantaneous Replica region coordinates are rotated onto the row 
      # of the path_colvar.
      grad = self.get_weighted_rotated(array_to_refer=self._path_colvar[i_replica, :], 
                                       array_to_align=replica.obtain_replica_coordinates(), 
                                       array_to_rotate=grad, )
      
      path_energies [i_replica   ] = ener
      path_gradients[i_replica, :] = grad

    return path_energies, path_gradients
  
  def execute(self, 
              num_steps: int = 1, 
              )-> None:
    """Execute the MEP calculation for num_steps optimization steps.
    
      Args: 
        num_steps (int):
          The number of optimization steps. 
    """
    cos, optimizer = self.get_cos(), self.get_optimizer()

    for _ in range(num_steps):
      # Get path_colvar. 
      # NOTE: path_colvar is updated one-way: Path->Replica.
      path_colvar = self.get_path_colvar()

      # Compute path_energies and path_gradients.
      path_energies, path_gradients = self.compute_path_potential_energies_and_gradients()
      
      # Before Path descent: if restraints are to be applied.
      if cos.is_restraint_based() == True:
        path_gradients = cos.apply_restraint(path_colvar=path_colvar, 
                                             path_energies=path_energies, 
                                             path_gradients=path_gradients, )
      
      # Apply Path descent.
      path_colvar_descent = optimizer.apply_path_descent(path_colvar=path_colvar, 
                                                         path_energies=path_energies, 
                                                         path_gradients=path_gradients, )
      
      # Apply Path fixing.
      path_colvar_descent = optimizer.apply_path_fixing(path_colvar=path_colvar, 
                                                        path_colvar_descent=path_colvar_descent, )
      
      # After Path descent: if Path constraints are to be applied.
      if cos.is_constraint_based() == True:
        path_colvar_descent_constrained = cos.apply_constraint(path_colvar=path_colvar_descent, 
                                                               path_energies=path_energies, )
        
      # Update the path_colvar and broadcast to Replica region coordinates.
      self.set_path_colvar(path_colvar=path_colvar_descent_constrained, 
                           update_to_replicas=True, )


