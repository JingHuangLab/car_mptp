#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The String Method."""

import numpy as np

from scipy.interpolate import CubicSpline

from pycospath.utils.geometry import get_rowwise_projection

from pycospath.cos import CoS

class StringM(CoS):
  """Implements the String Methods. 

    String method for the study of rare events.
    W. E, W. Ren, E. Vanden-Eijnden, Phys. Rev. B, 2002, 66, 052301. DOI: 10.1103/PhysRevB.66.052301
  
    Simplified and improved string method for computing the minimum energy paths in barrier-crossing
    events.
    W. E, W. Ren, E. Vanden-Eijnden, J. Chem. Phys., 2007, 126, 164103. DOI: 10.1063/1.2720838
  """

  CONSTRAINT_BASED = True
  RESTRAINT_BASED  = True # Gradient projections

  METHOD_REPARAM = ['cspline']

  def __init__(self,
               use_gradient_projection:  bool  = False,
               reparametrization_method: str = 'cspline', 
               ) -> None:
    """Initializes a String Method CoS condition instance.
    
      Args:
        use_gradient_projection (bool, optional):
          Specifies if Gradient Projection should be used for the String method. If True, it is the 
          original String Method with gradient projections; If False, it is the simplified String 
          Method. 
          Default: False;
        reparametrization_method (str, optional):
          Specifies the Path function used to reparametrize the string. 
          Default: 'cspline'.
            'cspline': cubic splines.
    """
    self._use_grad_proj = True if use_gradient_projection==True else False

    assert isinstance(reparametrization_method, str),       "Illegal reparametrization_method type."
    assert reparametrization_method in self.METHOD_REPARAM, "Illegal reparametrization_method spec."

    if reparametrization_method == 'cspline':
      self._intpol_class = CubicSpline

    else: # Default.
      self._intpol_class = CubicSpline

  def _get_reparametrized_path_colvar(self,
                                      path_colvar: np.ndarray, 
                                      old_nodes:   np.ndarray, 
                                      new_nodes:   np.ndarray,
                                      return_grad: bool, 
                                      ) -> np.ndarray:
    """Return the reparametrized Path colvar. The Path colvar is reparametrized using x-alphas and 
      y-path_colvar, and is redistributed to new_alphas. 

      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        old_nodes (np.ndarray):
          The Path node array normalized to [0., 1.], shape (num_replicas, ).
        new_nodes (np.ndarray):
          The Path node array to where the Path colvars are redistributed, shape (num_replicas, ).
        return_grad (bool):
          If true, returns the Path tangent vectors at the redistributed Path colvars; Otherwise, 
          returns the redistributed Path colvars.

      Returns:
        path_reparam (np.ndarray):
          The Path tangent vectors at new_alphas if return_grad is True, or the redistributed Path 
          colvars at new_alphas if return_grad is False.
    """
    nu = 1 if return_grad == True else 0
    path_reparam = np.zeros(path_colvar.shape)
    
    for i_dof in range(path_colvar.shape[1]):
      intpol = self._intpol_class(x=old_nodes, y=path_colvar[:, i_dof])
      path_reparam[:, i_dof] = intpol.__call__(x=new_nodes, nu=nu)

    return path_reparam
  
  def apply_restraint(self, 
                      path_colvar:    np.ndarray, 
                      path_gradients: np.ndarray,
                      **kwargs) -> np.ndarray:
    """Apply the String Method restraint condition on the Path gradient.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_grad (np.ndarray):
          The Path gradients, shape (num_replicas, num_replica_dofs).
    
      Returns:
        restrained_path_gradients (np.ndarray):
          The restrained Path gradients, shape (num_replicas, num_replica_dofs).
    """
    if self._use_grad_proj == True:
      rms = np.zeros((path_colvar.shape[0], ))
      rms[1:] = np.linalg.norm(path_colvar[1:, :] - path_colvar[:-1, :], axis=1)
      path_nodes = np.cumsum(rms) / np.sum(rms)
      
      path_tan = self._get_reparametrized_path_colvar(path_colvar = path_colvar,
                                                      old_nodes   = path_nodes,
                                                      new_nodes   = path_nodes,
                                                      return_grad = True, )
      path_gradients[1:-1] -= get_rowwise_projection(array_to_refer=  path_tan[1:-1], 
                                                      array_to_project=path_gradients[1:-1], )
    
    return path_gradients
  
  def apply_constraint(self, 
                       path_colvar: np.ndarray, 
                       **kwargs) -> np.ndarray:
    """Apply the String Method constraint condition on the Path colvar.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
    
      Returns:
        constrained_path_colvar (np.ndarray):
          The constrained Path colvar, shape (num_replicas, num_replica_dofs).
    """
    rms     = np.zeros((path_colvar.shape[0], ))
    rms[1:] = self.get_path_weighted_rms(path_colvar=path_colvar)
    
    path_nodes = np.cumsum(rms) / np.sum(rms)
    unif_nodes = np.linspace(0., 1., num=path_colvar.shape[0])

    path_reparam = self._get_reparametrized_path_colvar(path_colvar=path_colvar,
                                                        old_nodes=path_nodes,
                                                        new_nodes=unif_nodes, 
                                                        return_grad=False, )

    return path_reparam


