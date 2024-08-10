#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""OpenMM Langevin Integrator with Selective Force Scaling."""

from math import exp, sqrt

import numpy as np

from openmm import (Integrator as OpenMMIntegrator, 
                    LangevinIntegrator as OpenMMLangevinIntegrator, 
                    CustomIntegrator as OpenMMCustomIntegrator, )

from pycospath.comm.openmm import (OpenMMVec3, 
                                   OpenMMQuantity, 
                                   OPENMM_MOLAR_GAS_CONSTANT, 
                                   akma_time_to_openmm, 
                                   akma_temperature_to_openmm, 
                                   akma_friction_coef_to_openmm, 
                                   openmm_time_to_akma, )

class OpenMMScaledLangevinIntegrator(OpenMMCustomIntegrator):
  """The OpenMM LangevinIntegrator with potential gradients scaling."""

  # Boundaries on the parameters.
  _zero_timestep_size = akma_time_to_openmm         (akma_time=0.         )
  _zero_temperature   = akma_temperature_to_openmm  (akma_temperature=0.  )
  _zero_friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=0.)

  def __init__(self, 
               timestep_size: float | OpenMMQuantity =    .001,
               friction_coef: float | OpenMMQuantity =   1.   , 
               temperature:   float | OpenMMQuantity = 300.   , 
               scaling_coef:  float                  =   1.   , 
               ) -> None:
    """Create an OpenMM LangevinIntegrator with potential gradients scaling.
    
      Args:
        timestep_size (float | OpenMMQuantity, optional):
          The stepsize for each timestep, unit: ps. 
          Default: .001, allowed values are positives. 
        friction_coef (float | OpenMMQuantity, optional): 
          The friction coefficient, unit: ps**-1. 
          Default: 1., allowed values are non-negatives. 
        temperature (float | OpenMMQuantity, optional):
          The temperature of the heat bath, unit: kelvin.
          Default: 300., allowed values are non-negatives. 
        scaling_coef (float, optional):
          The scaling factor for potential gradients.
          Default: 1., allowed values are non-negatives.
    """
    # floats to OpenMMQuantity.
    if isinstance(timestep_size, float):
      timestep_size = akma_time_to_openmm(akma_time=timestep_size)
    if isinstance(friction_coef, float):
      friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=friction_coef)
    if isinstance(temperature,   float):
      temperature   = akma_temperature_to_openmm(akma_temperature=temperature)
    
    # Sanity checks.
    assert isinstance(timestep_size, OpenMMQuantity), "Illegal timestep_size type."
    assert timestep_size > self._zero_timestep_size,  "Illegal timestep_size spec."
    self._dt = timestep_size

    assert isinstance(friction_coef, OpenMMQuantity), "Illegal friction_coef type."
    assert friction_coef >= self._zero_friction_coef, "Illegal friction_coef spec."
    self._gamma = friction_coef

    assert isinstance(temperature, OpenMMQuantity), "Illegal temperature type."
    assert temperature >= self._zero_temperature,   "Illegal temperature spec."
    self._temperature = temperature

    assert isinstance(scaling_coef, float), "Illegal scaling_coef type."
    assert scaling_coef >= 0.,              "Illegal scaling_coef spec."
    self._scaling_coef = scaling_coef

    # Actual init. ---------------------------------------------------------------------------------
    
    OpenMMCustomIntegrator.__init__(self, self._dt)

    self.addGlobalVariable('a', exp(-self._gamma*self._dt))

    self.addGlobalVariable('b', self._dt)
    if self._gamma > self._zero_friction_coef:
      self.setGlobalVariableByName('b', (1.-exp(-self._gamma*self._dt)) / self._gamma)

    self.addGlobalVariable('c', sqrt(1.-exp(-2.*self._gamma*self._dt)))

    self.addGlobalVariable('kBT', OPENMM_MOLAR_GAS_CONSTANT*self._temperature)

    self.addGlobalVariable('fscal', self._scaling_coef)

    ## Add a temporal value for last step coordinates to the context.
    self.addPerDofVariable("x0", 0.)

    # Integration algorihtm. -----------------------------------------------------------------------

    ## Allow forces to change the Context (coordinates), e.g., BaroStats.
    self.addUpdateContextState()
    ## Record full step current coordinates.
    self.addComputePerDof("x0", "x")
    ## Integrate half step velocities.
    self.addComputePerDof("v",  "v*a + f*fscal*b/m + gaussian*c*sqrt(kBT/m)")
    ## Update full step next coordinates.
    self.addComputePerDof("x",  "x + v*dt")
    ## Impose positional constraints.
    self.addConstrainPositions()
    ## Update half step next velocities.
    self.addComputePerDof("v",  "(x-x0)/dt")
    
    # Sets the kinetic energy expression: half step velocities shifting with scaled potential.
    self.setKineticEnergyExpression("m*v1*v1/2; v1=v+0.5*dt*f*fscal/m")



class OpenMMMassScaledLangevinIntegrator(OpenMMCustomIntegrator):
  """The OpenMM LangevinIntegrator with mass scaling."""

  # Boundaries on the paramters.
  _zero_timestep_size = akma_time_to_openmm         (akma_time=0.         )
  _zero_temperature   = akma_temperature_to_openmm  (akma_temperature=0.  )
  _zero_friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=0.)

  def __init__(self, 
               timestep_size: float | OpenMMQuantity =    .001,
               friction_coef: float | OpenMMQuantity =   1.   , 
               temperature:   float | OpenMMQuantity = 300.   , 
               scaling_mass:  np.ndarray | None      = None   , 
               ) -> None:
    """Create an OpenMM LangevinIntegrator with mass scaling.
    
      Args:
        timestep_size (float | OpenMMQuantity, optional):
          The stepsize for each timestep, unit: ps. 
          Default: .001, allowed values are positives. 
        friction_coef (float | OpenMMQuantity, optional): 
          The friction coefficient, unit: ps**-1. 
          Default: 1., allowed values are non-negatives. 
        temperature (float | OpenMMQuantity, optional):
          The temperature of the heat bath, unit: kelvin.
          Default: 300., allowed values are non-negatives. 
        scaling_mass (np.ndarray | None, optional):
          The scaling factor of the per-DOF masses in the OpenMM system, shape (num_context_dofs, ).
          Default: None for identity scaling.
    """
    # floats to OpenMMQuantity.
    if isinstance(timestep_size, float):
      timestep_size = akma_time_to_openmm(akma_time=timestep_size)
    if isinstance(friction_coef, float):
      friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=friction_coef)
    if isinstance(temperature,   float):
      temperature   = akma_temperature_to_openmm(akma_temperature=temperature)

    # Sanity checks.
    assert isinstance(timestep_size, OpenMMQuantity), "Illegal timestep_size type."
    assert timestep_size > self._zero_timestep_size,  "Illegal timestep_size spec."
    self._timestep_size = timestep_size

    assert isinstance(friction_coef, OpenMMQuantity), "Illegal friction_coef type."
    assert friction_coef >= self._zero_friction_coef, "Illegal friction_coef spec."
    self._gamma = friction_coef

    assert isinstance(temperature, OpenMMQuantity), "Illegal temperature type."
    assert temperature >= self._zero_temperature,   "Illegal temperature spec."
    self._temperature = temperature

    # Actual init. ---------------------------------------------------------------------------------

    OpenMMCustomIntegrator.__init__(self, self._timestep_size)

    self.addGlobalVariable('a', exp(-self._gamma*self._timestep_size))

    self.addGlobalVariable('b', self._timestep_size)
    if self._gamma > self._zero_friction_coef:
      self.setGlobalVariableByName('b', (1.-exp(-self._gamma*self._timestep_size)) / self._gamma)

    self.addGlobalVariable('c', sqrt(1.-exp(-2.*self._gamma*self._timestep_size)))

    self.addGlobalVariable('kBT', OPENMM_MOLAR_GAS_CONSTANT*self._temperature)

    # Adds the per-DOF mass scaling variables to the Context.
    self.addPerDofVariable('mscal', 1.)
    self._scaling_mass = None # Default.

    if not scaling_mass is None:
      assert isinstance(scaling_mass, np.ndarray), "Illegal scaling_mass type."
      assert ( scaling_mass > 0. ).all(),          "Illegal scaling_mass spec."
      tmp = list(scaling_mass.reshape(-1, 3))
      self._scaling_mass = [OpenMMVec3(x=float(r[0]), y=float(r[1]), z=float(r[2])) for r in tmp]
      self.setPerDofVariableByName('mscal', self._scaling_mass)
    
    ## Add a temporal value for last step coordinates to the context.
    self.addPerDofVariable("x0", 0.)

    # Integration algorihtm. -----------------------------------------------------------------------

    ## Allow forces to change the Context (coordinates), e.g., BaroStats.
    self.addUpdateContextState()
    ## Record full step current coordinates.
    self.addComputePerDof("x0", "x")
    ## Integrate half step velocities.
    self.addComputePerDof("v",  "v*a + f*b/m/mscal + gaussian*c*sqrt(kBT/m/mscal)")
    ## Update full step next coordinates.
    self.addComputePerDof("x",  "x + v*dt")
    ## Impose positional constraints.
    self.addConstrainPositions()
    ## Update half step next velocities.
    self.addComputePerDof("v",  "(x-x0)/dt")
    
    # Sets the kinetic energy expression: half step velocities shifting with scaled potential.
    self.setKineticEnergyExpression("m*mscal*v1*v1/2; v1=v+0.5*dt*f/m/mscal")



class OpenMMReplicaScaledLangevinIntegrator(OpenMMCustomIntegrator):

  # Boundaries on the parameters.
  _zero_timestep_size = akma_time_to_openmm         (akma_time=0.         )
  _zero_temperature   = akma_temperature_to_openmm  (akma_temperature=0.  )
  _zero_friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=0.)

  def __init__(self, 
               timestep_size: float | OpenMMQuantity =    .001, 
               friction_coef: float | OpenMMQuantity =   1.   ,
               temperature:   float | OpenMMQuantity = 300.   ,
               scaling_per_dof: np.ndarray           = None, 
               ) -> None:
    """Create an OpenMM LangevinIntegrator with potential gradients scaling only on Replica atoms.
      
      Args:
        timestep_size (float | OpenMMQuantity, optional):
          The stepsize for each timestep, unit: ps. 
          Default: .001, allowed values are positives. 
        friction_coef (float | OpenMMQuantity, optional): 
          The friction coefficient, unit: ps**-1. 
          Default: 1., allowed values are non-negatives. 
        temperature (float | OpenMMQuantity, optional):
          The temperature of the heat bath, unit: kelvin.
          Default: 300., allowed values are non-negatives. 
        scaling_per_dof (np.ndarray, optional):
          The scaling factor of potential gradient per-DOF in the OpenMM system, shape (num_dofs, ).
          Default: None for identity scaling.
    """
    # floats to OpenMMQuantity.
    if isinstance(timestep_size, float):
      timestep_size = akma_time_to_openmm(akma_time=timestep_size)
    if isinstance(friction_coef, float):
      friction_coef = akma_friction_coef_to_openmm(akma_friction_coef=friction_coef)
    if isinstance(temperature,   float):
      temperature   = akma_temperature_to_openmm(akma_temperature=temperature)
    
    # Sanity checks.
    assert isinstance(timestep_size, OpenMMQuantity), "Illegal timestep_size type."
    assert timestep_size > self._zero_timestep_size,  "Illegal timestep_size spec."
    self._dt = timestep_size

    assert isinstance(friction_coef, OpenMMQuantity), "Illegal friction_coef type."
    assert friction_coef >= self._zero_friction_coef, "Illegal friction_coef spec."
    self._gamma = friction_coef

    assert isinstance(temperature, OpenMMQuantity), "Illegal temperature type."
    assert temperature >= self._zero_temperature,   "Illegal temperature spec."
    self._temperature = temperature

    # Actual init. ---------------------------------------------------------------------------------

    OpenMMCustomIntegrator.__init__(self, self._dt)

    self.addGlobalVariable('a', exp(-self._gamma*self._dt))

    self.addGlobalVariable('b', self._dt)
    if self._gamma > self._zero_friction_coef:
      self.setGlobalVariableByName('b', (1.-exp(-self._gamma*self._dt)) / self._gamma)

    self.addGlobalVariable('c', sqrt(1.-exp(-2.*self._gamma*self._dt)))

    self.addGlobalVariable('kBT', OPENMM_MOLAR_GAS_CONSTANT*self._temperature)

    # Adds the per-DOF force scaling variables to the context.
    self.addPerDofVariable('fscal', 1.)
    self._scaling_per_dof = None # Default.

    if not scaling_per_dof is None:
      assert isinstance(scaling_per_dof, np.ndarray), "Illegal scaling_per_dof type."
      assert ( scaling_per_dof >= 0. ).all(),         "Illegal scaling_per_dof spec."
      tmp = list(scaling_per_dof.reshape(-1, 3))
      self._scaling_per_dof = [OpenMMVec3(x=float(r[0]), y=float(r[1]), z=float(r[2])) for r in tmp]
      self.setPerDofVariableByName('fscal', self._scaling_per_dof)

    ## Add a temporal value for last step coordinates to the context.
    self.addPerDofVariable("x0", 0.)

    # Integration algorihtm. -----------------------------------------------------------------------

    ## Allow forces to change the Context (coordinates), e.g., BaroStats.
    self.addUpdateContextState()
    ## Record full step current coordinates.
    self.addComputePerDof("x0", "x")
    ## Integrate half step velocities.
    self.addComputePerDof("v",  "v*a + f*fscal*b/m + gaussian*c*sqrt(kBT/m)")
    ## Update full step next coordinates.
    self.addComputePerDof("x",  "x + v*dt")
    ## Impose positional constraints.
    self.addConstrainPositions()
    ## Update half step next velocities.
    self.addComputePerDof("v",  "(x-x0)/dt")
    
    # Sets the kinetic energy expression: half step velocities shifting with scaled potential.
    self.setKineticEnergyExpression("m*v1*v1/2; v1=v+0.5*dt*f*fscal/m")





def get_velocities_time_offset(integrator: OpenMMIntegrator) -> float:
  """Get the time interval by which the velocities are offset from the positions."""
  time_offset = 0.

  if isinstance(integrator, _HALFSTEP_VELOCITIES_INTEGRATORS) == True:
    time_offset = .5*openmm_time_to_akma(integrator.getStepSize())
    
  return time_offset


_HALFSTEP_VELOCITIES_INTEGRATORS = (OpenMMLangevinIntegrator, 
                                    OpenMMScaledLangevinIntegrator, 
                                    OpenMMMassScaledLangevinIntegrator, 
                                    OpenMMReplicaScaledLangevinIntegrator, )
"""Collection of OpenMMIntegrators integrates velocities at half timesteps."""
