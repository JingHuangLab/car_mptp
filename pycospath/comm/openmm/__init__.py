#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Communicators for OpenMM potentials."""

from ._akma import (OpenMMVec3, 
                    OpenMMQuantity, 
                    OPENMM_MOLAR_GAS_CONSTANT,  
                    
                    openmm_mass_to_akma, 
                    openmm_time_to_akma, 
                    openmm_energy_to_akma, 
                    openmm_forces_to_akma, 
                    openmm_velocities_to_akma, 
                    openmm_coordinates_to_akma, 
                    openmm_temperature_to_akma, 
                    openmm_friction_coef_to_akma, 
                    openmm_force_constant_to_akma, 
                    
                    openmm_energy_to_np_array,
                    openmm_forces_to_np_array,
                    openmm_velocities_to_np_array,
                    openmm_coordinates_to_np_array,

                    akma_mass_to_openmm,  
                    akma_time_to_openmm, 
                    akma_energy_to_openmm, 
                    akma_forces_to_openmm, 
                    akma_velocities_to_openmm, 
                    akma_coordinates_to_openmm, 
                    akma_temperature_to_openmm, 
                    akma_friction_coef_to_openmm, 
                    akma_force_constant_to_openmm, )

from ._openmm_forces import OpenMMRmsdCVForce

from ._openmm_integrators import (OpenMMScaledLangevinIntegrator, 
                                  # OpenMMMassScaledLangevinIntegrator, 
                                  OpenMMReplicaScaledLangevinIntegrator, 
                                  
                                  get_velocities_time_offset, )

__all__ = [
  'OpenMMVec3',
  'OpenMMQuantity', 
  'OPENMM_MOLAR_GAS_CONSTANT', 

  'openmm_mass_to_akma', 
  'openmm_time_to_akma', 
  'openmm_energy_to_akma', 
  'openmm_forces_to_akma', 
  'openmm_velocities_to_akma', 
  'openmm_coordinates_to_akma', 
  'openmm_temperature_to_akma', 
  'openmm_friction_coef_to_akma', 
  'openmm_force_constant_to_akma', 

  'openmm_energy_to_np_array',
  'openmm_forces_to_np_array',
  'openmm_velocities_to_np_array',
  'openmm_coordinates_to_np_array',

  'akma_mass_to_openmm',  
  'akma_time_to_openmm', 
  'akma_energy_to_openmm', 
  'akma_forces_to_openmm', 
  'akma_velocities_to_openmm', 
  'akma_coordinates_to_openmm', 
  'akma_temperature_to_openmm', 
  'akma_friction_coef_to_openmm', 
  'akma_force_constant_to_openmm', 

  'OpenMMRmsdCVForce', 

  'OpenMMScaledLangevinIntegrator', 
  # 'OpenMMMassScaledLangevinIntegrator', 
  'OpenMMReplicaScaledLangevinIntegrator', 

  'get_velocities_time_offset', 
]
