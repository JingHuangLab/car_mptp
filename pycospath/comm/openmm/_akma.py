#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""AKMA Unit converters for OpenMM Quantity objects."""

import numpy as np

from openmm import Vec3 as OpenMMVec3

from openmm.unit import (Quantity as OpenMMQuantity, 
                         MOLAR_GAS_CONSTANT_R, )

# AKMAs
from openmm.unit import (angstrom             as unit_angstrom,
                         picosecond           as unit_picosecond, 
                         atomic_mass_unit     as unit_atomic_mass, 
                         kilocalorie_per_mole as unit_kcal_p_mol, 
                         kelvin               as unit_kelvin, )

# OpenMM SIs.
from openmm.unit import (nanometer as unit_nanometer, 
                         kilojoule_per_mole as unit_kj_p_mol, )

# Constants.

OPENMM_MOLAR_GAS_CONSTANT = MOLAR_GAS_CONSTANT_R

# OpenMM to AKMA converters.

def openmm_mass_to_akma(openmm_mass: OpenMMQuantity) -> np.ndarray:
  """Convert the mass measuring OpenMMQuantity to AKMA np.ndarray."""
  return np.asarray(openmm_mass.value_in_unit(unit=unit_atomic_mass))


def openmm_time_to_akma(openmm_time: OpenMMQuantity) -> float:
  """Convert the time measuring OpenMMQuantity to AKMA float."""
  return float(openmm_time.value_in_unit(unit=unit_picosecond))


def openmm_energy_to_akma(openmm_energy: OpenMMQuantity) -> float:
  """Convert the energy measuring OpenMMQuantity to AKMA float."""
  return float(openmm_energy.value_in_unit(unit=unit_kcal_p_mol))


def openmm_forces_to_akma(openmm_forces: OpenMMQuantity) -> np.ndarray:
  """Convert the force measuring OpenMMQuantity to AKMA np.ndarray."""
  return np.asarray(openmm_forces.value_in_unit(unit=unit_kcal_p_mol/unit_angstrom))


def openmm_velocities_to_akma(openmm_velocities: OpenMMQuantity) -> np.ndarray:
  """Convert the velocities measuring OpenMMQuantity to AKMA np.ndarray."""
  return np.asarray(openmm_velocities.value_in_unit(unit=unit_angstrom/unit_picosecond))


def openmm_coordinates_to_akma(openmm_coordinates: OpenMMQuantity) -> np.ndarray:
  """Convert the coordinates measuring OpenMMQuantity to AKMA np.ndarray."""
  return np.asarray(openmm_coordinates.value_in_unit(unit=unit_angstrom))


def openmm_temperature_to_akma(openmm_temperature: OpenMMQuantity) -> float:
  """Convert the temperature meansuring OpenMMQuantity to AKMA float."""
  return float(openmm_temperature.value_in_unit(unit=unit_kelvin))


def openmm_friction_coef_to_akma(openmm_friction_coef: OpenMMQuantity) -> float:
  """Convert the friction coefficient measuring OpenMMQuantity to AKMA float."""
  return float(openmm_friction_coef.value_in_unit(unit=unit_picosecond**-1))


def openmm_force_constant_to_akma(openmm_force_constant: OpenMMQuantity) -> float:
  """Convert the force constant measuring OpenMMQuantity to AKMA float."""
  return float(openmm_force_constant.value_in_unit(unit=unit_kcal_p_mol/unit_angstrom**2))


# OpenMM unit remover.

def openmm_energy_to_np_array(openmm_energy: OpenMMQuantity) -> float:
  """Convert the energy measuring OpenMMQuantity to np.ndarray."""
  return float(openmm_energy.value_in_unit(unit=unit_kj_p_mol))


def openmm_forces_to_np_array(openmm_forces: OpenMMQuantity) -> np.ndarray:
  """Convert the forces measuring OpenMMQuantity to np.ndarray."""
  return np.asarray(openmm_forces.value_in_unit(unit=unit_kj_p_mol/unit_nanometer))


def openmm_velocities_to_np_array(openmm_velocities: OpenMMQuantity) -> np.ndarray:
  """Convert the velocities measuring OpenMMQuantity to np.ndarray."""
  return np.asarray(openmm_velocities.value_in_unit(unit=unit_nanometer/unit_picosecond))


def openmm_coordinates_to_np_array(openmm_coordinates: OpenMMQuantity) -> np.ndarray:
  """Convert the coordinates measuring OpenMMQuantity to np.ndarray."""
  return np.asarray(openmm_coordinates.value_in_unit(unit=unit_nanometer))


# AKMA to OpenMM converters.

def akma_mass_to_openmm(akma_mass: np.ndarray) -> OpenMMQuantity:
  """Convert the mass measuring AKMA np.ndarray to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_mass, unit=unit_atomic_mass)


def akma_time_to_openmm(akma_time: float) -> OpenMMQuantity:
  """Convert the time measuring AKMA float to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_time, unit=unit_picosecond)


def akma_energy_to_openmm(akma_energy: np.ndarray) -> OpenMMQuantity:
  """Convert the energy measuring AKMA np.ndarray to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_energy, unit=unit_kcal_p_mol)


def akma_forces_to_openmm(akma_forces: np.ndarray) -> OpenMMQuantity:
  """Convert the force measuring AKMA np.ndarray to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_forces, unit=unit_kcal_p_mol/unit_angstrom)


def akma_velocities_to_openmm(akma_velocities: np.ndarray) -> OpenMMQuantity:
  """Convert the velocities measuring AKMA np.ndarray to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_velocities, unit=unit_angstrom/unit_picosecond)


def akma_coordinates_to_openmm(akma_coordinates: np.ndarray) -> OpenMMQuantity:
  """Convert the coordinates meansuring AKMA np.ndarray to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_coordinates, unit=unit_angstrom)


def akma_temperature_to_openmm(akma_temperature: float) -> OpenMMQuantity:
  """Convert the temperature measuring AKMA float to OpenMMQuantity"""
  return OpenMMQuantity(value=akma_temperature, unit=unit_kelvin)


def akma_friction_coef_to_openmm(akma_friction_coef: float) -> OpenMMQuantity:
  """Convert the friction coefficient measuring AKMA float to OpenMMQuantity."""
  return OpenMMQuantity(value=akma_friction_coef, unit=unit_picosecond**-1)


def akma_force_constant_to_openmm(akma_force_constant: float) -> OpenMMQuantity:
  """Convert the force constant measuring OpenMMQuantity to AKMA float."""
  return OpenMMQuantity(value=akma_force_constant, unit=unit_kcal_p_mol/unit_angstrom**2)

