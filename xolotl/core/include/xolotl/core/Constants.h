/*
 * Constants.h
 *
 *  Created on: May 6, 2013
 *      Author: bkj
 */
#pragma once

#include <cmath>
#include <memory>

namespace xolotl
{
namespace core
{
//! Definitions of fundamental constants used in Xolotl

//! The Boltzmann constant in units of eV K^-1.
constexpr double kBoltzmann = 8.61733240000000000E-5;

//! Pi, taken from "100000 digits of Pi,"
//! at http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
constexpr double pi = 3.1415926535897932;

//! Lattice Parameter. Equal to 3.17 Angstroms, taken from Becquart et. al.
//! Journal of Nuclear Materials 403 (2010) 75–88. Given in units here of nm.
constexpr double tungstenLatticeConstant = 0.31700000000000000;

//! Lattice Parameter. Given in units here of nm.
constexpr double alloyLatticeConstant = 0.36000000000000000;

//! Lattice Parameter for UO2
constexpr double uraniumDioxydeLatticeConstant = 0.54700000000000000;

//! Lattice Parameter for Iron
constexpr double ironLatticeConstant = 0.28700000000000000;

//! Lattice Parameter. Given in units here of nm.
constexpr double alphaZrLatticeConstant = 0.32294500000000000; // the a-constant

//! Core radius. Given in units here of nm.
constexpr double alloyCoreRadius = 0.36000000000000000;

//! Core radius. Given in units here of nm.
constexpr double alphaZrCoreRadius = 0.64500000000000000;

//! Single helium radius. Given in units here of nm.
constexpr double heliumRadius = 0.30000000000000000;

//! Single xenon radius. Given in units here of nm.
constexpr double xenonRadius = 0.30000000000000000;

// Tungsten heat coefficient = 1 / (rho * C)
// constexpr double tungstenHeatCoefficient = 1.236e13;
constexpr double tungstenHeatCoefficient = 3.8868e20;

// UO2 heat coefficient = lambda / (rho * C) in nm2 s-1
constexpr double uo2HeatCoefficient = 0.0;

// Iron heat coefficient = lambda / (rho * C) in nm2 s-1
constexpr double feHeatCoefficient = 0.0;

// Alloy heat coefficient = lambda / (rho * C) in nm2 s-1
constexpr double alloyHeatCoefficient = 0.0;

// Alloy heat coefficient = lambda / (rho * C) in nm2 s-1
constexpr double alphaZrHeatCoefficient = 0.0;

// Tungsten heat conductivity = lambda in W K-1 nm-1
// constexpr double tungstenHeatConductivity = 31.8 * 1.0e-9;
constexpr double tungstenHeatConductivity = 159 * 1.0e-9;

// UO2 heat conductivity = lambda in W K-1 m-1
constexpr double uo2HeatConductivity = 0.0;

// Iron heat conductivity = lambda in W K-1 m-1
constexpr double feHeatConductivity = 0.0;

// Alloy heat conductivity = lambda in W K-1 m-1
constexpr double alloyHeatConductivity = 0.0;

// Alloy heat conductivity = lambda in W K-1 m-1
constexpr double alphaZrHeatConductivity = 0.0;

// Burgers vector magnitudes for loops in Alloy case
// In lattice parameter units
constexpr double perfectBurgers = 0.5;
constexpr double faultedBurgers = 0.333;
constexpr double frankBurgers = 0.333;

// Sink strength for Alloy case in nm^-2
constexpr double alloySinkStrength = 1.0e-5;

// Sink strength for alphaZr case in nm^-2
constexpr double alphaZrASinkStrength = 7.25e-8; // Single crystal
constexpr double alphaZrCSinkStrength = 2.25e-8; // Single crystal
// constexpr double alphaZrASinkStrength = 2e-5; //cold-worked Zr
// constexpr double alphaZrCSinkStrength = 5e-6; //cold-worked Zr

// Transition size for basal faulted pyramids to basal c-loops
constexpr double basalTransitionSize = 325;

// Tungsten density in nm^-3
constexpr double tungstenDensity = 62.8;

// Network dislocation density initial in nm^-2
constexpr double rho_d = 1E-6;

// Shear Modulus Tungsten in N/nm2
constexpr double G_W = 1.6E-7;

// Burgers vector Tungsten in nm
constexpr double B_W = 0.274;

// Tungsten atomic volume in nm
constexpr double atvol = 0.0158;

// Boltzman constant in eV/K (Repeated)
constexpr double kb = 8.61733240000000000E-5;

} /* end namespace core */
} /* end namespace xolotl */
