#pragma once

#include <xolotl/core/temperature/ITemperatureHandler.h>

namespace xolotl
{
namespace core
{
namespace temperature
{
class TemperatureHandler : public ITemperatureHandler
{
public:
	TemperatureHandler() = default;

	virtual ~TemperatureHandler();

	/**
	 * \see ITemperatureHandler.h
	 */
	void
	initialize(int dof) override;

	/**
	 * This operation sets the temperature given by the solver.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	void
	setTemperature(Kokkos::View<const double*> solution) override
	{
		return;
	}

	/**
	 * This operation sets the heat coefficient to use in the equation.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	void
	setHeatCoefficient(double coef) override
	{
		return;
	}

	/**
	 * This operation sets the heat conductivity to use in the equation.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	void
	setHeatConductivity(double cond) override
	{
		return;
	}

	/**
	 * This operation sets the surface position.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	void
	updateSurfacePosition(int surfacePos, std::vector<double> grid) override
	{
		return;
	}

	/**
	 * Compute the flux due to the heat equation.
	 * This method is called by the RHSFunction from the solver.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	void
	computeTemperature(double currentTime,
		Kokkos::View<const double*>* concVector,
		Kokkos::View<double*> updatedConcOffset, double hxLeft, double hxRight,
		int xi, double sy = 0.0, int iy = 0, double sz = 0.0,
		int iz = 0) override
	{
		return;
	}

	/**
	 * Compute the partials due to the heat equation.
	 * This method is called by the RHSJacobian from the solver.
	 * Don't do anything.
	 *
	 * \see ITemperatureHandler.h
	 */
	bool
	computePartialsForTemperature(double currentTime, const double** concVector,
		double* val, IdType* indices, double hxLeft, double hxRight, int xi,
		double sy = 0.0, int iy = 0, double sz = 0.0, int iz = 0) override
	{
		return false;
	}

	/**
	 * Get the heat flux at this time.
	 *
	 * @param currentTime The current time
	 * @return The heat flux
	 */
	double
	getHeatFlux(double currentTime) override
	{
		return 0.0;
	}

protected:
	/**
	 * The number of degrees of freedom in the network
	 */
	int _dof;

	/**
	 * The x grid
	 */
	std::vector<double> xGrid;
};
} // namespace temperature
} // namespace core
} // namespace xolotl
