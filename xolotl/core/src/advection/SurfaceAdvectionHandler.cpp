// Includes
#include <xolotl/core/advection/SurfaceAdvectionHandler.h>

namespace xolotl
{
namespace core
{
namespace advection
{
void
SurfaceAdvectionHandler::syncAdvectionGrid()
{
	advecGrid = Kokkos::View<int****>(
		Kokkos::ViewAllocateWithoutInitializing("Advection Grid"),
		advectionGrid.size(), advectionGrid[0].size(),
		advectionGrid[0][0].size(), advectingClusters.size());
	auto advGrid_h = create_mirror_view(advecGrid);
	for (IdType k = 0; k < advectionGrid.size(); ++k) {
		for (IdType j = 0; j < advectionGrid[0].size(); ++j) {
			for (IdType i = 0; i < advectionGrid[0][0].size(); ++i) {
				for (IdType n = 0; n < advectingClusters.size(); ++n) {
					advGrid_h(k, j, i, n) = advectionGrid[k][j][i][n];
				}
			}
		}
	}
	deep_copy(advecGrid, advGrid_h);
}

void
SurfaceAdvectionHandler::initializeAdvectionGrid(
	std::vector<IAdvectionHandler*> advectionHandlers, std::vector<double> grid,
	int nx, int xs, int ny, double hy, int ys, int nz, double hz, int zs)
{
	// Get the number of advecting clusters
	int nAdvec = advectingClusters.size();

	// Initialize the advection grid with true everywhere
	advectionGrid.clear();
	// Initialize it to True
	for (int k = 0; k < nz + 2; k++) {
		std::vector<std::vector<std::vector<bool>>> tempGridTer;
		for (int j = 0; j < ny + 2; j++) {
			std::vector<std::vector<bool>> tempGridBis;
			for (int i = 0; i < nx + 2; i++) {
				tempGridBis.emplace_back(nAdvec, true);
			}
			tempGridTer.push_back(tempGridBis);
		}
		advectionGrid.push_back(tempGridTer);
	}

	// Initialize the grid position
	plsm::SpaceVector<double, 3> gridPosition{0.0, 0.0, 0.0};

	// Consider each advection handler.
	for (auto const& currAdvecHandler : advectionHandlers) {
		// Get the list of advecting clusters
		auto const& otherAdvecClusters =
			currAdvecHandler->getAdvectingClusters();

		// Loop on the spatial grid
		for (int k = -1; k < nz + 1; k++) {
			// Set the grid position
			gridPosition[2] = hz * (double)(k + zs);
			for (int j = -1; j < ny + 1; j++) {
				// Set the grid position
				gridPosition[1] = hy * (double)(j + ys);
				for (int i = 0; i < nx + 2; i++) {
					// Set the grid position
					if (i + xs == nx + 1) {
						gridPosition[0] = grid[i + xs] +
							(grid[i + xs] - grid[i + xs - 1]) / 2.0;
					}
					else {
						gridPosition[0] =
							(grid[i + xs] + grid[i + xs + 1]) / 2.0;
					}

					// Check if we are on a sink
					if (currAdvecHandler->isPointOnSink(gridPosition)) {
						// We have to find the corresponding index in the
						// diffusion index vector
						for (auto const currAdvCluster : otherAdvecClusters) {
							auto it = find(advectingClusters.begin(),
								advectingClusters.end(), currAdvCluster);
							if (it != advectingClusters.end()) {
								// Set this diffusion grid value to false
								advectionGrid[k + 1][j + 1][i + 1][(*it)] =
									false;
							}
							else {
								throw std::runtime_error(
									"\nThe advecting cluster of id: " +
									std::to_string(currAdvCluster) +
									" was not found in the advecting clusters, "
									"cannot "
									"use the advection!");
							}
						}
					}
				}
			}
		}
	}

	syncAdvectionGrid();
}

void
SurfaceAdvectionHandler::computeAdvection(network::IReactionNetwork& network,
	const plsm::SpaceVector<double, 3>& pos, const StencilConcArray& concVector,
	Kokkos::View<double*> updatedConcOffset, double hxLeft, double hxRight,
	int ix, double hy, int iy, double hz, int iz) const
{
	if (concVector.size() < 3) {
		throw std::runtime_error("Wrong size for 1D concentration stencil; "
								 "should be at least 3, got " +
			std::to_string(concVector.size()));
	}
	Kokkos::Array<Kokkos::View<const double*>, 3> concVec = {
		concVector[0], concVector[1], concVector[2]};

	auto location_ = location;
	auto clusterIds = this->advClusterIds;
	auto clusters = this->advClusters;
	auto sinkStrengths = this->advSinkStrengths;
	auto advGrid = this->advecGrid;

	// Consider each advecting cluster
	Kokkos::parallel_for(
		clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
			auto currId = clusterIds[i];
			auto cluster = clusters[i];

			// Get the initial concentrations
			double oldConc = concVec[0][currId] *
				advGrid(iz + 1, iy + 1, ix + 1, i); // middle
			double oldRightConc = concVec[2][currId] *
				advGrid(iz + 1, iy + 1, ix + 2, i); // right

			// Compute the concentration as explained in the description of the
			// method
			double conc = (3.0 * sinkStrengths[i] *
							  cluster.getDiffusionCoefficient(ix + 1)) *
				((oldRightConc / pow(pos[0] - location_ + hxRight, 4)) -
					(oldConc / pow(pos[0] - location_, 4))) /
				(kBoltzmann * cluster.getTemperature(ix + 1) * hxRight);

			conc += (3.0 * sinkStrengths[i] * oldConc) *
				(cluster.getDiffusionCoefficient(ix + 2) /
						cluster.getTemperature(ix + 2) -
					cluster.getDiffusionCoefficient(ix + 1) /
						cluster.getTemperature(ix + 1)) /
				(kBoltzmann * hxRight * pow(pos[0] - location_, 4));

			// Update the concentration of the cluster
			updatedConcOffset[currId] += conc;
		});
}

void
SurfaceAdvectionHandler::computePartialsForAdvection(
	network::IReactionNetwork& network, Kokkos::View<double*> val,
	const plsm::SpaceVector<double, 3>& pos, double hxLeft, double hxRight,
	int ix, double hy, int iy, double hz, int iz) const
{
	auto location_ = location;
	auto clusterIds = this->advClusterIds;
	auto clusters = this->advClusters;
	auto sinkStrengths = this->advSinkStrengths;
	auto advGrid = this->advecGrid;

	// Consider each advecting cluster.
	Kokkos::parallel_for(
		clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
			auto currId = clusterIds[i];
			auto cluster = clusters[i];

			// Get the diffusion coefficient of the cluster
			double diffCoeff = cluster.getDiffusionCoefficient(ix + 1);
			// Get the sink strength value
			double sinkStrength = sinkStrengths[i];

			// Compute the partial derivatives for advection of this cluster as
			// explained in the description of this method
			val[i * 2] = -(3.0 * sinkStrength * diffCoeff) /
				(kBoltzmann * cluster.getTemperature(ix + 1) * hxRight *
					pow(pos[0] - location_, 4)) *
				advGrid(iz + 1, iy + 1, ix + 1, i); // middle
			val[i * 2] += (3.0 * sinkStrength) *
				(cluster.getDiffusionCoefficient(ix + 2) /
						cluster.getTemperature(ix + 2) -
					cluster.getDiffusionCoefficient(ix + 1) /
						cluster.getTemperature(ix + 1)) /
				(kBoltzmann * hxRight * pow(pos[0] - location_, 4)) *
				advGrid(iz + 1, iy + 1, ix + 1, i); // middle
			val[(i * 2) + 1] = (3.0 * sinkStrength * diffCoeff) /
				(kBoltzmann * cluster.getTemperature(ix + 1) * hxRight *
					pow(pos[0] - location_ + hxRight, 4)) *
				advGrid(iz + 1, iy + 1, ix + 2, i); // right
		});
}

} /* end namespace advection */
} /* end namespace core */
} /* end namespace xolotl */
