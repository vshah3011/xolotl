// Includes
#include <xolotl/core/advection/XGBAdvectionHandler.h>
#include <xolotl/core/network/IPSIReactionNetwork.h>

namespace xolotl
{
namespace core
{
namespace advection
{
void
XGBAdvectionHandler::initialize(
	network::IReactionNetwork& network, std::vector<RowColPair>& idPairs)
{
	// Clear the index and sink strength vectors
	advectingClusters.clear();
	sinkStrengthVector.clear();

	using NetworkType = network::IPSIReactionNetwork;
	using AmountType = NetworkType::AmountType;

	auto psiNetwork = dynamic_cast<NetworkType*>(&network);
	auto numSpecies = psiNetwork->getSpeciesListSize();
	auto specIdHe = psiNetwork->getHeliumSpeciesId();

	// Initialize the composition
	auto comp = std::vector<AmountType>(numSpecies, 0);

	// Loop on helium clusters from size 1 to 7
	for (std::size_t i = 1; i <= 7; i++) {
		comp[specIdHe()] = i;
		auto clusterId = psiNetwork->findClusterId(comp);

		// Check that the helium cluster is present in the network
		if (clusterId == NetworkType::invalidIndex()) {
			throw std::runtime_error("\nThe helium cluster of size " +
				std::to_string(i) +
				"is not present in the network, "
				"cannot use the advection option!");
		}

		auto cluster = psiNetwork->getClusterCommon(clusterId);

		// Get its diffusion factor
		double diffFactor = cluster.getDiffusionFactor();

		// Don't do anything if the diffusion factor is 0.0
		if (util::equal(diffFactor, 0.0))
			continue;

		// Switch on the size to get the sink strength (in eV.nm3)
		double sinkStrength = 0.0;
		switch (i) {
		case 1:
			sinkStrength = 0.54e-3;
			break;
		case 2:
			sinkStrength = 1.01e-3;
			break;
		case 3:
			sinkStrength = 3.03e-3;
			break;
		case 4:
			sinkStrength = 3.93e-3;
			break;
		case 5:
			sinkStrength = 7.24e-3;
			break;
		case 6:
			sinkStrength = 10.82e-3;
			break;
		case 7:
			sinkStrength = 19.26e-3;
			break;
		}

		// If the sink strength is still 0.0, this cluster is not advecting
		if (util::equal(sinkStrength, 0.0))
			continue;

		// Add it to our collection of advecting clusters.
		advectingClusters.emplace_back(clusterId);

		// Add the sink strength to the vector
		sinkStrengthVector.push_back(sinkStrength);

		// Add Jacobian entry for this cluster
		idPairs.push_back({clusterId, clusterId});
	}

	this->syncAdvectingClusters(network);
	this->syncSinkStrengths();
}

void
XGBAdvectionHandler::computeAdvection(network::IReactionNetwork& network,
	const plsm::SpaceVector<double, 3>& pos, const StencilConcArray& concVector,
	Kokkos::View<double*> updatedConcOffset, double hxLeft, double hxRight,
	int ix, double hy, int iy, double hz, int iz) const
{
	// Consider each advecting cluster.
	// TODO Maintaining a separate index assumes that advectingClusters is
	// visited in same order as advectionGrid array for given point
	// and the sinkStrengthVector.
	// Currently true with C++11, but we'd like to be able to visit the
	// advecting clusters in any order (so that we can parallelize).
	// Maybe with a zip? or a std::transform?

	if (concVector.size() < 5) {
		throw std::runtime_error("Wrong sizse for 2D concentration stencil; "
								 "should be at least 5, got " +
			std::to_string(concVector.size()));
	}
	Kokkos::Array<Kokkos::View<const double*>, 5> concVec = {concVector[0],
		concVector[1], concVector[2], concVector[3], concVector[4]};

	auto clusterIds = this->advClusterIds;
	auto clusters = this->advClusters;
	auto sinkStrengths = this->advSinkStrengths;

	if (isPointOnSink(pos)) {
		Kokkos::parallel_for(
			clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
				auto currId = clusterIds[i];
				auto cluster = clusters[i];

				// If we are on the sink, the behavior is not the same
				// Both sides are giving their concentrations to the center
				double oldLeftConc = concVec[1][currId]; // left
				double oldRightConc = concVec[2][currId]; // right

				double conc = (3.0 * sinkStrengths[i] *
								  cluster.getDiffusionCoefficient(ix + 1)) *
					((oldLeftConc / pow(hxLeft, 5)) +
						(oldRightConc / pow(hxRight, 5))) /
					(kBoltzmann * cluster.getTemperature(ix + 1));

				// Update the concentration of the cluster
				updatedConcOffset[currId] += conc;
			});
	}
	// Here we are NOT on the GB sink
	else {
		auto location_ = location;
		Kokkos::parallel_for(
			clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
				// for (auto const& currId : advectingClusters) {
				auto currId = clusterIds[i];
				auto cluster = clusters[i];

				// Get the initial concentrations
				double oldConc = concVec[0][currId]; // middle
				double oldRightConc = concVec[2 * (pos[0] > location_) +
					1 * (pos[0] < location_)][currId]; // left or right

				// Get the a=d and b=d+h positions
				double a = fabs(location_ - pos[0]);
				double b = fabs(location_ - pos[0]) +
					hxRight * (pos[0] > location_) +
					hxLeft * (pos[0] < location_);

				// Compute the concentration as explained in the description of
				// the method
				double conc = (3.0 * sinkStrengths[i] *
								  cluster.getDiffusionCoefficient(ix + 1)) *
					((oldRightConc / (b * b * b * b)) -
						(oldConc / (a * a * a * a))) /
					(kBoltzmann * cluster.getTemperature(ix + 1) *
						(hxRight * (pos[0] > location_) +
							hxLeft * (pos[0] < location_)));

				// Update the concentration of the cluster
				updatedConcOffset[currId] += conc;
			});
	}
}

void
XGBAdvectionHandler::computePartialsForAdvection(
	network::IReactionNetwork& network, Kokkos::View<double*> val,
	const plsm::SpaceVector<double, 3>& pos, double hxLeft, double hxRight,
	int ix, double hy, int iy, double hz, int iz) const
{
	auto location_ = location;
	auto clusterIds = this->advClusterIds;
	auto clusters = this->advClusters;
	auto sinkStrengths = this->advSinkStrengths;
	auto dim = this->dimension;

	// If we are on the sink, the partial derivatives are not the same
	// Both sides are giving their concentrations to the center
	if (isPointOnSink(pos)) {
		Kokkos::parallel_for(
			clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
				auto temperature = clusters[i].getTemperature(ix + 1);
				double diffCoeff = clusters[i].getDiffusionCoefficient(ix + 1);
				double sinkStrength = sinkStrengths[i];

				// 1D case
				if (dim == 1) {
					val[i * 2] = (3.0 * sinkStrength * diffCoeff) /
						(kBoltzmann * temperature * pow(hxLeft, 5)); // left
					val[(i * 2) + 1] = (3.0 * sinkStrength * diffCoeff) /
						(kBoltzmann * temperature * pow(hxRight, 5)); // right
				}
			});
	}
	// Here we are NOT on the GB sink
	else {
		// Get the a=d and b=d+h positions
		double a = fabs(location - pos[0]);
		double b = fabs(location - pos[0]) + hxRight * (pos[0] > location) +
			hxLeft * (pos[0] < location);
		Kokkos::parallel_for(
			clusterIds.size(), KOKKOS_LAMBDA(IdType i) {
				auto temperature = clusters[i].getTemperature(ix + 1);
				double diffCoeff = clusters[i].getDiffusionCoefficient(ix + 1);
				double sinkStrength = sinkStrengths[i];

				// Compute the partial derivatives for advection of this cluster
				// as explained in the description of this method
				val[i * 2] = -(3.0 * sinkStrength * diffCoeff) /
					(kBoltzmann * temperature * (a * a * a * a) *
						(hxRight * (pos[0] > location_) +
							hxLeft * (pos[0] < location_))); // middle
				val[(i * 2) + 1] = (3.0 * sinkStrength * diffCoeff) /
					(kBoltzmann * temperature * (b * b * b * b) *
						(hxRight * (pos[0] > location_) +
							hxLeft * (pos[0] < location_))); // left or right
			});
	}
}

std::array<int, 3>
XGBAdvectionHandler::getStencilForAdvection(
	const plsm::SpaceVector<double, 3>& pos) const
{
	// The first index is positive by convention if we are on the sink
	if (isPointOnSink(pos))
		return {1, 0, 0};
	// The first index is positive if pos[0] > location
	// negative if pos[0] < location
	return {(pos[0] > location) - (pos[0] < location), 0, 0};
}

} /* end namespace advection */
} /* end namespace core */
} /* end namespace xolotl */
