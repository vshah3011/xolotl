#pragma once

#include <xolotl/core/network/detail/impl/ConstantReactionGenerator.tpp>
#include <xolotl/core/network/detail/impl/SinkReactionGenerator.tpp>
#include <xolotl/core/network/impl/ReactionNetwork.tpp>
#include <xolotl/core/network/impl/ZrClusterGenerator.tpp>
#include <xolotl/core/network/impl/ZrReaction.tpp>

namespace xolotl
{
namespace core
{
namespace network
{
double
ZrReactionNetwork::checkLatticeParameter(double latticeParameter)
{
	if (latticeParameter <= 0.0) {
		return alphaZrLatticeConstant;
	}
	return latticeParameter;
}

double
ZrReactionNetwork::checkImpurityRadius(double impurityRadius)
{
	if (impurityRadius <= 0.0) {
		return alphaZrCoreRadius;
	}
	return impurityRadius;
}

ZrReactionNetwork::IndexType
ZrReactionNetwork::checkLargestClusterId()
{
	// Copy the cluster data for the parallel loop
	auto clData = _clusterData.d_view;
	using Reducer = Kokkos::MaxLoc<ZrReactionNetwork::AmountType,
		ZrReactionNetwork::IndexType>;
	Reducer::value_type maxLoc;
	Kokkos::parallel_reduce(
		_numClusters,
		KOKKOS_LAMBDA(IndexType i, Reducer::value_type & update) {
			const Region& clReg = clData().getCluster(i).getRegion();
			Composition hi = clReg.getUpperLimitPoint();

			// adding basal
			auto size = hi[Species::V] + hi[Species::I] + hi[Species::Basal];

			if (size > update.val) {
				update.val = size;
				update.loc = i;
			}
		},
		Reducer(maxLoc));

	return maxLoc.loc;
}

void
ZrReactionNetwork::setConstantRates(RateVector rates)
{
	auto dRateView = RatesView("dRates", rates.size(), rates[0].size());
	auto hRateView = create_mirror_view(dRateView);
	for (auto i = 0; i < rates.size(); i++)
		for (auto j = 0; j < rates[0].size(); j++) {
			hRateView(i, j) = rates[i][j];
		}
	deep_copy(dRateView, hRateView);

	_reactions.forEachOn<ZrConstantReaction>(
		"ReactionCollection::setConstantRates", DEVICE_LAMBDA(auto&& reaction) {
			reaction.setRate(dRateView);
			reaction.updateRates();
		});
}

void
ZrReactionNetwork::initializeExtraClusterData(const options::IOptions& options)
{
	this->_clusterData.h_view().extraData.initialize(
		this->_clusterData.h_view().numClusters,
		this->_clusterData.h_view().gridSize);
	this->copyClusterDataView();

	auto data = this->_clusterData.h_view();
	Kokkos::parallel_for(
		this->_numClusters, KOKKOS_LAMBDA(const IndexType i) {
			auto cluster = data.getCluster(i);
			const auto& reg = cluster.getRegion();
			Composition lo(reg.getOrigin());

			// Set the dislocation capture radii for vacancy a-loops (convert to
			// nm): First index in dislocation capture radius is for I capture;
			// second is for V capture
			if (lo.isOnAxis(Species::V)) {
				// Spontaneous radii:
				 //data.extraData.dislocationCaptureRadius(i, 0) = 3.05 * pow(lo[Species::V], 0.12) / 10;
				 //data.extraData.dislocationCaptureRadius(i, 1) = 0.39 * pow(lo[Species::V], 0.4) / 10;

				// Thermal radii:
                if (lo[Species::V] < 1000){
                    //data.extraData.dislocationCaptureRadius(i, 0) = 4.2 * pow(lo[Species::V], 0.09) / 10;
                    data.extraData.dislocationCaptureRadius(i, 0) = 2.8 * pow(lo[Species::V], 0.15) / 10;
                    data.extraData.dislocationCaptureRadius(i, 1) = 2.0 * pow(lo[Species::V], 0.3) / 10;
                }
                else {
                    //data.extraData.dislocationCaptureRadius(i, 0) = 0.902;
                    //data.extraData.dislocationCaptureRadius(i, 1) = 2.57;
                    data.extraData.dislocationCaptureRadius(i, 0) = 0.79;
                    data.extraData.dislocationCaptureRadius(i, 1) = 1.59;
                }
                //data.extraData.dislocationCaptureRadius(i, 0) = 0;
                //data.extraData.dislocationCaptureRadius(i, 1) = 0;

			}

			// adding basal
			// Set the dislocation capture radii for vacancy c-loops (convert to
			// nm): First index in dislocation capture radius is for I capture;
			// second is for V capture
			if (lo.isOnAxis(Species::Basal)) {
				// Spontaneous radii:
                //if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize) data.extraData.dislocationCaptureRadius(i, 0) = 3.9 * pow(lo[Species::Basal], 0.07) / 10;
				//if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize) data.extraData.dislocationCaptureRadius(i, 1) = 0.55 * pow(lo[Species::Basal], 0.33) / 10;

                // Thermal radii:
                if (lo[Species::Basal] < 1000){
                    if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize) data.extraData.dislocationCaptureRadius(i, 0) = 1.1;
                    else data.extraData.dislocationCaptureRadius(i, 0) = 5.2 * pow(lo[Species::Basal], 0.06) / 10;
                    data.extraData.dislocationCaptureRadius(i, 1) = 1.55 * pow(lo[Species::Basal], 0.28) / 10;
                }
                else{
                    data.extraData.dislocationCaptureRadius(i, 0) = 0.787;
                    data.extraData.dislocationCaptureRadius(i, 1) = 1.072;
                }

                // Turning off capture radii for FBP:
                //if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize) data.extraData.dislocationCaptureRadius(i, 0) = 0;
                //if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize) data.extraData.dislocationCaptureRadius(i, 1) = 0;
            }

			// Set the dislocation capture radii for interstitial a-loops
			// (convert to nm)
			else if (lo.isOnAxis(Species::I)) {
				// Spontaneous radii:
				 //data.extraData.dislocationCaptureRadius(i, 0) = 4.2 * pow(lo[Species::I], 0.05) / 10;
				 //data.extraData.dislocationCaptureRadius(i, 1) = 5.1 * pow(lo[Species::I], -0.01) / 10;

                // Thermal radii
                if (lo[Species::I] < 1000){
                    data.extraData.dislocationCaptureRadius(i, 0) = 4.5 * pow(lo[Species::I], 0.205) / 10;
                    //data.extraData.dislocationCaptureRadius(i, 0) = 5.0 * pow(lo[Species::I], 0.18) / 10;
                    data.extraData.dislocationCaptureRadius(i, 1) = 6.0 * pow(lo[Species::I], 0.08) / 10;
                }
                else{
                    //data.extraData.dislocationCaptureRadius(i, 0) = 2.58;
                    //data.extraData.dislocationCaptureRadius(i, 1) = 0.84;
                    data.extraData.dislocationCaptureRadius(i, 0) = 1.85;
                    data.extraData.dislocationCaptureRadius(i, 1) = 1.04;
                }
                //data.extraData.dislocationCaptureRadius(i, 0) = 0;
                //data.extraData.dislocationCaptureRadius(i, 1) = 0;

			}
		}); // Goes with parallel_for
}

void
ZrReactionNetwork::computeFluxesPreProcess(ConcentrationsView concentrations,
	FluxesView fluxes, IndexType gridIndex, double surfaceDepth, double spacing)
{
	/*
	if (this->_enableSink) {
		updateIntegratedConcentrations(concentrations, gridIndex);
	}
	*/
}

void
ZrReactionNetwork::computePartialsPreProcess(ConcentrationsView concentrations,
	Kokkos::View<double*> values, IndexType gridIndex, double surfaceDepth,
	double spacing)
{
	/*
	if (this->_enableSink) {
		updateIntegratedConcentrations(concentrations, gridIndex);
	}
	*/
}

void
ZrReactionNetwork::updateIntegratedConcentrations(
	ConcentrationsView concentrations, IndexType gridIndex)
{
	auto& data = this->_clusterData.h_view().extraData;
	// Get the V and I integrated concentration
	auto vInt = this->getTotalConcentration(concentrations, Species::V);
	auto iInt = this->getTotalConcentration(concentrations, Species::I);
	auto intConcs = create_mirror_view(data.integratedConcentrations);
	intConcs(0) = vInt;
	intConcs(1) = iInt;
	deep_copy(data.integratedConcentrations, intConcs);
}

namespace detail
{
template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
ZrReactionGenerator::operator()(IndexType i, IndexType j, TTag tag) const
{
	using Species = typename Network::Species;
	using Composition = typename Network::Composition;
	using AmountType = typename Network::AmountType;

	// Get the diffusion factors
	auto diffusionFactor = this->_clusterData.diffusionFactor;

	if (i == j) {
		if (diffusionFactor(i) != 0.0)
			addSinks(i, tag);
		this->addConstantReaction(tag, {i, Network::invalidIndex()});
	}

	// Add every possibility
	this->addConstantReaction(tag, {i, j});
	if (j != i)
		this->addConstantReaction(tag, {j, i});

	auto& subpaving = this->getSubpaving();
	auto previousIndex = subpaving.invalidIndex();

	// Check the diffusion factors
	if (diffusionFactor(i) == 0.0 && diffusionFactor(j) == 0.0) {
		return;
	}

	// Get the composition of each cluster
	const auto& cl1Reg = this->getCluster(i).getRegion();
	const auto& cl2Reg = this->getCluster(j).getRegion();
	Composition lo1 = cl1Reg.getOrigin();
	Composition hi1 = cl1Reg.getUpperLimitPoint();
	Composition lo2 = cl2Reg.getOrigin();
	Composition hi2 = cl2Reg.getUpperLimitPoint();

	// vac + vac = vac
	if (lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::V)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::V] + lo2[Species::V];
		auto hiSize = hi1[Species::V] + hi2[Species::V] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::V] = size;
			auto vProdId = subpaving.findTileId(comp);
			if (vProdId != subpaving.invalidIndex() &&
				vProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, vProdId});
				if (lo1[Species::V] == 1 || lo2[Species::V] == 1) {
					this->addDissociationReaction(tag, {vProdId, i, j});
				}
                previousIndex = vProdId;

                //Special case to allow size 9 basal clusters to dissociate into vacancies
                if (size == 9 && (lo1[Species::V] == 1 || lo2[Species::V] == 1)){
                    Composition comp = Composition::zero();
                    comp[Species::Basal] = size;
                    auto basalProdId = subpaving.findTileId(comp);
                    if (basalProdId != subpaving.invalidIndex() &&
                        basalProdId != previousIndex) {
                        //No production (vacancies do not accumulate into basal clusters)
                        this->addDissociationReaction(tag, {basalProdId, i, j});
                    }
                }
			}
		}

		return;
	}

	// Adding basal
	// Basal + Basal = Basal
	if (lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::Basal)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::Basal] + lo2[Species::Basal];
		auto hiSize = hi1[Species::Basal] + hi2[Species::Basal] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::Basal] = size;
			auto vProdId = subpaving.findTileId(comp);
			if (vProdId != subpaving.invalidIndex() &&
				vProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, vProdId});
				if (lo1[Species::Basal] == 1 || lo2[Species::Basal] == 1) {
					//this->addDissociationReaction(tag, {vProdId, i, j});
                    //Dissociating basal clusters produce V
				}
				previousIndex = vProdId;
			}
		}

		return;
	}

	// vac + Basal = Basal
	if ((lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::V)) ||
		(lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::Basal))) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::V] + lo2[Species::V] + lo1[Species::Basal] +
			lo2[Species::Basal]; // They can all be added because they are
								 // orthogonal
		auto hiSize = hi1[Species::V] + hi2[Species::V] + hi1[Species::Basal] +
			hi2[Species::Basal] - 4;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {

            if (size > 9){
                // Find Basal
                Composition comp = Composition::zero();
                comp[Species::Basal] = size;
                auto basalProdId = subpaving.findTileId(comp);
                if (basalProdId != subpaving.invalidIndex() &&
                    basalProdId != previousIndex) {
                    this->addProductionReaction(tag, {i, j, basalProdId});
                    if (lo1[Species::V] == 1 || lo2[Species::V] == 1) {
                        this->addDissociationReaction(tag, {basalProdId, i, j});
                    }
                    previousIndex = basalProdId;
                }
            }
            else {
                // Find V
                Composition comp = Composition::zero();
                comp[Species::V] = size;
                auto vProdId = subpaving.findTileId(comp);
                if (vProdId != subpaving.invalidIndex() &&
                    vProdId != previousIndex) {
                    this->addProductionReaction(tag, {i, j, vProdId});
                    previousIndex = vProdId;
                }
            }

		}

		return;
	}

	// int + Basal = Basal | vac | int | recombine
	if ((lo1.isOnAxis(Species::Basal) && lo2.isOnAxis(Species::I)) ||
		(lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::Basal))) {
		// Compute the largest possible product and the smallest one
		int largestProd = (int)hi1[Species::Basal] + (int)hi2[Species::Basal] -
			2 - (int)lo1[Species::I] - (int)lo2[Species::I];
		int smallestProd = (int)lo1[Species::Basal] + (int)lo2[Species::Basal] -
			(int)hi1[Species::I] - (int)hi2[Species::I] + 2;
		// Loop on the products
		for (int prodSize = smallestProd; prodSize <= largestProd; prodSize++) {
			// 4 cases
			if (prodSize > 9) {
				// Looking for Basal cluster
				Composition comp = Composition::zero();
				comp[Species::Basal] = prodSize;
				auto basalProdId = subpaving.findTileId(comp);
				if (basalProdId != subpaving.invalidIndex() &&
					basalProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, basalProdId});
					// No dissociation
                    previousIndex = basalProdId;
				}

			}
            else if (prodSize > 0) {
                // Looking for V cluster
                Composition comp = Composition::zero();
                comp[Species::V] = prodSize;
                auto vProdId = subpaving.findTileId(comp);
                if (vProdId != subpaving.invalidIndex() &&
                    vProdId != previousIndex) {
                    this->addProductionReaction(tag, {i, j, vProdId});
                    // No dissociation
                    previousIndex = vProdId;
                }
            }
			else if (prodSize < 0) {
				// Looking for I cluster
				Composition comp = Composition::zero();
				comp[Species::I] = -prodSize;
				auto iProdId = subpaving.findTileId(comp);
				if (iProdId != subpaving.invalidIndex() &&
					iProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, iProdId});
					// No dissociation
					previousIndex = iProdId;
				}
			}
			else {
				// No product
				this->addProductionReaction(tag, {i, j});
			}
		}

		return;
	}

	// vac + int = vac | int | recombine
	if (((lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::V)) ||
			(lo1.isOnAxis(Species::V) && lo2.isOnAxis(Species::I)))) {
		// Compute the largest possible product and the smallest one
		int largestProd = (int)hi1[Species::V] + (int)hi2[Species::V] - 2 -
			(int)lo1[Species::I] - (int)lo2[Species::I];
		int smallestProd = (int)lo1[Species::V] + (int)lo2[Species::V] -
			(int)hi1[Species::I] - (int)hi2[Species::I] + 2;
		// Loop on the products
		for (int prodSize = smallestProd; prodSize <= largestProd; prodSize++) {
			// 3 cases
			if (prodSize > 0) {
				// Looking for V cluster
				Composition comp = Composition::zero();
				comp[Species::V] = prodSize;
				auto vProdId = subpaving.findTileId(comp);
				if (vProdId != subpaving.invalidIndex() &&
					vProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, vProdId});
					// No dissociation
					previousIndex = vProdId;
				}
			}
			else if (prodSize < 0) {
				// Looking for I cluster
				Composition comp = Composition::zero();
				comp[Species::I] = -prodSize;
				auto iProdId = subpaving.findTileId(comp);
				if (iProdId != subpaving.invalidIndex() &&
					iProdId != previousIndex) {
					this->addProductionReaction(tag, {i, j, iProdId});
					// No dissociation
					previousIndex = iProdId;
				}
			}
			else {
				// No product
				this->addProductionReaction(tag, {i, j});
			}
		}

		return;
	}

	// int + int = int
	if (lo1.isOnAxis(Species::I) && lo2.isOnAxis(Species::I)) {
		// Compute the composition of the new cluster
		auto loSize = lo1[Species::I] + lo2[Species::I];
		auto hiSize = hi1[Species::I] + hi2[Species::I] - 2;
		// Loop on the possible sizes
		for (auto size = loSize; size <= hiSize; size++) {
			// Find the corresponding cluster
			Composition comp = Composition::zero();
			comp[Species::I] = size;
			auto iProdId = subpaving.findTileId(comp);
			if (iProdId != subpaving.invalidIndex() &&
				iProdId != previousIndex) {
				this->addProductionReaction(tag, {i, j, iProdId});
				if (lo1[Species::I] == 1 || lo2[Species::I] == 1) {
					this->addDissociationReaction(tag, {iProdId, i, j});
				}
				previousIndex = iProdId;
			}
		}

		return;
	}
}

template <typename TTag>
KOKKOS_INLINE_FUNCTION
void
ZrReactionGenerator::addSinks(IndexType i, TTag tag) const
{
	using Species = typename Network::Species;
	using Composition = typename Network::Composition;

	const auto& clReg = this->getCluster(i).getRegion();
	Composition lo = clReg.getOrigin();

	// I
	if (clReg.isSimplex() && lo.isOnAxis(Species::I)) {
		this->addSinkReaction(tag, {i, Network::invalidIndex()});
	}

	// V
	if (clReg.isSimplex() && lo.isOnAxis(Species::V)) {
		this->addSinkReaction(tag, {i, Network::invalidIndex()});
	}
}

inline ReactionCollection<ZrReactionGenerator::Network>
ZrReactionGenerator::getReactionCollection() const
{
	ReactionCollection<Network> ret(this->_clusterData.gridSize,
		this->getProductionReactions(), this->getDissociationReactions(),
		this->getSinkReactions(), this->getConstantReactions());
	return ret;
}
} // namespace detail

inline detail::ZrReactionGenerator
ZrReactionNetwork::getReactionGenerator() const noexcept
{
	return detail::ZrReactionGenerator{*this};
}

namespace detail
{
KOKKOS_INLINE_FUNCTION
void
ZrClusterUpdater::updateDiffusionCoefficient(
	const ClusterData& data, IndexType clusterId, IndexType gridIndex) const
{
	// I migration energies in eV

	constexpr Kokkos::Array<double, 6> iMigrationA = {
		0.0, 0.17, 0.23, 0.49, 0.75, 0.87};
	constexpr Kokkos::Array<double, 6> iMigrationC = {
		0.0, 0.30, 0.54, 0.93, 1.2, 1.6};
	// I diffusion factors in nm^2/s
	constexpr Kokkos::Array<double, 6> iDiffusionA = {
		0.0, 2.4e+11, 3.2e+11, 4.9e+12, 5.1e+13, 4.3e+13};
	constexpr Kokkos::Array<double, 6> iDiffusionC = {
		0.0, 6.8e+11, 2.6e+12, 6.8e+13, 4.2e+14, 5.5e+15};

	// V migration energies in eV (up to n = 6)
	constexpr Kokkos::Array<double, 7> vMigrationA = {
		0.0, 0.59, 0.58, 0.94, 0.16, 0.81, 0.25};
	constexpr Kokkos::Array<double, 7> vMigrationC = {
		0.0, 0.67, 0.41, 1.12, 0.58, 0.29, 0.18};
	// V diffusions factors in nm^2/s
	constexpr Kokkos::Array<double, 7> vDiffusionA = {
		0.0, 1.6e+12, 2.7e+12, 4.9e+13, 2.5e+10, 2e+13, 3.2e+10};
	constexpr Kokkos::Array<double, 7> vDiffusionC = {
		0.0, 2.2e+12, 2.3e+11, 1.27e+15, 4.5e+11, 5.7e+11, 9.1e+9};


    /*
    //Slower vacancies:
    constexpr Kokkos::Array<double, 6> iMigrationA = {
        0.0, 0.06, 0.23, 0.49, 0.75, 0.87};
    constexpr Kokkos::Array<double, 6> iMigrationC = {
        0.0, 0.15, 0.54, 0.93, 1.2, 1.6};
    // I diffusion factors in nm^2/s
    constexpr Kokkos::Array<double, 6> iDiffusionA = {
        0.0, 3.5e+10, 3.2e+11, 4.9e+12, 5.1e+13, 4.3e+13};
    constexpr Kokkos::Array<double, 6> iDiffusionC = {
        0.0, 4.7e+10, 2.6e+12, 6.8e+13, 4.2e+14, 5.5e+15};

    // V migration energies in eV (up to n = 6)
    constexpr Kokkos::Array<double, 7> vMigrationA = {
        0.0, 0.91, 0.58, 0.94, 0.16, 0.81, 0.25};
    constexpr Kokkos::Array<double, 7> vMigrationC = {
        0.0, 0.96, 0.41, 1.12, 0.58, 0.29, 0.18};
    // V diffusions factors in nm^2/s
    constexpr Kokkos::Array<double, 7> vDiffusionA = {
        0.0, 5.87e+12, 2.7e+12, 4.9e+13, 2.5e+10, 2e+13, 3.2e+10};
    constexpr Kokkos::Array<double, 7> vDiffusionC = {
        0.0, 8.2e+12, 2.3e+11, 1.27e+15, 4.5e+11, 5.7e+11, 9.1e+9};
    */

    // Literature values:
    /*
    // I migration energies in eV
    constexpr Kokkos::Array<double, 6> iMigrationA = {
    0.0, 0.17, 0.23, 0.49, 0.75, 0.87};
    constexpr Kokkos::Array<double, 6> iMigrationC = {
    0.0, 0.17, 0.54, 0.93, 1.2, 1.6};
    // I diffusion factors in nm^2/s
    constexpr Kokkos::Array<double, 6> iDiffusionA = {
    0.0, 3.23e+9, 0, 0, 0, 0};
    constexpr Kokkos::Array<double, 6> iDiffusionC = {
    0.0, 3.23e+9, 0, 0, 0, 0};

    // V migration energies in eV (up to n = 6)
    constexpr Kokkos::Array<double, 7> vMigrationA = {
    0.0, 0.59, 0.58, 0.94, 0.16, 0.81, 0.25};
    constexpr Kokkos::Array<double, 7> vMigrationC = {
    0.0, 0.59, 0.41, 1.12, 0.58, 0.29, 0.18};
    // V diffusions factors in nm^2/s
    constexpr Kokkos::Array<double, 7> vDiffusionA = {
    0.0, 443, 2.7e+12, 4.9e+13, 2.5e+10, 2e+13, 3.2e+10};
    constexpr Kokkos::Array<double, 7> vDiffusionC = {
    0.0, 443, 2.3e+11, 1.27e+15, 4.5e+11, 5.7e+11, 9.1e+9};
    */

	// 3D diffuser case
	if (data.migrationEnergy(clusterId) < 0.0) {
		double kernel = -1.0 / (kBoltzmann * data.temperature(gridIndex));
		const auto& clReg = data.getCluster(clusterId).getRegion();
		Network::Composition lo = clReg.getOrigin();
		using Species = Network::Species;

		if (lo.isOnAxis(Species::I)) {
			// Compute each contribution
			double Da = iDiffusionA[lo[Species::I]] *
				exp(iMigrationA[lo[Species::I]] * kernel);
			double Dc = iDiffusionC[lo[Species::I]] *
				exp(iMigrationC[lo[Species::I]] * kernel);

			// Compute the mean
			data.diffusionCoefficient(clusterId, gridIndex) =
				pow(Da * Da * Dc, 1.0 / 3.0);

			// Compute the anisotropy factor
			data.extraData.anisotropyRatio(clusterId, gridIndex) =
				pow(Dc / Da, 1.0 / 6.0);
			return;
		}

		if (lo.isOnAxis(Species::V)) {
			// Compute each contribution
			double Da = vDiffusionA[lo[Species::V]] *
				exp(vMigrationA[lo[Species::V]] * kernel);
			double Dc = vDiffusionC[lo[Species::V]] *
				exp(vMigrationC[lo[Species::V]] * kernel);

			// Compute the mean
			data.diffusionCoefficient(clusterId, gridIndex) =
				pow(Da * Da * Dc, 1.0 / 3.0);

			// Compute the anisotropy factor
			data.extraData.anisotropyRatio(clusterId, gridIndex) =
				pow(Dc / Da, 1.0 / 6.0);

			return;
		}
	}

	// 1D diffuser case
	data.diffusionCoefficient(clusterId, gridIndex) =
		data.diffusionFactor(clusterId) *
		exp(-data.migrationEnergy(clusterId) /
			(kBoltzmann * data.temperature(gridIndex)));
}
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
