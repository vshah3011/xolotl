#pragma once

#include <experimental/Reaction.h>

namespace xolotlCore
{
namespace experimental
{
template <typename TNetwork, typename TDerived>
class SinkReaction : public Reaction<TNetwork, TDerived>
{
    friend class Reaction<TNetwork, TDerived>;

public:
    using NetworkType = TNetwork;
    using Superclass = Reaction<TNetwork, TDerived>;
    using ClusterDataRef = typename Superclass::ClusterDataRef;
    using IndexType = typename Superclass::IndexType;
    using Connectivity = typename Superclass::Connectivity;
    using ConcentrationsView = typename Superclass::ConcentrationsView;
    using FluxesView = typename Superclass::FluxesView;
    using AmountType = typename Superclass::AmountType;

    SinkReaction() = default;

    KOKKOS_INLINE_FUNCTION
    SinkReaction(detail::ReactionDataRef reactionData,
            ClusterDataRef clusterData, IndexType reactionId,
            IndexType cluster0)
        :
        Superclass(reactionData, clusterData, reactionId),
        _reactant(cluster0)
    {
        this->initialize();
    }

    KOKKOS_INLINE_FUNCTION
    SinkReaction(detail::ReactionDataRef reactionData,
            ClusterDataRef clusterData, IndexType reactionId,
            const detail::ClusterSet& clusterSet)
        :
        SinkReaction(reactionData, clusterData, reactionId, clusterSet.cluster0)
    {
    }

private:
    KOKKOS_INLINE_FUNCTION
    void
    computeCoefficients()
    {
        // No coefs
    }

    KOKKOS_INLINE_FUNCTION
    double
    computeRate(IndexType gridIndex)
    {
        double r0 = this->_clusterData.getLatticeParameter() * 0.75 * sqrt(3.0);
        double rho = 0.0003;

        auto cl = this->_clusterData.getCluster(_reactant);
        double r = cl.getReactionRadius();
        double dc = cl.getDiffusionCoefficient(gridIndex);

        constexpr double pi = ::xolotlCore::pi;

        double strength = this->asDerived()->getSinkBias() * -4.0 * pi * rho
                / log(pi * rho * (r + r0)
                * (r + r0)) * dc;

        return strength;
    }

    KOKKOS_INLINE_FUNCTION
    void
    computeConnectivity(const Connectivity& connectivity)
    {
        // The reactant connects with the reactant
        this->addConnectivity(_reactant, _reactant, connectivity);
    }

    KOKKOS_INLINE_FUNCTION
    void
    computeFlux(ConcentrationsView concentrations, FluxesView fluxes,
        IndexType gridIndex)
    {
        Kokkos::atomic_sub(&fluxes(_reactant),
            concentrations(_reactant) * this->_rate(gridIndex));
    }

    KOKKOS_INLINE_FUNCTION
    void
    computePartialDerivatives(ConcentrationsView concentrations,
        Kokkos::View<double*> values, Connectivity connectivity,
        IndexType gridIndex)
    {
        Kokkos::atomic_sub(&values(connectivity(_reactant, _reactant)),
            this->_rate(gridIndex));
    }

    KOKKOS_INLINE_FUNCTION
    double
    computeLeftSideRate(ConcentrationsView concentrations, IndexType clusterId,
        IndexType gridIndex)
    {
        return 0.0;
    }

protected:
    IndexType _reactant;
    static constexpr auto invalidIndex = Superclass::invalidIndex;
};
}
}

#include <experimental/detail/SinkReactionGenerator.h>
