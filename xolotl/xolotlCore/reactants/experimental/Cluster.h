#pragma once

#include <experimental/ClusterData.h>

namespace xolotlCore
{
namespace experimental
{
template <typename TDerived>
class ClusterBase
{
public:
    KOKKOS_INLINE_FUNCTION
    ClusterBase(std::size_t id)
        :
        _id(id)
    {
    }

    KOKKOS_INLINE_FUNCTION
    double
    getReactionRadius()
    {
        return asDerived()->_data.reactionRadius(_id);
    }

    KOKKOS_INLINE_FUNCTION
    double
    getFormationEnergy()
    {
        return asDerived()->_data.formationEnergy(_id);
    }

    KOKKOS_INLINE_FUNCTION
    double
    getTemperature(std::size_t gridIndex)
    {
        return asDerived()->_data.temperature(gridIndex);
    }

    KOKKOS_INLINE_FUNCTION
    double
    getDiffusionCoefficient(std::size_t gridIndex)
    {
        return asDerived()->_data.diffusionCoefficient(_id, gridIndex);
    }

    KOKKOS_INLINE_FUNCTION
    double
    getDiffusionFactor()
    {
        return asDerived()->_data.diffusionFactor(_id);
    }

    KOKKOS_INLINE_FUNCTION
    std::size_t
    getId() const noexcept
    {
        return _id;
    }

private:
    KOKKOS_INLINE_FUNCTION
    TDerived*
    asDerived() noexcept
    {
        return static_cast<TDerived*>(this);
    }

private:
    std::size_t _id {plsm::invalid<std::size_t>};
};

template <typename PlsmContext>
class ClusterCommon : public ClusterBase<ClusterCommon<PlsmContext>>
{
    friend class ClusterBase<ClusterCommon<PlsmContext>>;

public:
    using Superclass = ClusterBase<ClusterCommon<PlsmContext>>;
    using ClusterData = detail::ClusterDataCommon<PlsmContext>;
    using ClusterDataRef = detail::ClusterDataCommonRef<PlsmContext>;

    ClusterCommon() = delete;

    KOKKOS_INLINE_FUNCTION
    ClusterCommon(const ClusterData& data, std::size_t id)
        :
        Superclass(id),
        _data{data}
    {
    }

    KOKKOS_INLINE_FUNCTION
    ClusterCommon(const ClusterDataRef& data, std::size_t id)
        :
        Superclass(id),
        _data{data}
    {
    }

private:
    ClusterDataRef _data;
};

template <typename TNetwork, typename PlsmContext>
class Cluster : public ClusterBase<Cluster<TNetwork, PlsmContext>>
{
    friend class ClusterBase<Cluster>;

    using Types = detail::ReactionNetworkTypes<TNetwork>;

public:
    using Superclass = ClusterBase<Cluster<TNetwork, PlsmContext>>;
    using Subpaving = typename Types::Subpaving;
    using Region = typename Subpaving::RegionType;
    using ClusterData = detail::ClusterData<TNetwork, PlsmContext>;
    using ClusterDataRef = detail::ClusterDataRef<TNetwork, PlsmContext>;

    Cluster() = delete;

    KOKKOS_INLINE_FUNCTION
    Cluster(const ClusterData& data, std::size_t id)
        :
        Superclass(id),
        _data{data}
    {
    }

    KOKKOS_INLINE_FUNCTION
    Cluster(const ClusterDataRef& data, std::size_t id)
        :
        Superclass(id),
        _data{data}
    {
    }

    KOKKOS_INLINE_FUNCTION
    Region
    getRegion() const
    {
        return _data.tiles(this->getId()).getRegion();
    }

    KOKKOS_INLINE_FUNCTION
    decltype(auto)
    getMomentIds()
    {
        return Kokkos::subview(_data.momentIds, this->getId(), Kokkos::ALL);
    }

private:
    ClusterDataRef _data;
};
}
}
