#pragma once

#include <xolotl/core/network/detail/ReactionGenerator.h>

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
/**
 * @brief Inherits from ReactionGeneratorBase and additionally fills collections
 * sink reactions depending on the subpaving.
 *
 * @tparam TBase The templated base class.
 */
template <typename TBase>
class SinkReactionGenerator : public TBase
{
public:
	using Superclass = TBase;
	using NetworkType = typename TBase::NetworkType;
	using NetworkTraits = ReactionNetworkTraits<NetworkType>;
	using SinkReactionType = typename NetworkTraits::SinkReactionType;
	using IndexType = typename NetworkType::IndexType;
	using IndexView = typename Superclass::IndexView;
	using ClusterSetSubView = typename Superclass::ClusterSetSubView;
	using Count = typename Superclass::Count;
	using Construct = typename Superclass::Construct;

	SinkReactionGenerator(const NetworkType& network);

	IndexType
	getRowMapAndTotalReactionCount();

	void
	setupCrsClusterSetSubView();

	KOKKOS_INLINE_FUNCTION
	void
	addSinkReaction(Count, const ClusterSet& clusterSet) const;

	KOKKOS_INLINE_FUNCTION
	void
	addSinkReaction(Construct, const ClusterSet& clusterSet) const;

	Kokkos::View<SinkReactionType*>
	getSinkReactions() const
	{
		return _sinkReactions;
	}

	IndexType
	getNumberOfSinkReactions() const
	{
		return _sinkReactions.size();
	}

private:
	IndexView _clusterSinkReactionCounts;

	IndexType _numPrecedingReactions{};
	IndexType _numSinkReactions{};

	IndexView _sinkCrsRowMap;
	ClusterSetSubView _sinkCrsClusterSets;

	Kokkos::View<SinkReactionType*> _sinkReactions;
};

template <typename TNetwork, typename TReaction, typename TBase>
struct WrapTypeSpecificReactionGenerator<TNetwork, TReaction, TBase,
	std::enable_if_t<
		std::is_base_of_v<SinkReaction<TNetwork, TReaction>, TReaction>>>
{
	using Type = SinkReactionGenerator<TBase>;
};

/**
 * @brief Inherits from ReactionGeneratorBase and additionally fills collections
 * dislo sink reactions depending on the subpaving.
 *
 * @tparam TBase The templated base class.
 */
template <typename TBase>
class DisloSinkReactionGenerator : public TBase
{
public:
	using Superclass = TBase;
	using NetworkType = typename TBase::NetworkType;
	using NetworkTraits = ReactionNetworkTraits<NetworkType>;
	using DisloSinkReactionType = typename NetworkTraits::DisloSinkReactionType;
	using IndexType = typename NetworkType::IndexType;
	using IndexView = typename Superclass::IndexView;
	using ClusterSetSubView = typename Superclass::ClusterSetSubView;
	using Count = typename Superclass::Count;
	using Construct = typename Superclass::Construct;

	DisloSinkReactionGenerator(const NetworkType& network);

	IndexType
	getRowMapAndTotalReactionCount();

	void
	setupCrsClusterSetSubView();

	KOKKOS_INLINE_FUNCTION
	void
	addDisloSinkReaction(Count, const ClusterSet& clusterSet) const;

	KOKKOS_INLINE_FUNCTION
	void
	addDisloSinkReaction(Construct, const ClusterSet& clusterSet) const;

	Kokkos::View<DisloSinkReactionType*>
	getDisloSinkReactions() const
	{
		return _sinkReactions;
	}

	IndexType
	getNumberOfDisloSinkReactions() const
	{
		return _sinkReactions.size();
	}

private:
	IndexView _clusterDisloSinkReactionCounts;

	IndexType _numPrecedingReactions{};
	IndexType _numDisloSinkReactions{};

	IndexView _sinkCrsRowMap;
	ClusterSetSubView _sinkCrsClusterSets;

	Kokkos::View<DisloSinkReactionType*> _sinkReactions;
};

template <typename TNetwork, typename TReaction, typename TBase>
struct WrapTypeSpecificReactionGenerator<TNetwork, TReaction, TBase,
	std::enable_if_t<
		std::is_base_of_v<DisloSinkReaction<TNetwork, TReaction>, TReaction>>>
{
	using Type = DisloSinkReactionGenerator<TBase>;
};

/**
 * @brief Inherits from ReactionGeneratorBase and additionally fills collections
 * GB sink reactions depending on the subpaving.
 *
 * @tparam TBase The templated base class.
 */
template <typename TBase>
class GBSinkReactionGenerator : public TBase
{
public:
	using Superclass = TBase;
	using NetworkType = typename TBase::NetworkType;
	using NetworkTraits = ReactionNetworkTraits<NetworkType>;
	using GBSinkReactionType = typename NetworkTraits::GBSinkReactionType;
	using IndexType = typename NetworkType::IndexType;
	using IndexView = typename Superclass::IndexView;
	using ClusterSetSubView = typename Superclass::ClusterSetSubView;
	using Count = typename Superclass::Count;
	using Construct = typename Superclass::Construct;

	GBSinkReactionGenerator(const NetworkType& network);

	IndexType
	getRowMapAndTotalReactionCount();

	void
	setupCrsClusterSetSubView();

	KOKKOS_INLINE_FUNCTION
	void
	addGBSinkReaction(Count, const ClusterSet& clusterSet) const;

	KOKKOS_INLINE_FUNCTION
	void
	addGBSinkReaction(Construct, const ClusterSet& clusterSet) const;

	Kokkos::View<GBSinkReactionType*>
	getGBSinkReactions() const
	{
		return _sinkReactions;
	}

	IndexType
	getNumberOfGBSinkReactions() const
	{
		return _sinkReactions.size();
	}

private:
	IndexView _clusterGBSinkReactionCounts;

	IndexType _numPrecedingReactions{};
	IndexType _numGBSinkReactions{};

	IndexView _sinkCrsRowMap;
	ClusterSetSubView _sinkCrsClusterSets;

	Kokkos::View<GBSinkReactionType*> _sinkReactions;
};

template <typename TNetwork, typename TReaction, typename TBase>
struct WrapTypeSpecificReactionGenerator<TNetwork, TReaction, TBase,
	std::enable_if_t<
		std::is_base_of_v<GBSinkReaction<TNetwork, TReaction>, TReaction>>>
{
	using Type = GBSinkReactionGenerator<TBase>;
};
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
