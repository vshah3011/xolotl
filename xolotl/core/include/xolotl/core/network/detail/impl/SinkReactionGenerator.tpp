#pragma once

namespace xolotl
{
namespace core
{
namespace network
{
namespace detail
{
template <typename TBase>
SinkReactionGenerator<TBase>::SinkReactionGenerator(
	const NetworkType& network) :
	Superclass(network),
	_clusterSinkReactionCounts(
		"Sink Reaction Counts", Superclass::getNumberOfClusters())
{
}

template <typename TBase>
typename SinkReactionGenerator<TBase>::IndexType
SinkReactionGenerator<TBase>::getRowMapAndTotalReactionCount()
{
	_numPrecedingReactions = Superclass::getRowMapAndTotalReactionCount();
	_numSinkReactions = Kokkos::get_crs_row_map_from_counts(
		_sinkCrsRowMap, _clusterSinkReactionCounts);

	_sinkReactions =
		Kokkos::View<SinkReactionType*>("Sink Reactions", _numSinkReactions);

	return _numPrecedingReactions + _numSinkReactions;
}

template <typename TBase>
void
SinkReactionGenerator<TBase>::setupCrsClusterSetSubView()
{
	Superclass::setupCrsClusterSetSubView();
	_sinkCrsClusterSets = this->getClusterSetSubView(std::make_pair(
		_numPrecedingReactions, _numPrecedingReactions + _numSinkReactions));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
SinkReactionGenerator<TBase>::addSinkReaction(
	Count, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	Kokkos::atomic_increment(&_clusterSinkReactionCounts(clusterSet.cluster0));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
SinkReactionGenerator<TBase>::addSinkReaction(
	Construct, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	auto id = _sinkCrsRowMap(clusterSet.cluster0);
	for (; !Kokkos::atomic_compare_exchange_strong(
			 &_sinkCrsClusterSets(id).cluster0, NetworkType::invalidIndex(),
			 clusterSet.cluster0);
		 ++id) { }
	_sinkCrsClusterSets(id) = clusterSet;
}

template <typename TBase>
DisloSinkReactionGenerator<TBase>::DisloSinkReactionGenerator(
	const NetworkType& network) :
	Superclass(network),
	_clusterDisloSinkReactionCounts(
		"Dislo Sink Reaction Counts", Superclass::getNumberOfClusters())
{
}

template <typename TBase>
typename DisloSinkReactionGenerator<TBase>::IndexType
DisloSinkReactionGenerator<TBase>::getRowMapAndTotalReactionCount()
{
	_numPrecedingReactions = Superclass::getRowMapAndTotalReactionCount();
	_numDisloSinkReactions = Kokkos::get_crs_row_map_from_counts(
		_sinkCrsRowMap, _clusterDisloSinkReactionCounts);

	_sinkReactions = Kokkos::View<DisloSinkReactionType*>(
		"Dislo Sink Reactions", _numDisloSinkReactions);

	return _numPrecedingReactions + _numDisloSinkReactions;
}

template <typename TBase>
void
DisloSinkReactionGenerator<TBase>::setupCrsClusterSetSubView()
{
	Superclass::setupCrsClusterSetSubView();
	_sinkCrsClusterSets =
		this->getClusterSetSubView(std::make_pair(_numPrecedingReactions,
			_numPrecedingReactions + _numDisloSinkReactions));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
DisloSinkReactionGenerator<TBase>::addDisloSinkReaction(
	Count, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	Kokkos::atomic_increment(
		&_clusterDisloSinkReactionCounts(clusterSet.cluster0));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
DisloSinkReactionGenerator<TBase>::addDisloSinkReaction(
	Construct, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	auto id = _sinkCrsRowMap(clusterSet.cluster0);
	for (; !Kokkos::atomic_compare_exchange_strong(
			 &_sinkCrsClusterSets(id).cluster0, NetworkType::invalidIndex(),
			 clusterSet.cluster0);
		 ++id) { }
	_sinkCrsClusterSets(id) = clusterSet;
}

template <typename TBase>
GBSinkReactionGenerator<TBase>::GBSinkReactionGenerator(
	const NetworkType& network) :
	Superclass(network),
	_clusterGBSinkReactionCounts(
		"GB Sink Reaction Counts", Superclass::getNumberOfClusters())
{
}

template <typename TBase>
typename GBSinkReactionGenerator<TBase>::IndexType
GBSinkReactionGenerator<TBase>::getRowMapAndTotalReactionCount()
{
	_numPrecedingReactions = Superclass::getRowMapAndTotalReactionCount();
	_numGBSinkReactions = Kokkos::get_crs_row_map_from_counts(
		_sinkCrsRowMap, _clusterGBSinkReactionCounts);

	_sinkReactions = Kokkos::View<GBSinkReactionType*>(
		"GB Sink Reactions", _numGBSinkReactions);

	return _numPrecedingReactions + _numGBSinkReactions;
}

template <typename TBase>
void
GBSinkReactionGenerator<TBase>::setupCrsClusterSetSubView()
{
	Superclass::setupCrsClusterSetSubView();
	_sinkCrsClusterSets = this->getClusterSetSubView(std::make_pair(
		_numPrecedingReactions, _numPrecedingReactions + _numGBSinkReactions));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
GBSinkReactionGenerator<TBase>::addGBSinkReaction(
	Count, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	Kokkos::atomic_increment(
		&_clusterGBSinkReactionCounts(clusterSet.cluster0));
}

template <typename TBase>
KOKKOS_INLINE_FUNCTION
void
GBSinkReactionGenerator<TBase>::addGBSinkReaction(
	Construct, const ClusterSet& clusterSet) const
{
	if (!this->_clusterData.enableSink())
		return;

	auto id = _sinkCrsRowMap(clusterSet.cluster0);
	for (; !Kokkos::atomic_compare_exchange_strong(
			 &_sinkCrsClusterSets(id).cluster0, NetworkType::invalidIndex(),
			 clusterSet.cluster0);
		 ++id) { }
	_sinkCrsClusterSets(id) = clusterSet;
}
} // namespace detail
} // namespace network
} // namespace core
} // namespace xolotl
