#pragma once

namespace xolotlCore
{
namespace experimental
{
template <typename TImpl>
ReactionNetwork<TImpl>::ReactionNetwork(Subpaving&& subpaving,
        std::size_t gridSize)
    :
    _subpaving(std::move(subpaving)),
    _temperature("Temperature", gridSize),
    _numClusters(_subpaving.getNumberOfTiles(plsm::onDevice)),
    _reactionRadius("Reaction Radius", _numClusters),
    _diffusionCoefficient("Diffusion Coefficient", _numClusters, gridSize),
    _formationEnergy("Formation Energy", _numClusters)
{
    defineMomentIds();
    defineReactions();
}

template <typename TImpl>
void
ReactionNetwork<TImpl>::defineMomentIds()
{
    constexpr auto invalid = plsm::invalid<std::size_t>;

    _subpaving.syncAll(plsm::onHost);
    auto tiles = _subpaving.getTiles(plsm::onHost);
    auto nClusters = _subpaving.getNumberOfTiles(plsm::onHost);

    //FIXME: _momentIds data lives on the device this way
    _momentIds = Kokkos::View<std::size_t*[4]>("Moment Ids", nClusters);

    auto current = nClusters;
    for (std::size_t c = 0; c < nClusters; ++c) {
        const auto& reg = tiles(c).getRegion();
        for (auto k : getSpeciesRangeNoI()) {
            if (reg[k].length() == 1) {
                _momentIds(c, k()) = invalid;
            }
            else {
                _momentIds(c, k()) = current;
                ++current;
            }
        }
    }
}

template <typename TImpl>
void
ReactionNetwork<TImpl>::defineReactions()
{
    //TODO
}
}
}