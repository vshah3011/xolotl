#pragma once

#include <xolotl/core/network/SinkReaction.h>
#include <xolotl/core/network/ZrTraits.h>

namespace xolotl
{
namespace core
{
namespace network
{
class ZrReactionNetwork;

class ZrProductionReaction :
	public ProductionReaction<ZrReactionNetwork, ZrProductionReaction>
{
public:
	using Superclass =
		ProductionReaction<ZrReactionNetwork, ZrProductionReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	getRateForProduction(IndexType gridIndex);
};

class ZrDissociationReaction :
	public DissociationReaction<ZrReactionNetwork, ZrDissociationReaction>
{
public:
	using Superclass =
		DissociationReaction<ZrReactionNetwork, ZrDissociationReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	getRateForProduction(IndexType gridIndex);

	KOKKOS_INLINE_FUNCTION
	double
	computeBindingEnergy();
};

class ZrSinkReaction : public SinkReaction<ZrReactionNetwork, ZrSinkReaction>
{
public:
	using Superclass = SinkReaction<ZrReactionNetwork, ZrSinkReaction>;

	using Superclass::Superclass;

	KOKKOS_INLINE_FUNCTION
	double
	computeRate(IndexType gridIndex);
};
} // namespace network
} // namespace core
} // namespace xolotl