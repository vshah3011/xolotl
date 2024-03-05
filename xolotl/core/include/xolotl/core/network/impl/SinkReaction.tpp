#pragma once

#include <plsm/EnumIndexed.h>

#include <xolotl/core/Constants.h>
#include <xolotl/util/MathUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
SinkReaction<TNetwork, TDerived>::computeRate(IndexType gridIndex, double time)
{
	auto cl = this->_clusterData->getCluster(_reactant);
	double dc = cl.getDiffusionCoefficient(gridIndex);

	double strength = this->asDerived()->getSinkBias() *
		this->asDerived()->getSinkStrength() * dc;
		
	//std::cout<<strength<<std::endl;	

	return strength;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
DisloSinkReaction<TNetwork, TDerived>::computeRate(
	IndexType gridIndex, double time)
{
	auto cl = this->_clusterData->getCluster(_reactant);
	double dc = cl.getDiffusionCoefficient(gridIndex);

	double strength = this->asDerived()->getSinkBias() *
		this->asDerived()->getSinkStrength() * dc;
   
    //std::cout<<strength<<std::endl;

	return strength;
}

template <typename TNetwork, typename TDerived>
KOKKOS_INLINE_FUNCTION
double
GBSinkReaction<TNetwork, TDerived>::computeRate(
	IndexType gridIndex, double time)
{
	auto cl = this->_clusterData->getCluster(_reactant);
	double dc = cl.getDiffusionCoefficient(gridIndex);

	double strength = this->asDerived()->getSinkBias() *
		this->asDerived()->getSinkStrength() * dc;

    //std::cout<<strength<<std::endl;
	return strength;
}
} // namespace network
} // namespace core
} // namespace xolotl
