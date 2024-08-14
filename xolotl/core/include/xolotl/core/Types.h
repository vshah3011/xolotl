#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

#include <xolotl/config.h>
#include <xolotl/util/Array.h>

namespace xolotl
{
namespace core
{
using RowColPair = util::Array<IdType, 2>;

using StencilConcArray = Kokkos::Array<Kokkos::View<const double*>,
	KOKKOS_INVALID_INDEX, Kokkos::Array<>::contiguous>;
} // namespace core

using DefaultMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
template <typename T>
using PetscOffsetView =
	Kokkos::Experimental::OffsetView<T, Kokkos::LayoutRight, DefaultMemSpace>;

} // namespace xolotl
