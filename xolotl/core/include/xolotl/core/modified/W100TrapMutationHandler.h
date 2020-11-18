#ifndef W100TRAPMUTATIONHANDLER_H
#define W100TRAPMUTATIONHANDLER_H

#include <xolotl/core/modified/TrapMutationHandler.h>

namespace xolotl
{
namespace core
{
namespace modified
{
/**
 * This class realizes the ITrapMutationHandler interface responsible for the
 * modified trap-mutation of small helium clusters close to the surface for a
 * (100) oriented tungsten material.
 */
class W100TrapMutationHandler : public TrapMutationHandler
{
private:
	/**
	 * \see TrapMutationHandler.h
	 */
	void
	initializeDepthSize(double temp)
	{
		// Switch values depending on the temperature
		if (temp < 1066.5) {
			depthVec = {-0.1, 0.5, 0.6, 0.6, 0.8, 0.8, 0.8};
			sizeVec = {0, 1, 1, 1, 1, 2, 2};

			// He2 desorpts with 4%
			desorp = Desorption(2, 0.04);
		}
		else {
			depthVec = {-0.1, 0.5, 0.6, 0.8, 0.6, 0.8, 0.8};
			sizeVec = {0, 1, 1, 1, 2, 2, 2};

			// He2 desorpts with 19%
			desorp = Desorption(2, 0.19);
		}

		return;
	}

public:
	/**
	 * The constructor
	 */
	W100TrapMutationHandler()
	{
	}

	/**
	 * The Destructor
	 */
	~W100TrapMutationHandler()
	{
	}
};
// end class W100TrapMutationHandler

} /* namespace modified */
} /* namespace core */
} /* namespace xolotl */

#endif
