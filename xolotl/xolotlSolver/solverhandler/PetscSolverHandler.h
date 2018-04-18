#ifndef PETSCSOLVERHANDLER_H
#define PETSCSOLVERHANDLER_H

// Includes
#include "SolverHandler.h"

namespace xolotlSolver {

#ifndef CHECK_PETSC_ERROR
#define CHECK_PETSC_ERROR
/**
 * This operation checks a PETSc error code and throws an exception with given error message.
 *
 * @param errorCode The PETSc error code.
 * @param errMsg The error message in the thrown exception.
 */
inline void checkPetscError(PetscErrorCode errorCode, const char* errorMsg) {
	if (PetscUnlikely(errorCode))
		throw std::string(errorMsg);
}
#endif

/**
 * This class and its subclasses realize the ISolverHandler interface to solve the
 * advection-diffusion-reaction problem with the PETSc solvers from Argonne
 * National Laboratory.
 *
 * This class does NOT implement most of the methods that are needed by the
 * PetscSolver. Only subclasses of this class must be used by the PetscSolver.
 */
class PetscSolverHandler: public SolverHandler {
protected:

	/**
	 * The last temperature on the grid. In the future this will have to be an
	 * array or map, but for now the temperature is isotropic.
	 */
	double lastTemperature;

	/**
	 * A vector for holding the partial derivatives of one cluster. It is sized in
	 * the createSolverContext() operation.
	 *
	 * The vector is used for every cluster and immediately reset to zero before
	 * being used for the next. This allows the acquisition of the partial
	 * derivatives to take up minimal memory and require no additional dynamic
	 * allocations.
	 */
	std::vector<double> clusterPartials;

	/**
	 * A vector for holding the partial derivatives for one cluster in the order
	 * that PETSc expects. It is sized in the createSolverContext() operation.
	 *
	 * The vector is used for every cluster and immediately reset to zero before
	 * being used for the next. This allows the acquisition of the partial
	 * derivatives to take up minimal memory and require no additional dynamic
	 * allocations.
	 */
	std::vector<double> reactingPartialsForCluster;

    /**
     * Number of valid partial derivatives at each grid point.
     */
    PetscInt* reactionSize;

	/**
	 * A pointer to an array of the size dof * dof keeping the partial
	 * derivatives for all the reactions at one grid point.
	 *
	 * Its size must be initialized.
	 */
	PetscScalar *reactionVals;

	/**
	 * A pointer to an array of the size dof * dof keeping the indices for partial
	 * derivatives for all the reactions at one grid point.
	 *
	 * Its size must be initialized.
	 */
	PetscInt *reactionIndices;

public:

	/**
	 * Default constructor, deleted because we need to construct with objects.
	 */
	PetscSolverHandler() = delete;

	/**
	 * Construct a PetscSolverHandler.
	 *
	 * @param _network The reaction network to use.
	 */
	PetscSolverHandler(xolotlCore::IReactionNetwork& _network) :
			SolverHandler(_network), lastTemperature(0.0), 
            reactionSize(nullptr), reactionVals(nullptr), 
            reactionIndices(nullptr) {
	}

	//! The Destructor
	~PetscSolverHandler() {
		// Delete arrays
        delete[] reactionSize;
		delete[] reactionVals;
		delete[] reactionIndices;
	}

};
//end class PetscSolverHandler

} /* end namespace xolotlSolver */
#endif
