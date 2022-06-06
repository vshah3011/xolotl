#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Regression

#include <boost/test/unit_test.hpp>

#include <xolotl/test/KokkosFixture.h>

BOOST_GLOBAL_FIXTURE(KokkosFixture);

#include <xolotl/test/MPITestUtils.h>
#include <xolotl/test/SystemTestCase.h>
using xolotl::test::getMPICommSize;
using xolotl::test::SystemTestCase;

BOOST_GLOBAL_FIXTURE(MPIFixture);

BOOST_AUTO_TEST_SUITE(System)

BOOST_AUTO_TEST_CASE(NE_0)
{
	if (getMPICommSize() > 1) {
		return;
	}
	SystemTestCase::copyFile("reaction_system_NE_0.dat");
	SystemTestCase{"system_NE_0"}.run();
}

BOOST_AUTO_TEST_CASE(PSI_1)
{
	if (getMPICommSize() > 20) {
		return;
	}
	// 1D + HeVI + advection + modifiedTM + attenuation + surface + reflective
	// bulk
	SystemTestCase{"system_PSI_1"}.run();
}

BOOST_AUTO_TEST_CASE(PSI_2)
{
	if (getMPICommSize() > 20) {
		return;
	}
	SystemTestCase::copyFile("tridyn_system_PSI_2.dat");
	// 1D + HeTVI + advection + modifiedTM + attenuation + surface + reflective
	SystemTestCase{"system_PSI_2"}.run();
}

BOOST_AUTO_TEST_CASE(PSI_3)
{
	if (getMPICommSize() > 10) {
		return;
	}
	// 1D + HeVI + pulsed flux + sink + I grouping + surface + reflective
	// bulk
	SystemTestCase{"system_PSI_3"}.run();
}

BOOST_AUTO_TEST_CASE(PSI_4)
{
	if (getMPICommSize() > 100) {
		return;
	}
	// 1D + He + heat
	SystemTestCase{"system_PSI_4"}.tolerance(1.0e-4).run();
}

BOOST_AUTO_TEST_CASE(Fe_1)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 0D
	SystemTestCase{"system_Fe_1", "bubble_36.dat"}.run();
}

BOOST_AUTO_TEST_CASE(Alloy_1)
{
	if (getMPICommSize() > 1) {
		return;
	}
	// 0D
	SystemTestCase{"system_Alloy_1", "Alloy.dat"}.run();
}

BOOST_AUTO_TEST_SUITE_END()
