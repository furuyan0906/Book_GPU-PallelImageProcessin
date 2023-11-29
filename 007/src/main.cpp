#include  <iostream>
#include  <cstdlib>
#include  "Particle2DSimulation.hpp"


int main(int argc, const char** argv)
{
    if (initParticle2DSimulation(argc, argv))
    {
        return EXIT_FAILURE;
    }

    startParticle2DSimulation();

	return EXIT_SUCCESS;
}

