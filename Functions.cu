//#include <iostream>
#include "Functions.h"
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>


DistFunction<fcn_gaussian_nhood,fcn_rad_decay,fcn_lrate_decay> fcn_gaussian((char*)"gaussian");

typedef DistFunction<fcn_gaussian_nhood,fcn_rad_decay,fcn_lrate_decay> gaussian;

template <class F>
struct functor {
  float fCycle;
  float fCycles;

  functor(float cycle, float cycles) : fCycle(cycle), fCycles(cycles) {}

  __host__ __device__
  float operator()(float lrate) {
    return F::lrate_decay(lrate, fCycle, fCycles);
  }
};



void test() {
        unsigned int iWidth     = 4096;
        thrust::device_vector<float> dvLearningRate(iWidth, 0.f);
        thrust::device_vector<float> dvLRate(iWidth, 0.f);
	
	thrust::transform( dvLRate.begin(),
	dvLRate.end(),
	dvLearningRate.begin(),
	functor<gaussian>(1, 100) );
}

