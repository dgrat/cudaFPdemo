/*
#-------------------------------------------------------------------------------
# Copyright (c) 2012 Daniel <dgrat> Frenzel.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Lesser Public License v2.1
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
# 
# Contributors:
#     Daniel <dgrat> Frenzel - initial API and implementation
#-------------------------------------------------------------------------------
*/

#ifndef TRANSFERFUNCTIONS_H_
#define TRANSFERFUNCTIONS_H_

#ifndef SWIG
#include <cmath>
#include <stdio.h>
#include <string.h>
#endif

#define PI    3.14159265358979323846f 



typedef float (*pDistanceFu) (float, float);
typedef float (*pDecayFu) (float, float, float);


//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Distance functions for self organizing maps
 */
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_bubble_nhood (float dist, float sigmaT) {
	if(dist < sigmaT)
		return 1.f;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_gaussian_nhood (float dist, float sigmaT) {
	return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_cutgaussian_nhood (float dist, float sigmaT) {
	if(dist < sigmaT)
		return exp(-pow(dist, 2.f)/(2.f*pow(sigmaT, 2.f)));
	else return 0.f;
}


//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_mexican_nhood (float dist, float sigmaT) {
	return 	2.f/(sqrt(3.f * sigmaT) * pow(PI, 0.25f) ) * 
		(1.f-pow(dist, 2.f) / pow(sigmaT, 2.f) ) * 
		fcn_gaussian_nhood(dist, sigmaT);
}


//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_epanechicov_nhood (float dist, float sigmaT) {
	float fVal = 1 - pow(dist/sigmaT, 2.f);
	if(fVal > 0)
		return fVal;
	else return 0.f;
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_rad_decay (float sigma0, float T, float lambda) {
	return std::floor(sigma0*exp(-T/lambda) + 0.5f);
}

//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
	__host__ __device__
#endif
inline float
fcn_lrate_decay (float sigma0, float T, float lambda) {
	return sigma0*exp(-T/lambda);
}

/** 
 * @class DistFunction
 * @brief Represents a neighborhood and decay function.
 * Consists of a distance and a decay function. 
 * Normally just the neighborhood function is free to be changed. 
 */
typedef float (*pDistanceFu) (float, float);
typedef float (*pDecayFu) (float, float, float);

template <pDistanceFu Dist, pDecayFu Rad, pDecayFu LRate>
class DistFunction {	
public:
	DistFunction() {}
	DistFunction(const char *cstr) : name(cstr) {};
	
	const char *name;
	
	#ifdef __CUDACC__
		__host__ __device__
	#endif
	static float distance(float a, float b) { return Dist(a,b); };
	#ifdef __CUDACC__
		__host__ __device__
	#endif
	static float rad_decay(float a, float b, float c) { return Rad(a,b,c); };
	#ifdef __CUDACC__
		__host__ __device__
	#endif
	static float lrate_decay(float a, float b, float c) { return LRate(a,b,c); };
};

void test();

#endif /* TRANSFERFUNCTIONS_H_ */
