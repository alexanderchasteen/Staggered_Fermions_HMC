#ifndef RNG_H
#define RNG_H

#include <random>

// Declare thread_local extern variable (defined in rng.cpp)
extern thread_local std::mt19937 rng;

// Declare function (default argument belongs ONLY in the header)
void seed_thread_rng(unsigned int base_seed = 0);

#endif // RNG_H