#ifndef RNG_H
#define RNG_H

#include <random>

// Declare functions (default argument belongs ONLY in the header)
void seed_thread_rng(unsigned int base_seed = 0U);
double get_uniform_random();
double get_normal_random(double mean, double sigma);

#endif // RNG_H