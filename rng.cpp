#include "rng.h"
#include <omp.h>
#include <cstdint>

static thread_local std::mt19937 thread_generator;
static thread_local bool is_seeded = false;

void seed_thread_rng(unsigned int base_seed) {
    std::random_device rd;
    // Mix hardware noise with thread ID to guarantee distinct seeds across threads
    unsigned int thread_id = static_cast<unsigned int>(omp_get_thread_num());
    unsigned int seed = rd() ^ (thread_id + 1013554900U); 
    
    thread_generator.seed(seed);
    is_seeded = true;
}

double get_uniform_random() {
    // Safety check: automatically seed if seed_thread_rng() wasn't called manually
    if (!is_seeded) {
        seed_thread_rng();
    }
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(thread_generator);
}

double get_normal_random(double mean, double sigma)
{
    thread_local std::normal_distribution<double> dist;
    return dist(thread_generator) * sigma + mean;
}