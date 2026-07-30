#include "rng.h"
#include <omp.h>
#include <cstdint>

// Definition must match thread_local
thread_local std::mt19937 rng;

// Definition (no 'inline', no '= 0' default value here)
void seed_thread_rng(unsigned int base_seed) {
    std::random_device rd;
    int thread_id = omp_get_thread_num();
    
    // Mix hardware seed, thread ID, and base_seed
    uint64_t seed = static_cast<uint64_t>(rd()) ^ 
                    (static_cast<uint64_t>(thread_id + 1) * 0x9e3779b97f4a7c15ULL) ^ 
                    static_cast<uint64_t>(base_seed);
                    
    rng.seed(static_cast<unsigned int>(seed));
}