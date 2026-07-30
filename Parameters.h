#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>


// Lattice Configurations 
constexpr int Spatial_Size = 16;
constexpr int temporal_size = 8;
constexpr int array_size = 4*Spatial_Size*Spatial_Size*Spatial_Size*18*temporal_size;
// 4n^4 total sites, 18 elements (9 real, 9 imaginary) of an SU(3) matrix to be stored at everysingle point
constexpr int Nplaq = Spatial_Size*Spatial_Size*Spatial_Size*temporal_size*6;
// Number of plaquettes is 6 times the volume 
const int maxdistance = std::floor(Spatial_Size/2)-1;





// Beta configurations
constexpr double dbeta=0.5;
constexpr double beta_min=0.25;
constexpr int CONFIG = 15;
constexpr double beta_max=beta_min+CONFIG*dbeta;
constexpr std::array<double, CONFIG> make_couplings() {
    std::array<double, CONFIG> arr{};

    for (int i = 0; i < CONFIG; ++i) {
        arr[i] = beta_min + i * dbeta;
    }

    return arr;
}

inline std::array<double, CONFIG> coupling_constants = make_couplings();

// Measurments
constexpr int thermal_sweeps=1000;  
constexpr int autocorrelation_sweeps=100;  
constexpr int measurment_sweeps=1000;  
constexpr int maxlag=10;

// Symanzik Improvement Coefficients 
constexpr double c_11 = 5.0/3.0;
constexpr double c_12 = -1.0/12.0;




