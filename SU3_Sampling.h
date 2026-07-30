#pragma once
#include <array>
#include <cmath>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <complex>


constexpr double pi=3.14159265358979323846;




// To generate an SU(2) matrix is the same as generating x_0,\vec{x} according to gatringer (Num. Sim. Pure Gauge Theory
// I will recreate this
using four_complex_array = std::array<std::complex<double>,4>;
using SU3 = Eigen::Matrix<std::complex<double>, 3, 3>;
using SU2 = Eigen::Matrix<std::complex<double>, 2, 2>;
using complex = std::complex<double>;
namespace Pauli {

inline const SU2 sigma_x = (SU2() << 0,1,1,0).finished();
inline const SU2 sigma_y = (SU2() << 0,-complex(0,1),complex(0,1),0).finished();
inline const SU2 sigma_z = (SU2() << 1,0,0,-1).finished();
inline const SU2 I = (SU2() << 1,0,0,1).finished();
}

// Main Function 
SU3 SU3_TypeR_generator(double a,double beta);
SU3 SU3_TypeS_generator(double a,double beta);
SU3 SU3_TypeT_generator(double a,double beta);
SU2 SU2_generator(double a,double beta);
SU2 Random_SU2_generator();



// helper functions 
double generate_x0(double a,double beta);
void sample_x0_check(double a, double beta, int j);
double get_norm_x(double x0);
std::array<double,3> generate_random_unit_3_vector();
std::array<double,3> generate_3_vector_norm_x(double x0);
four_complex_array SU_2_matrix_as_array(double a, double beta);
std::array<double,4> generate_random_unit_4_vector();
