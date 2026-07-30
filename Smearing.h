#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>

using matrix_3by3 = Eigen::Matrix<std::complex<double>, 3, 3>;
using matrix_4by4 = Eigen::Matrix<std::complex<double>, 4, 4>;


const matrix_4by4 rho = (1.0 / 10.0) * (matrix_4by4() << 0, 1, 1, 1,
                                                        1, 0, 1, 1, 
                                                        1, 1, 1, 0,
                                                        1, 1, 1, 0).finished();


