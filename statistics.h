#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include "Parameters.h"

double mean(const std::array<double,autocorrelation_sweeps>& arr);
double variance(const std::array<double,autocorrelation_sweeps>& arr);
std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& arr);
int tau_int(const std::array<double,autocorrelation_sweeps>& arr);