#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>

double alpha1 = 0.95;
double alpha2 = 0.76;
double alpha3 = 0.38;

const int smearing_size_1_index = 4*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size; //4 directions for mu and the lattice volume
const int smearing_size_2_index = 4*4*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size; //4 directions for mu, 4 directions for nu, and the lattice volume
const int smearing_size_3_index = 4*4*4*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size; 

int flat_index_smearing_3_tensor(const int mu, const int nu, const int rho, const int x, const int y, const int z, const int t);
std::array<int, 7> tensor_index_smearing_3_tensor(int flat_index);
int flat_index_smearing_2_tensor(const int mu, const int nu, const int x, const int y, const int z, const int t);
std::array<int, 6> tensor_index_smearing_2_tensor(int flat_index);
int flat_index_smearing_1_tensor(const int mu, const int x, const int y, const int z, const int t);     
std::array<int, 5> tensor_index_smearing_1_tensor(int flat_index) ;

std::vector<SU3> apply_step_1(const Link_array& U);
std::vector<SU3> apply_step_2(std::vector<SU3> Gamma1_mu_nu_rho, const Link_array& U_array);
std::vector<SU3> apply_step_3(std::vector<SU3> V1_mu_nu_rho);
std::vector<SU3> apply_step_4(std::vector<SU3> Gamma2_mu_nu, const Link_array& U_array);
std::vector<SU3> apply_step_5(std::vector<SU3> V2_mu_nu);
void apply_step_6(std::vector<SU3> Gamma_3_mu, Link_array& U_array);

void Apply_HEX_smearing(Link_array& U_array);
void Apply_4HEX_smearing(Link_array& U_array);
std::vector<Link_array> Apply_4HEX_smearing_with_history(const Link_array& U_thin);

SU3 projection(const SU3& matrix);