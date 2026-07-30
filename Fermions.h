#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include <complex>
#include "Parameters.h"
#include "Lattice.h"




const double fermion_mass = 1.0;
const int Staggered_Fermion_field_indices = 2*3*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size; //3 color indices which are complex numbers (2 reals) and lattice volume
const int conjugate_field_array_size = Spatial_Size*Spatial_Size*Spatial_Size*temporal_size*4*18; //Same size as the gauge field 
const double epsilon;
const int number_of_steps_fermion=10;


// Field configuration architecture
int moveup_fermion(lattice_index& lattice_index_array, int link_direction);
int movedown_fermion(lattice_index& lattice_index_array, int link_direction);
std::vector<double> fermion_field; // 3 color indices which are complex numbers (2 reals) and lattice volume
using fermion_field_index_type = std::array<int, 6>; // 3 color indices which are complex numbers (2 reals) and lattice volume. Treat as Spatial_size^3*Temporal_size*3*2 array(6 indices)
int flat_index_fermion_field(const fermion_field_index_type& index);
fermion_field_index_type tensor_index_fermion_field(int flat_index);
void set_fermion_field_value(std::vector<double>& fermion_field, const fermion_field_index_type& index, const double& value);
double get_fermion_field_value(const std::vector<double>& fermion_field, const fermion_field_index_type& index);
complex get_fermion_field_value_at_color(const std::vector<double>& fermion_field, const lattice_index& index, int color_index);
Eigen::Vector3cd extract_color_vector(const std::vector<double>& fermion_field, const lattice_index& index);
void set_fermion_field_value_at_lattice_site(std::vector<double>& fermion_field, const lattice_index& index, const Eigen::Vector3cd & value);

// Field generation
std::complex<double> generateGaussianComplex();
void generate_chi_field(std::vector<double>& fermion_field);
std::vector<double> generate_phi_field(const Link_array& arr);
std::vector<double> generate_conjugate_field_configuration();

// Dirac operator
int staggered_phase_factor(const lattice_index& index, int direction);
Eigen::Vector3cd apply_dirac_operator_at_lattice_site(const std::vector<double>& fermion_field, const Link_array& arr,const lattice_index& index);
Eigen::Vector3cd apply_dirac_dagger_at_site(const std::vector<double>& fermion_field, const Link_array& arr, const lattice_index& index);

// SU3/su3 stuff
std::array<Eigen::Matrix3cd, 8> get_gell_mann_matrices();
double sinx_over_x_stable(double x);
SU3 numerically_stable_matrix_exponential(const SU3& Q); //for lie algebra elements su3



// Actual algorithm
Eigen::Matrix3cd proj_su3(const Eigen::Matrix3cd& M);
Eigen::Matrix3cd compute_gauge_force_at_link(const Link_array& arr, const link_index& idx, double beta);
Eigen::Matrix3cd compute_fermion_force_at_link(const std::vector<double>& X_field,const std::vector<double>& Y_field, const Link_array& arr,const link_index& idx);
double field_inner_product_real(const std::vector<double>& A,const std::vector<double>& B);
void apply_DDdagger(const std::vector<double>& in_field, const Link_array& arr, std::vector<double>& out_field);
std::vector<double> solve_CG_DDdagger(const std::vector<double>& phi_field, const Link_array& arr, double tol = 1e-10, int max_iter = 5000) ;
std::vector<double> solve_X_field(const std::vector<double>& phi_field,const Link_array& arr);
std::vector<double> solve_Y_field(const std::vector<double>& X_field, const Link_array& arr);
std::vector<double> compute_full_force_field(const std::vector<double>& phi_field,const Link_array& arr,double beta);
std::vector<double> compute_full_force_field_last_step(const std::vector<double>& X_field, const std::vector<double>& phi_field,const Link_array& arr,double beta);
double compute_Tr_P2(const std::vector<double>& P_field);
double compute_phi_dagger_DDdagger_inverse_phi(const std::vector<double>& phi_field, const Link_array& arr);
void apply_initial_step_to_conjugate_field(std::vector<double>& P_field, double beta,double epsilon,const std::vector<double>& phi_field,const Link_array& arr);
void apply_intermediate_update_to_gauge_field_and_conjugate_field(std::vector<double> & P_field, double beta,double epsilon,const std::vector<double>& phi_field,Link_array& arr);
std::vector<double> apply_final_step(std::vector<double> & P_field,double beta,double epsilon,const std::vector<double>& phi_field,Link_array& arr);
void Monte_carlo_step(
    double initial_Tr_P2,
    double initial_phi_field_action,
    const std::vector<double>& P_field_final,
    const std::vector<double>& phi_field,
    double gauge_field_initial_action, 
    Link_array& gauge_field_initial,
    Link_array& gauge_field_final,
    const std::vector<double>& X_field, 
    double beta);
void single_sweep_monte_carlo_update_WITH_FERMIONS(Link_array& gauge_field,double beta);
