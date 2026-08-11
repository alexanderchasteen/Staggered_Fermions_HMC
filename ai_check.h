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
#include "Fermions.h"


SU3 random_su3_algebra_debug();
void check_SU3_matrix_debug(const SU3& U, const std::string& name);
double action_with_replaced_link(
    const Link_array& arr,
    const link_index& idx,
    const SU3& U_new,
    double beta);

double numerical_action_derivative(
    const Link_array& arr,
    const link_index& idx,
    const SU3& H,
    double beta,
    double eps);

double analytic_action_derivative(
    const Link_array& arr,
    const link_index& idx,
    const SU3& H,
    double beta);

void test_cold_start_staples( const Link_array& arr);
void test_cold_start_action(const Link_array& arr,double beta);
SU3 compute_plaquette_force_debug(
    const Link_array& arr,
    const link_index& idx,
    double beta);

SU3 compute_rectangle_force_debug(
    const Link_array& arr,
    const link_index& idx,
    double beta);

SU3 compute_test_force(
    const Link_array& arr,
    const link_index& idx,
    double beta,
    double c0,
    double c1);

void compare_force_to_numerical_derivative(
    const Link_array& arr,
    const link_index& idx,
    double beta,
    double c0,
    double c1,
    const std::string& label);

double test_action_with_coefficients(
    const Link_array& arr,
    const link_index& replaced_idx,
    const SU3& replaced_U,
    double beta,
    double c0,
    double c1);

void run_all_force_tests(const Link_array& arr, const double beta);
void test_rectangle_staple_types(const Link_array& arr);  //hot start this 

struct RectangleStapleDebug
{
    SU3 type1 = SU3::Zero();
    SU3 type2 = SU3::Zero();
    SU3 type3 = SU3::Zero();
    SU3 type4 = SU3::Zero();
    SU3 type5 = SU3::Zero();
    SU3 type6 = SU3::Zero();

    SU3 total = SU3::Zero();
};

RectangleStapleDebug
compute_rectangle_staples_debug(
    const Link_array& arr,
    const link_index& link_index_array);


SU3 rectangle_force_from_staple(
    const SU3& U,
    const SU3& A,
    double beta);

void test_one_rectangle_type(
    const Link_array& arr,
    const link_index& idx,
    const SU3& A,
    double beta,
    const std::string& name);

void test_all_rectangle_types( const Link_array& arr,double beta);

void print_displacement(
    const lattice_index& start,
    const lattice_index& current,
    const std::string& label);