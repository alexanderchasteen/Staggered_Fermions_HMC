#include "Lattice.h"
#include "Parameters.h"
#include "Fermions.h"
#include "rng.h"
#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "ai_check.h"
#include "Fermions.h"

// ============================================================================
// DEBUGGING / CONSISTENCY TESTS FOR SYMANZIK GAUGE ACTION
// ============================================================================
//
// These tests assume:
//     c_11 = 5.0 / 3.0
//     c_12 = -1.0 / 12.0
//
// and the convention
//
//     U -> exp(i * epsilon * H) U
//
// where H is Hermitian and traceless.
//
// With T_a = lambda_a / 2 and Tr(T_a T_b) = delta_ab / 2,
// the matrix force satisfies:
//
//     dS/depsilon = 2 Tr(F H)
//
// ============================================================================


// -----------------------------------------------------------------------------
// Generate a random Hermitian traceless SU(3) algebra element H
// -----------------------------------------------------------------------------
SU3 random_su3_algebra_debug()
{
    static const auto gell_mann = get_gell_mann_matrices();

    SU3 H = SU3::Zero();

    for (int a = 0; a < 8; ++a)
    {
        double r = get_normal_random(0.0, 1.0);

        // T_a = lambda_a / 2
        H += 0.5 * r * gell_mann[a];
    }

    // Remove any numerical trace contamination
    H -= (H.trace() / 3.0) * SU3::Identity();

    // Hermitize numerically
    H = 0.5 * (H + H.adjoint());

    return H;
}


// -----------------------------------------------------------------------------
// Check that a matrix is approximately SU(3)
// -----------------------------------------------------------------------------
void check_SU3_matrix_debug(const SU3& U, const std::string& name)
{
    SU3 unitary_error =
        U * U.adjoint() - SU3::Identity();

    double unitarity_error = unitary_error.norm();

    double determinant_error =
        std::abs(U.determinant() - complex(1.0, 0.0));

    std::cout << "\n" << name << "\n";
    std::cout << "  unitarity error = "
              << unitarity_error << "\n";
    std::cout << "  determinant error = "
              << determinant_error << "\n";
}


// -----------------------------------------------------------------------------
// Replace ONE link temporarily.
//
// We cannot directly modify a const Link_array, so this helper copies the
// lattice, changes one link, and evaluates the action.
// -----------------------------------------------------------------------------
double action_with_replaced_link(
    const Link_array& arr,
    const link_index& idx,
    const SU3& U_new,
    double beta)
{
    Link_array test_arr = arr;

    set_link_SU3(test_arr, idx, U_new);

    return compute_action(test_arr, beta);
}


// -----------------------------------------------------------------------------
// Numerical derivative of the action under:
//
//     U -> exp(i eps H) U
//
// Central difference:
//     dS/deps = [S(+eps)-S(-eps)]/(2 eps)
// -----------------------------------------------------------------------------
double numerical_action_derivative(
    const Link_array& arr,
    const link_index& idx,
    const SU3& H,
    double beta,
    double eps)
{
    SU3 U = get_SU3_at_link(arr, idx);

    SU3 U_plus =
        numerically_stable_matrix_exponential(
            eps * H
        ) * U;

    SU3 U_minus =
        numerically_stable_matrix_exponential(
            -1 * eps * H
        ) * U;

    double S_plus =
        action_with_replaced_link(
            arr, idx, U_plus, beta);

    double S_minus =
        action_with_replaced_link(
            arr, idx, U_minus, beta);

    return (S_plus - S_minus) / (2.0 * eps);
}


// -----------------------------------------------------------------------------
// Analytic derivative predicted by the gauge force:
//
//     dS/deps = 2 Tr(F H)
//
// -----------------------------------------------------------------------------
double analytic_action_derivative(
    const Link_array& arr,
    const link_index& idx,
    const SU3& H,
    double beta)
{
    SU3 F =
        compute_gauge_force_at_link(
            arr, idx, beta);

    complex value =
        (F * H).trace();

    return 2.0 * value.real();
}

// ============================================================================
// TEST 1: COLD-START STAPLE COUNTING
// ============================================================================

void test_cold_start_staples( const Link_array& arr)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "TEST 1: COLD-START STAPLE COUNTING\n";
    std::cout << "============================================================\n";

    link_index idx = {
        0, 0, 0, 0, 0
    };

    SU3 U =
        get_SU3_at_link(arr, idx);

    SU3 A1 =
        compute_staple_sum_at_link(arr, idx);

    SU3 A2 =
        compute_rectangle_staple_sum_at_link(arr, idx);

    SU3 expected_A1 =
        6.0 * SU3::Identity();

    SU3 expected_A2 =
        18.0 * SU3::Identity();

    std::cout << "U =\n" << U << "\n\n";

    std::cout << "A1 =\n" << A1 << "\n\n";
    std::cout << "Expected A1 = 6 I\n\n";

    std::cout << "A2 =\n" << A2 << "\n\n";
    std::cout << "Expected A2 = 18 I\n\n";

    std::cout << "A1 error = "
              << (A1 - expected_A1).norm()
              << "\n";

    std::cout << "A2 error = "
              << (A2 - expected_A2).norm()
              << "\n";

    if ((A1 - expected_A1).norm() < 1e-12)
        std::cout << "PLAQUETTE STAPLE: PASS\n";
    else
        std::cout << "PLAQUETTE STAPLE: FAIL\n";

    if ((A2 - expected_A2).norm() < 1e-12)
        std::cout << "RECTANGLE STAPLE: PASS\n";
    else
        std::cout << "RECTANGLE STAPLE: FAIL\n";
}



void test_cold_start_action(const Link_array& arr, double beta)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "TEST 2: COLD-START TRACE-ONLY ACTION\n";
    std::cout << "============================================================\n";

    const double volume =
        static_cast<double>(
            Spatial_Size *
            Spatial_Size *
            Spatial_Size *
            temporal_size
        );

    const double N_plaq =
        6.0 * volume;

    const double N_rect =
        12.0 * volume;

    const double c_11 =
        5.0 / 3.0;

    const double c_12 =
        -1.0 / 12.0;

    const double expected_action =
        -beta *
        (
            c_11 * N_plaq
            +
            c_12 * N_rect
        );

    const double actual_action =
        compute_action(arr, beta);

    const double error =
        std::abs(actual_action - expected_action);

    std::cout
        << "Spatial_Size = "
        << Spatial_Size
        << "\n";

    std::cout
        << "temporal_size = "
        << temporal_size
        << "\n";

    std::cout
        << "Volume = "
        << volume
        << "\n";

    std::cout
        << "N_plaq = "
        << N_plaq
        << "\n";

    std::cout
        << "N_rect = "
        << N_rect
        << "\n";

    std::cout
        << "beta = "
        << beta
        << "\n";

    std::cout
        << "c_11 = "
        << c_11
        << "\n";

    std::cout
        << "c_12 = "
        << c_12
        << "\n";

    std::cout
        << "\nExpected cold action = "
        << expected_action
        << "\n";

    std::cout
        << "Actual cold action   = "
        << actual_action
        << "\n";

    std::cout
        << "Absolute error       = "
        << error
        << "\n";

    if (error < 1e-10)
    {
        std::cout
            << "\nCOLD TRACE-ONLY ACTION: PASS\n";
    }
    else
    {
        std::cout
            << "\nCOLD TRACE-ONLY ACTION: FAIL\n";
    }
}


SU3 compute_plaquette_force_debug(
    const Link_array& arr,
    const link_index& idx,
    double beta)
{
    SU3 U =
        get_SU3_at_link(arr, idx);

    SU3 A1 =
        compute_staple_sum_at_link(arr, idx);

    SU3 Q = U * A1;

    SU3 Q_traceless =
        Q - (Q.trace() / 3.0) * SU3::Identity();

    SU3 F =
        -complex(0.0, 1.0)
        * (beta / 12.0)
        * (Q_traceless - Q_traceless.adjoint());

    return F;
}


SU3 compute_rectangle_force_debug(
    const Link_array& arr,
    const link_index& idx,
    double beta)
{
    SU3 U =
        get_SU3_at_link(arr, idx);

    SU3 A2 =
        compute_rectangle_staple_sum_at_link(arr, idx);

    SU3 Q = U * A2;

    SU3 Q_traceless =
        Q - (Q.trace() / 3.0) * SU3::Identity();

    SU3 F =
        -complex(0.0, 1.0)
        * (beta / 12.0)
        * (c_12 / c_11)
        * (Q_traceless - Q_traceless.adjoint());

    return F;
}


SU3 compute_test_force(
    const Link_array& arr,
    const link_index& idx,
    double beta,
    double c0,
    double c1)
{
    SU3 U =
        get_SU3_at_link(arr, idx);

    SU3 A1 =
        compute_staple_sum_at_link(arr, idx);

    SU3 A2 =
        compute_rectangle_staple_sum_at_link(arr, idx);

    SU3 A =
        c0 * A1 + c1 * A2;

    SU3 Q =
        U * A;

    SU3 Q_traceless =
        Q - (Q.trace() / 3.0) * SU3::Identity();

    SU3 F =
        -complex(0.0, 1.0)
        * (beta / 12.0)
        * (Q_traceless - Q_traceless.adjoint());

    return F;
}


double test_action_with_coefficients(
    const Link_array& arr,
    const link_index& replaced_idx,
    const SU3& replaced_U,
    double beta,
    double c0,
    double c1)
{
    Link_array test_arr = arr;

    set_link_SU3(
        test_arr,
        replaced_idx,
        replaced_U
    );

    double volume =
        static_cast<double>(
            Spatial_Size *
            Spatial_Size *
            Spatial_Size *
            temporal_size
        );

    const double N_plaq = 6.0 * volume;
    const double N_rect = 12.0 * volume;

    double plaq_trace_sum = 0.0;
    double rect_trace_sum = 0.0;

    for (int i1 = 0; i1 < Spatial_Size; ++i1)
    {
        for (int i2 = 0; i2 < Spatial_Size; ++i2)
        {
            for (int i3 = 0; i3 < Spatial_Size; ++i3)
            {
                for (int i4 = 0; i4 < temporal_size; ++i4)
                {
                    for (int d = 0; d < 4; ++d)
                    {
                        link_index idx =
                            {i1, i2, i3, i4, d};

                        SU3 U =
                            get_SU3_at_link(
                                test_arr, idx);

                        SU3 A1 =
                            compute_staple_sum_at_link(
                                test_arr, idx);

                        SU3 A2 =
                            compute_rectangle_staple_sum_at_link(
                                test_arr, idx);

                        plaq_trace_sum +=
                            (U * A1).trace().real()
                            / 4.0;

                        rect_trace_sum +=
                            (U * A2).trace().real()
                            / 6.0;
                    }
                }
            }
        }
    }

    double S =
        beta *
        (
            c0 *
            (
                N_plaq
                - plaq_trace_sum / 3.0
            )
            +
            c1 *
            (
                N_rect
                - rect_trace_sum / 3.0
            )
        );

    return S;
}


void compare_force_to_numerical_derivative(
    const Link_array& arr,
    const link_index& idx,
    double beta,
    double c0,
    double c1,
    const std::string& label)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "FORCE TEST: " << label << "\n";
    std::cout << "============================================================\n";

    SU3 U =
        get_SU3_at_link(arr, idx);

    SU3 A1 =
        compute_staple_sum_at_link(arr, idx);

    SU3 A2 =
        compute_rectangle_staple_sum_at_link(arr, idx);

    SU3 A =
        c0 * A1 + c1 * A2;

    SU3 Q =
        U * A;

    SU3 Q_traceless =
        Q - (Q.trace() / 3.0) * SU3::Identity();

    SU3 F =
        -complex(0.0, 1.0)
        * (beta / 12.0)
        * (Q_traceless - Q_traceless.adjoint());

    std::cout << "Force matrix norm = "
              << F.norm() << "\n";

    std::vector<double> eps_list = {
        1e-2,
        5e-3,
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
        5e-6
    };

    std::cout << "\n";
    std::cout
        << "eps"
        << "\t\tNumerical"
        << "\tAnalytic"
        << "\tAbs Error"
        << "\tRatio"
        << "\n";

    double previous_error = 0.0;

    for (double eps : eps_list)
    {
        SU3 H =
            random_su3_algebra_debug();

        SU3 U_plus =
            numerically_stable_matrix_exponential(
                eps * H
            ) * U;

        SU3 U_minus =
            numerically_stable_matrix_exponential(
                -1 * eps * H
            ) * U;

        double S_plus =
            test_action_with_coefficients(
                arr, idx, U_plus,
                beta, c0, c1);

        double S_minus =
            test_action_with_coefficients(
                arr, idx, U_minus,
                beta, c0, c1);

        double numerical =
            (S_plus - S_minus)
            / (2.0 * eps);

        double analytic =
            2.0 *
            (F * H).trace().real();

        double error =
            std::abs(numerical - analytic);

        double ratio = 0.0;

        if (previous_error > 0.0)
            ratio = previous_error / error;

        std::cout
            << eps << "\t\t"
            << numerical << "\t"
            << analytic << "\t"
            << error << "\t"
            << ratio << "\n";

        previous_error = error;
    }
}


void run_all_force_tests(const Link_array& arr, const double beta)
{
    std::cout << "\n";
    std::cout << "############################################################\n";
    std::cout << "# FORCE / ACTION CONSISTENCY TESTS\n";
    std::cout << "############################################################\n";

    link_index idx = {
        0, 0, 0, 0, 0
    };

    compare_force_to_numerical_derivative(
        arr,
        idx,
        beta,
        1.0,
        0.0,
        "PLAQUETTE ONLY"
    );

    compare_force_to_numerical_derivative(
        arr,
        idx,
        beta,
        0.0,
        1.0,
        "RECTANGLE ONLY"
    );

    compare_force_to_numerical_derivative(
        arr,
        idx,
        beta,
        5.0 / 3.0,
        -1.0 / 12.0,
        "FULL SYMANZIK"
    );
}


// ============================================================================
// INTEGRATED: COMPUTE RECTANGLE STAPLE SUM AT LINK
// ============================================================================

// SU3 compute_rectangle_staple_sum_at_link(const Link_array& arr, const link_index& link_index_array){
//     // This will be used for the Symanzik improvement to the lattice action to get rid of o(a^6) lattice artifacts. 
//     // See "Renormalization Group Flow of SU(3) Gauge Theory" by Y. Iwasaki, Nucl. Phys. B258 (1985) 141-156 for more details. 
//     lattice_index lattice_index_array = {link_index_array[0], link_index_array[1], link_index_array[2], link_index_array[3]};
//     int d = link_index_array[4];
//     link_index local_link_index_array;

//     SU3 staple;
//     staple.setZero();
//     SU3 rectangle_staple_sum; 
//     rectangle_staple_sum.setZero();

//     for (int dperp = 0; dperp < 4; dperp++){
//         if (dperp != d){
//             lattice_index tmp = lattice_index_array;
    
//             // Bottom rectangle: Type 1: The link is on the short side of the rectangle.
//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = get_SU3_at_link(arr, local_link_index_array).adjoint(); 
            
//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
//             moveup(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             rectangle_staple_sum += staple.adjoint();
//             moveup(tmp, dperp);
//             movedown(tmp, d);
            
//             assert(tmp == lattice_index_array);

//             // Top rectangle: Type 2: The link is on the short side of the rectangle.
//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = get_SU3_at_link(arr, local_link_index_array); 
            
//             moveup(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, dperp);
//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             rectangle_staple_sum += staple;
//             assert(tmp == lattice_index_array);

//             // Type 3 and Type 4: The link is on the long side of the rectangle. Looks like Z block tetris
//             // Type 3: The rectangle is below the link
//             movedown(tmp, dperp); 
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = get_SU3_at_link(arr, local_link_index_array).adjoint();
            
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, dperp);
//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, d);
//             rectangle_staple_sum += staple.adjoint();
//             assert(tmp == lattice_index_array);

//             // Type 4: The rectangle is above the link
//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = get_SU3_at_link(arr, local_link_index_array);
            
//             moveup(tmp, dperp);
//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, d);
//             rectangle_staple_sum += staple;
//             assert(tmp == lattice_index_array);

//             // Type 5 and Type 6: The link is on the long side of the rectangle. Looks like reverse Z block tetris
//             // Type 5: The rectangle is below the link
//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
//             moveup(tmp, d); 
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, dperp);
//             movedown(tmp, d);
//             rectangle_staple_sum += staple.adjoint();
//             assert(tmp == lattice_index_array);

//             // Type 6: The rectangle is above the link
//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array);

//             moveup(tmp, dperp);
//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, d);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, d);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             movedown(tmp, dperp);
//             local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
//             staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

//             rectangle_staple_sum += staple;
//             assert(tmp == lattice_index_array);
//         }
//     }
//     return rectangle_staple_sum;
// }


// ============================================================================
// DEBUG RECTANGLE STAPLES & TYPES
// ============================================================================

RectangleStapleDebug compute_rectangle_staples_debug(
    const Link_array& arr,
    const link_index& link_index_array)
{
    RectangleStapleDebug result;

    result.type1.setZero();
    result.type2.setZero();
    result.type3.setZero();
    result.type4.setZero();
    result.type5.setZero();
    result.type6.setZero();
    result.total.setZero();

    lattice_index x =
    {
        link_index_array[0],
        link_index_array[1],
        link_index_array[2],
        link_index_array[3]
    };

    int d = link_index_array[4];
    link_index local_link_index_array;

    for (int dperp = 0; dperp < 4; ++dperp)
    {
        if (dperp == d)
            continue;

        // TYPE 1
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = get_SU3_at_link(arr, local_link_index_array).adjoint(); 
            
            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
            moveup(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            result.type1 += staple.adjoint();
        }

        // TYPE 2
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = get_SU3_at_link(arr, local_link_index_array); 
            
            moveup(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, dperp);
            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            result.type2 += staple;
        }

        // TYPE 3
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            movedown(tmp, dperp); 
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = get_SU3_at_link(arr, local_link_index_array).adjoint();
            
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, dperp);
            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, d);
            result.type3 += staple.adjoint();
        }

        // TYPE 4
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = get_SU3_at_link(arr, local_link_index_array);
            
            moveup(tmp, dperp);
            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, d);
            result.type4 += staple;
        }

        // TYPE 5
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);
            
            moveup(tmp, d); 
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, dperp);
            movedown(tmp, d);
            result.type5 += staple.adjoint();
        }

        // TYPE 6
        {
            lattice_index tmp = x;
            SU3 staple;
            staple.setZero();

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array);

            moveup(tmp, dperp);
            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, d);
            local_link_index_array = combine_lattice_index_with_direction(tmp, d);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            movedown(tmp, dperp);
            local_link_index_array = combine_lattice_index_with_direction(tmp, dperp);
            staple = staple * get_SU3_at_link(arr, local_link_index_array).adjoint();

            result.type6 += staple;
        }
    }

    result.total =
          result.type1
        + result.type2
        + result.type3
        + result.type4
        + result.type5
        + result.type6;

    return result;
}


void test_rectangle_staple_types(const Link_array& arr)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "TEST: SIX RECTANGLE STAPLE TYPES\n";
    std::cout << "============================================================\n";

    link_index idx =
    {
        0, 0, 0, 0, 0
    };

    SU3 U =
        get_SU3_at_link(arr, idx);

    RectangleStapleDebug R =
        compute_rectangle_staples_debug(
            arr,
            idx
        );

    std::cout << "\nLink U:\n";
    std::cout << U << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 1\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type1 << "\n";
    std::cout << "norm = " << R.type1.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 2\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type2 << "\n";
    std::cout << "norm = " << R.type2.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 3\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type3 << "\n";
    std::cout << "norm = " << R.type3.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 4\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type4 << "\n";
    std::cout << "norm = " << R.type4.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 5\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type5 << "\n";
    std::cout << "norm = " << R.type5.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TYPE 6\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.type6 << "\n";
    std::cout << "norm = " << R.type6.norm() << "\n";

    std::cout << "\n------------------------------------------------------------\n";
    std::cout << "TOTAL RECTANGLE STAPLE SUM\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << R.total << "\n";
    std::cout << "norm = " << R.total.norm() << "\n";
}