#include "Lattice.h"
#include "Parameters.h"
#include "Fermions.h"
#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <cassert>
#include <cmath>
#include <algorithm>


int moveup_fermion(lattice_index& lattice_index_array, int link_direction) {
    lattice_index_array[link_direction] += 1;
    if (link_direction == 3) {
        if (lattice_index_array[link_direction] >= temporal_size) {
            lattice_index_array[link_direction] -= temporal_size;
            return -1; // Anti-periodic boundary crossed
        }
    } else {
        if (lattice_index_array[link_direction] >= Spatial_Size) {
            lattice_index_array[link_direction] -= Spatial_Size;
        }
    }
    return 1; // Normal phase
}

int movedown_fermion(lattice_index& lattice_index_array, int link_direction) {
    if (lattice_index_array[link_direction] == 0) {
        if (link_direction == 3) {
            lattice_index_array[link_direction] = temporal_size - 1;
            return -1; // Anti-periodic boundary crossed
        } else {
            lattice_index_array[link_direction] = Spatial_Size - 1;
        }
    } else {
        lattice_index_array[link_direction] -= 1;
    }
    return 1; // Normal phase
}




int flat_index_fermion_field(const fermion_field_index_type& index) {
    int i1 = index[0];
    int i2 = index[1];
    int i3 = index[2];
    int i4 = index[3];
    int i5 = index[4];
    int i6 = index[5];
    return (((((i1 * Spatial_Size + i2) * Spatial_Size + i3) * temporal_size + i4) * 3 + i5)) * 2 + i6;
}

fermion_field_index_type tensor_index_fermion_field(int flat_index) {
    int i6 = flat_index % 2;
    flat_index /= 2;
    int i5 = flat_index % 3;
    flat_index /= 3;
    int i4 = flat_index % temporal_size;
    flat_index /= temporal_size;
    int i3 = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int i2 = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int i1 = flat_index;
    return {i1, i2, i3, i4, i5, i6};
}

void set_fermion_field_value(std::vector<double>& fermion_field, const fermion_field_index_type& index, const double& value) {
    int idx = flat_index_fermion_field(index);
    fermion_field[idx] = value;
}

double get_fermion_field_value(const std::vector<double>& fermion_field, const fermion_field_index_type& index) {
    int idx = flat_index_fermion_field(index);
    return fermion_field[idx];
}

complex get_fermion_field_value_at_color(const std::vector<double>& fermion_field, const lattice_index& index, int color_index) {
    if (color_index < 0 || color_index >= 3) {
        throw std::out_of_range("Color index must be 0, 1, or 2.");
    }
    fermion_field_index_type modified_index = {index[0], index[1], index[2], index[3], 0, 0}; 
    modified_index[4] = color_index; 

    int idx = flat_index_fermion_field(modified_index);
    return complex(fermion_field[idx], fermion_field[idx + 1]); 
}

Eigen::Vector3cd extract_color_vector(const std::vector<double>& fermion_field, const lattice_index& index) {
    Eigen::Vector3cd color_vector;
    for (int color_index = 0; color_index < 3; ++color_index) {
        color_vector[color_index] = get_fermion_field_value_at_color(fermion_field, index, color_index);
    }
    return color_vector;
}

void set_fermion_field_value_at_lattice_site(std::vector<double>& fermion_field, const lattice_index& index, const Eigen::Vector3cd& value) {
    for (int color_index = 0; color_index < 3; ++color_index) {
        fermion_field_index_type modified_index = {index[0], index[1], index[2], index[3], 0, 0}; 
        modified_index[4] = color_index; 

        int idx = flat_index_fermion_field(modified_index);
        fermion_field[idx] = value[color_index].real();
        fermion_field[idx + 1] = value[color_index].imag();
    }
}

complex generateGaussianComplex() {
    static const double stddev = 1.0 / std::sqrt(2.0);
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::normal_distribution<double> d(0.0, stddev);

    return complex(d(gen), d(gen));
}

Eigen::Vector3cd apply_dirac_operator_at_lattice_site(const std::vector<double>& fermion_field, const Link_array& arr, const lattice_index& index) {
    Eigen::Vector3cd color_vector = extract_color_vector(fermion_field, index);
    Eigen::Vector3cd result_vector = fermion_mass * color_vector;
    lattice_index index_copy = index; 
    
    for (int d = 0; d < 4; ++d) {
        int phase = staggered_phase_factor(index, d);

        // Forward hop
        SU3 U = get_SU3_at_link(arr, combine_lattice_index_with_direction(index_copy, d)); 
        int bc_phase_up = moveup_fermion(index_copy, d); // Use fermion move
        Eigen::Vector3cd neighbor_color_vector_eigen = extract_color_vector(fermion_field, index_copy);
        
        result_vector += 0.5 * phase * bc_phase_up * U * neighbor_color_vector_eigen;

        movedown_fermion(index_copy, d); // Reset to origin
        
        // Backward hop
        int bc_phase_down = movedown_fermion(index_copy, d); // Use fermion move
        U = get_SU3_at_link(arr, combine_lattice_index_with_direction(index_copy, d)).adjoint();
        neighbor_color_vector_eigen = extract_color_vector(fermion_field, index_copy);
        
        result_vector -= 0.5 * phase * bc_phase_down * U * neighbor_color_vector_eigen;
        
        moveup_fermion(index_copy, d); // Reset to origin
        assert(index_copy == index);
    }
    return result_vector;
}

Eigen::Vector3cd apply_dirac_dagger_at_site(const std::vector<double>& fermion_field, const Link_array& arr, const lattice_index& index) {
    Eigen::Vector3cd color_vector = extract_color_vector(fermion_field, index);
    Eigen::Vector3cd result_vector = fermion_mass * color_vector;
    lattice_index index_copy = index;

    for (int d = 0; d < 4; ++d) {
        int phase = staggered_phase_factor(index, d);

        // Forward hop
        SU3 U = get_SU3_at_link(arr, combine_lattice_index_with_direction(index_copy, d)); 
        int bc_phase_up = moveup_fermion(index_copy, d);
        Eigen::Vector3cd neighbor_forward = extract_color_vector(fermion_field, index_copy);
        
        result_vector -= 0.5 * phase * bc_phase_up * (U * neighbor_forward);

        movedown_fermion(index_copy, d); // Reset to origin
        
        // Backward hop
        int bc_phase_down = movedown_fermion(index_copy, d); 
        U = get_SU3_at_link(arr, combine_lattice_index_with_direction(index_copy, d)).adjoint();
        Eigen::Vector3cd neighbor_backward = extract_color_vector(fermion_field, index_copy);
        
        result_vector += 0.5 * phase * bc_phase_down * (U * neighbor_backward);

        moveup_fermion(index_copy, d); // Reset to origin
        assert(index_copy == index);
    }
    return result_vector;
}

void generate_chi_field(std::vector<double>& fermion_field) {
    for (int i = 0; i < Staggered_Fermion_field_indices; i += 2) {
        complex random_value = generateGaussianComplex();
        fermion_field[i] = random_value.real();
        fermion_field[i + 1] = random_value.imag();
    }
}

std::vector<double> generate_phi_field(const Link_array& arr) {
    std::vector<double> phi_field(Staggered_Fermion_field_indices, 0.0);
    std::vector<double> chi(Staggered_Fermion_field_indices, 0.0);
    generate_chi_field(chi); 

    for (int i1 = 0; i1 < Spatial_Size; ++i1) {
        for (int i2 = 0; i2 < Spatial_Size; ++i2) {
            for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                for (int i4 = 0; i4 < temporal_size; ++i4) {
                    lattice_index site = {i1, i2, i3, i4};
                    Eigen::Vector3cd phi_n = apply_dirac_operator_at_lattice_site(chi, arr, site);
                    set_fermion_field_value_at_lattice_site(phi_field, site, phi_n);
                }
            }
        }
    }
    return phi_field;
}

int staggered_phase_factor(const lattice_index& index, int direction) {
    if (direction < 0 || direction >= 4) {
        throw std::out_of_range("Direction must be between 0 and 3.");
    }
    int phase = 1;
    for (int mu = 0; mu < direction; ++mu) {
        phase *= (index[mu] % 2 == 0) ? 1 : -1;
    }
    return phase;
}

std::array<Eigen::Matrix3cd, 8> get_gell_mann_matrices() {
    using namespace std::complex_literals; 
    std::array<Eigen::Matrix3cd, 8> gell_mann;

    for (auto& lambda : gell_mann) {
        lambda.setZero();
    }

    gell_mann[0] << 0, 1, 0, 1, 0, 0, 0, 0, 0;
    gell_mann[1] << 0, -1i, 0, 1i,  0, 0, 0,   0, 0;
    gell_mann[2] << 1,  0, 0, 0, -1, 0, 0,  0, 0;
    gell_mann[3] << 0, 0, 1, 0, 0, 0, 1, 0, 0;
    gell_mann[4] << 0, 0, -1i, 0, 0,   0, 1i, 0,   0;
    gell_mann[5] << 0, 0, 0, 0, 0, 1, 0, 1, 0;
    gell_mann[6] << 0, 0,   0, 0, 0, -1i, 0, 1i,  0;
    double a = 1.0 / std::sqrt(3.0);
    gell_mann[7] << a, 0, 0, 0, a, 0, 0, 0, -2.0 * a;

    return gell_mann;
}

std::vector<double> generate_conjugate_field_configuration() {
    std::vector<double> P_field(conjugate_field_array_size, 0.0);
    thread_local std::mt19937_64 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);
    static const auto gellman_matrices = get_gell_mann_matrices();

    for (int i = 0; i < Spatial_Size; i++) {
        for (int j = 0; j < Spatial_Size; j++) {
            for (int k = 0; k < Spatial_Size; k++) {
                for (int l = 0; l < temporal_size; l++) {
                    for (int d = 0; d < 4; d++) {
                        link_index link_index_array = {i, j, k, l, d};
                        SU3 U;
                        U.setZero();
                        for (int m = 0; m < 8; m++) {
                            U += dist(rng) * 0.5 * gellman_matrices[m];
                        }
                        set_link_SU3(P_field, link_index_array, U); 
                    }
                }
            }
        }
    }
    return P_field;
}

Eigen::Matrix3cd proj_su3(const Eigen::Matrix3cd& M) {
    Eigen::Matrix3cd A = 0.5 * (M - M.adjoint());
    complex trace_third = (A.trace() / 3.0);
    return A - trace_third * Eigen::Matrix3cd::Identity();
}

Eigen::Matrix3cd compute_gauge_force_at_link(const Link_array& arr, const link_index& idx, double beta) {
    SU3 U = get_SU3_at_link(arr, idx);
    SU3 V = compute_full_staple_sum_symanzik_at_link(arr, idx); 
    Eigen::Matrix3cd Q = U * V.adjoint();
    return -complex(0, 1) * (beta / 6.0) * proj_su3(Q);
}

Eigen::Matrix3cd compute_fermion_force_at_link(const std::vector<double>& X_field, const std::vector<double>& Y_field, const Link_array& arr, const link_index& idx) {
    lattice_index site = {idx[0], idx[1], idx[2], idx[3]};
    int mu = idx[4];

    Eigen::Vector3cd X_n = extract_color_vector(X_field, site);
    Eigen::Vector3cd Y_n = extract_color_vector(Y_field, site);

    lattice_index site_plus_mu = site;
    moveup(site_plus_mu, mu);

    Eigen::Vector3cd X_n_plus_mu = extract_color_vector(X_field, site_plus_mu);
    Eigen::Vector3cd Y_n_plus_mu = extract_color_vector(Y_field, site_plus_mu);

    SU3 U = get_SU3_at_link(arr, idx);
    int eta = staggered_phase_factor(site, mu);

    Eigen::Matrix3cd Term1 = (U * Y_n_plus_mu) * X_n.adjoint();
    Eigen::Matrix3cd Term2 = Y_n * (U * X_n_plus_mu).adjoint();
    Eigen::Matrix3cd K = Term1 + Term2;

    return complex(0.0, 0.5) * static_cast<double>(eta) * proj_su3(K);
}

double field_inner_product_real(const std::vector<double>& A, const std::vector<double>& B) {
    double sum = 0.0;
    for (size_t i = 0; i < Staggered_Fermion_field_indices; ++i) {
        sum += A[i] * B[i];
    }
    return sum;
}

// FIXED: Performance Optimization - Pass out_field by reference to avoid reallocation
void apply_DDdagger(const std::vector<double>& in_field, const Link_array& arr, std::vector<double>& out_field) {
    thread_local std::vector<double> temp(Staggered_Fermion_field_indices, 0.0);
    std::fill(temp.begin(), temp.end(), 0.0);
    std::fill(out_field.begin(), out_field.end(), 0.0);

    // Step 1: Apply D^\dagger
    for (int i1 = 0; i1 < Spatial_Size; ++i1) {
        for (int i2 = 0; i2 < Spatial_Size; ++i2) {
            for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                for (int i4 = 0; i4 < temporal_size; ++i4) {
                    lattice_index site = {i1, i2, i3, i4};
                    Eigen::Vector3cd vec = apply_dirac_dagger_at_site(in_field, arr, site);
                    set_fermion_field_value_at_lattice_site(temp, site, vec);
                }
            }
        }
    }

    // Step 2: Apply D
    for (int i1 = 0; i1 < Spatial_Size; ++i1) {
        for (int i2 = 0; i2 < Spatial_Size; ++i2) {
            for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                for (int i4 = 0; i4 < temporal_size; ++i4) {
                    lattice_index site = {i1, i2, i3, i4};
                    Eigen::Vector3cd vec = apply_dirac_operator_at_lattice_site(temp, arr, site);
                    set_fermion_field_value_at_lattice_site(out_field, site, vec);
                }
            }
        }
    }
}

std::vector<double> solve_CG_DDdagger(const std::vector<double>& phi_field, const Link_array& arr, double tol = 1e-10, int max_iter = 5000) {
    std::vector<double> X(Staggered_Fermion_field_indices, 0.0);
    std::vector<double> r = phi_field;
    std::vector<double> p = r;
    
    std::vector<double> v(Staggered_Fermion_field_indices, 0.0);

    double r_sq = field_inner_product_real(r, r);
    double phi_sq = field_inner_product_real(phi_field, phi_field);

    if (phi_sq < 1e-28) {
        return X;
    }

    double target_r_sq = tol * tol * phi_sq;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (r_sq < target_r_sq) break;

        apply_DDdagger(p, arr, v);

        double p_dot_v = field_inner_product_real(p, v);
        double alpha = r_sq / p_dot_v;

        for (size_t i = 0; i < Staggered_Fermion_field_indices; ++i) {
            X[i] += alpha * p[i];
            r[i] -= alpha * v[i];
        }

        double r_sq_new = field_inner_product_real(r, r);
        double beta_cg = r_sq_new / r_sq;
        r_sq = r_sq_new;

        for (size_t i = 0; i < Staggered_Fermion_field_indices; ++i) {
            p[i] = r[i] + beta_cg * p[i];
        }
    }
    if (r_sq >= target_r_sq) {
        throw std::runtime_error("CG Solver failed to converge after " + 
                                 std::to_string(max_iter) + 
                                 " iterations. Final r_sq = " + std::to_string(r_sq) + 
                                 ", Target = " + std::to_string(target_r_sq));
    }

    return X;
}


std::vector<double> solve_X_field(const std::vector<double>& phi_field, const Link_array& arr) {
    return solve_CG_DDdagger(phi_field, arr);
}

std::vector<double> solve_Y_field(const std::vector<double>& X_field, const Link_array& arr) {
    std::vector<double> Y_field(Staggered_Fermion_field_indices, 0.0);
    for (int i1 = 0; i1 < Spatial_Size; ++i1) {
        for (int i2 = 0; i2 < Spatial_Size; ++i2) {
            for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                for (int i4 = 0; i4 < temporal_size; ++i4) {
                    lattice_index site = {i1, i2, i3, i4};
                    Eigen::Vector3cd Y_n = apply_dirac_dagger_at_site(X_field, arr, site);
                    set_fermion_field_value_at_lattice_site(Y_field, site, Y_n);
                }
            }
        }
    }
    return Y_field;
}

std::vector<double> compute_full_force_field(const std::vector<double>& phi_field, const Link_array& arr, double beta) {   
    std::vector<double> X_field = solve_X_field(phi_field, arr);
    std::vector<double> Y_field = solve_Y_field(X_field, arr);
    std::vector<double> force_field(conjugate_field_array_size, 0.0);

    for (int i = 0; i < Spatial_Size; i++) {
        for (int j = 0; j < Spatial_Size; j++) {
            for (int k = 0; k < Spatial_Size; k++) {
                for (int l = 0; l < temporal_size; l++) {
                    for (int d = 0; d < 4; d++) {
                        link_index link_index_array = {i, j, k, l, d};
                        SU3 F_gauge   = compute_gauge_force_at_link(arr, link_index_array, beta);
                        SU3 F_fermion = compute_fermion_force_at_link(X_field, Y_field, arr, link_index_array);
                        SU3 F_total   = F_gauge + F_fermion;
                        set_link_SU3(force_field, link_index_array, F_total); 
                    }
                }
            }
        }
    }
    return force_field;
}

std::vector<double> compute_full_force_field_last_step(const std::vector<double>& X_field, const std::vector<double>& phi_field, const Link_array& arr, double beta) {   
    std::vector<double> Y_field = solve_Y_field(X_field, arr);
    std::vector<double> force_field(conjugate_field_array_size, 0.0);

    for (int i = 0; i < Spatial_Size; i++) {
        for (int j = 0; j < Spatial_Size; j++) {
            for (int k = 0; k < Spatial_Size; k++) {
                for (int l = 0; l < temporal_size; l++) {
                    for (int d = 0; d < 4; d++) {
                        link_index link_index_array = {i, j, k, l, d};
                        SU3 F_gauge   = compute_gauge_force_at_link(arr, link_index_array, beta);
                        SU3 F_fermion = compute_fermion_force_at_link(X_field, Y_field, arr, link_index_array);
                        SU3 F_total   = F_gauge + F_fermion;
                        set_link_SU3(force_field, link_index_array, F_total); 
                    }
                }
            }
        }
    }
    return force_field;
}

// FIXED: Performance Optimization - Pass by const reference
double compute_Tr_P2(const std::vector<double>& P_field) {
    double computation = 0.0;
    for (int i = 0; i < conjugate_field_array_size; i++) {
        computation += 0.5 * P_field[i] * P_field[i];
    }
    return computation;
}

double compute_phi_dagger_DDdagger_inverse_phi(const std::vector<double>& phi_field, const Link_array& arr) {
    std::vector<double> X_field = solve_X_field(phi_field, arr);
    return field_inner_product_real(phi_field, X_field); 
}

void apply_initial_step_to_conjugate_field(std::vector<double>& P_field, double beta, double epsilon, const std::vector<double>& phi_field, const Link_array& arr) {
    std::vector<double> Force_field = compute_full_force_field(phi_field, arr, beta);
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_field[i] -= epsilon * 0.5 * Force_field[i];
    }
}

void apply_intermediate_update_to_gauge_field_and_conjugate_field(std::vector<double>& P_field, double beta, double epsilon, const std::vector<double>& phi_field, Link_array& arr) {
    SU3 U_pre;
    SU3 P_pre; 

    // FIXED: Properly scoped nested loops and correctly ordered leapfrog updates
    for (int n = 1; n < number_of_steps_fermion; n++) {
        for (int i = 0; i < Spatial_Size; i++) {
            for (int j = 0; j < Spatial_Size; j++) {
                for (int k = 0; k < Spatial_Size; k++) {
                    for (int l = 0; l < temporal_size; l++) {
                        for (int d = 0; d < 4; d++) {
                            link_index link_index_array = {i, j, k, l, d};
                            U_pre = get_SU3_at_link(arr, link_index_array);
                            P_pre = complex(0, epsilon) * get_SU3_at_link(P_field, link_index_array);
                            U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                            set_link_SU3(arr, link_index_array, U_pre);
                        }
                    }
                }
            }
        }               

        std::vector<double> force_field_full = compute_full_force_field(phi_field, arr, beta);
        for (int i = 0; i < conjugate_field_array_size; i++) {
            P_field[i] -= epsilon * force_field_full[i];
        }
    } 
}

std::vector<double> apply_final_step(std::vector<double>& P_field, double beta, double epsilon, const std::vector<double>& phi_field, Link_array& arr) {
    SU3 U_pre;
    SU3 P_pre; 
    
    // FIXED: Formatted deep nesting with explicit braces
    for (int i = 0; i < Spatial_Size; i++) {
        for (int j = 0; j < Spatial_Size; j++) {
            for (int k = 0; k < Spatial_Size; k++) {
                for (int l = 0; l < temporal_size; l++) {
                    for (int d = 0; d < 4; d++) {
                        link_index link_index_array = {i, j, k, l, d};
                        U_pre = get_SU3_at_link(arr, link_index_array);
                        P_pre = complex(0, epsilon) * get_SU3_at_link(P_field, link_index_array);
                        U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                        set_link_SU3(arr, link_index_array, U_pre);
                    }
                }   
            }
        }
    }
        
    std::vector<double> X_field = solve_X_field(phi_field, arr);
    std::vector<double> force_field_full = compute_full_force_field_last_step(X_field, phi_field, arr, beta);
    for (int i = 0; i < conjugate_field_array_size; i++) {
        P_field[i] = P_field[i] - epsilon * 0.5 * force_field_full[i];
    }
    return X_field; 
}

// FIXED: Indentation pulled to the left margin. Passed X_field by const ref. Applied the 0.5 kinetic energy factor.
void Monte_carlo_step(
    double initial_Tr_P2,
    double initial_phi_field_action,
    const std::vector<double>& P_field_final,
    const std::vector<double>& phi_field,
    double gauge_field_initial_action, // Passed by value is safer here since it's just a double
    Link_array& gauge_field_initial,
    Link_array& gauge_field_final,
    const std::vector<double>& X_field, 
    double beta) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double r = dis(gen);

    
    double computation = initial_Tr_P2 - compute_Tr_P2(P_field_final)
                       + gauge_field_initial_action - compute_action(gauge_field_final, beta) 
                       + initial_phi_field_action - field_inner_product_real(phi_field, X_field);
    
    double acceptance_prob = exp(computation);
    if (r < acceptance_prob) { 
        gauge_field_initial = gauge_field_final; 
    }       
}


void single_sweep_monte_carlo_update_WITH_FERMIONS(Link_array& gauge_field, double beta) {
    
    double gauge_action = compute_action(gauge_field, beta);
    
    std::vector<double> phi_field = generate_phi_field(gauge_field);
    std::vector<double> P_field = generate_conjugate_field_configuration();
    
    double initial_conjugate_action = compute_Tr_P2(P_field);
    double initial_fermion_action = compute_phi_dagger_DDdagger_inverse_phi(phi_field, gauge_field);
    std::vector<double> gauge_field_copy = gauge_field;

    apply_initial_step_to_conjugate_field(P_field, beta, epsilon, phi_field, gauge_field_copy);
    apply_intermediate_update_to_gauge_field_and_conjugate_field(P_field, beta, epsilon, phi_field, gauge_field_copy);
    std::vector<double> X_field = apply_final_step(P_field, beta, epsilon, phi_field, gauge_field_copy);
    
    Monte_carlo_step(initial_conjugate_action, initial_fermion_action, P_field, phi_field, gauge_action, gauge_field, gauge_field_copy, X_field, beta);
}

double sinx_over_x_stable(double x) {
    if (std::abs(x) <= 0.05) {
        double x2 = x * x;
        return 1.0 - (x2 / 6.0) * (1.0 - (x2 / 20.0) * (1.0 - x2 / 42.0));
    } else {
        return std::sin(x) / x;
    }
}

SU3 numerically_stable_matrix_exponential(const SU3& Q) {
    SU3 Q_squared = Q * Q;
    SU3 Q_cubed = Q_squared * Q;

    if (std::abs(Q_cubed.trace().imag()) > 1e-10) {
        throw std::invalid_argument("Error: Q cubed matrix has non-zero imaginary trace.");
    }

    double c0 = (1.0 / 3.0) * Q_cubed.trace().real();
    double c1 = (1.0 / 2.0) * Q_squared.trace().real();

    if (c1 < 0.0) {
        throw std::invalid_argument("Error: Q squared matrix has negative trace.");
    }

    if (c1 < 1e-12) {
        return Eigen::Matrix3cd::Identity() + std::complex<double>(0, 1) * Q - 0.5 * Q_squared;
    }

    bool is_negative_c0 = (c0 < 0.0);
    c0 = std::abs(c0);

    double c0_max = 2.0 * std::sqrt((c1 * c1 * c1) / 27.0);
    double ratio = (c0_max > 0.0) ? std::clamp(c0 / c0_max, -1.0, 1.0) : 1.0;
    double theta = std::acos(ratio);

    double w = std::sqrt(c1) * std::sin(theta / 3.0);
    double u = std::sqrt(c1 / 3.0) * std::cos(theta / 3.0);

    const std::complex<double> i2(0.0, 2.0);
    const std::complex<double> i1_neg(0.0, -1.0);
    const std::complex<double> i3(0.0, 3.0);

    std::complex<double> exp_2u = std::exp(i2 * u);
    std::complex<double> exp_neg_u = std::exp(i1_neg * u);

    std::complex<double> h0 = (u * u - w * w) * exp_2u + 
        exp_neg_u * (8.0 * u * u * std::cos(w) + i2 * u * (3.0 * u * u + w * w) * sinx_over_x_stable(w));

    std::complex<double> h1 = 2.0 * u * exp_2u - 
        exp_neg_u * (2.0 * u * std::cos(w) + i1_neg * (3.0 * u * u - w * w) * sinx_over_x_stable(w));

    std::complex<double> h2 = exp_2u - 
        exp_neg_u * (std::cos(w) + i3 * u * sinx_over_x_stable(w));

    double denom = 9.0 * u * u - w * w;
    
    std::complex<double> f0 = h0 / denom;
    std::complex<double> f1 = h1 / denom;
    std::complex<double> f2 = h2 / denom;

    if (is_negative_c0) {
        f0 = std::conj(f0);
        f1 = -std::conj(f1);
        f2 = std::conj(f2);
    }

    return f0 * Eigen::Matrix3cd::Identity() + f1 * Q + f2 * Q_squared;
}