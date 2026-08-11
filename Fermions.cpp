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



// ============================================================================
// GLOBAL LOOKUP TABLES FOR FAST SITE INDEXING & STAGGERED PHASES
// ============================================================================
struct LatticeTables {
    int V;
    std::vector<int> up_site;
    std::vector<int> dn_site;
    std::vector<int> bc_up;
    std::vector<int> bc_dn;
    std::vector<int> phase;
    std::vector<lattice_index> site_coords;

    LatticeTables() {
        V = Spatial_Size * Spatial_Size * Spatial_Size * temporal_size;
        up_site.resize(V * 4);
        dn_site.resize(V * 4);
        bc_up.resize(V * 4);
        bc_dn.resize(V * 4);
        phase.resize(V * 4);
        site_coords.resize(V);

        for (int i1 = 0; i1 < Spatial_Size; ++i1) {
            for (int i2 = 0; i2 < Spatial_Size; ++i2) {
                for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                    for (int i4 = 0; i4 < temporal_size; ++i4) {
                        int s = (((i1 * Spatial_Size + i2) * Spatial_Size + i3) * temporal_size + i4);
                        site_coords[s] = {i1, i2, i3, i4};

                        for (int d = 0; d < 4; ++d) {
                            // Staggered Phase Factor
                            int eta = 1;
                            lattice_index idx = {i1, i2, i3, i4};
                            for (int mu = 0; mu < d; ++mu) {
                                eta *= (idx[mu] % 2 == 0) ? 1 : -1;
                            }
                            phase[s * 4 + d] = eta;

                            // Upward Neighbor & Boundary Condition
                            lattice_index up_idx = idx;
                            up_idx[d] += 1;
                            int b_up = 1;
                            if (d == 3) {
                                if (up_idx[d] >= temporal_size) {
                                    up_idx[d] -= temporal_size;
                                    b_up = -1; // Anti-periodic temporal BC
                                }
                            } else {
                                if (up_idx[d] >= Spatial_Size) {
                                    up_idx[d] -= Spatial_Size;
                                }
                            }
                            int up_s = (((up_idx[0] * Spatial_Size + up_idx[1]) * Spatial_Size + up_idx[2]) * temporal_size + up_idx[3]);
                            up_site[s * 4 + d] = up_s;
                            bc_up[s * 4 + d] = b_up;

                            // Downward Neighbor & Boundary Condition
                            lattice_index dn_idx = idx;
                            int b_dn = 1;
                            if (dn_idx[d] == 0) {
                                if (d == 3) {
                                    dn_idx[d] = temporal_size - 1;
                                    b_dn = -1; // Anti-periodic temporal BC
                                } else {
                                    dn_idx[d] = Spatial_Size - 1;
                                }
                            } else {
                                dn_idx[d] -= 1;
                            }
                            int dn_s = (((dn_idx[0] * Spatial_Size + dn_idx[1]) * Spatial_Size + dn_idx[2]) * temporal_size + dn_idx[3]);
                            dn_site[s * 4 + d] = dn_s;
                            bc_dn[s * 4 + d] = b_dn;
                        }
                    }
                }
            }
        }
    }
};

static const LatticeTables tables;

// Fast contiguous memory accessors for site vectors
inline Eigen::Vector3cd extract_color_vector_s(const std::vector<double>& f, int s) {
    int idx = s * 6;
    return Eigen::Vector3cd(
        std::complex<double>(f[idx],     f[idx + 1]),
        std::complex<double>(f[idx + 2], f[idx + 3]),
        std::complex<double>(f[idx + 4], f[idx + 5])
    );
}

inline void set_color_vector_s(std::vector<double>& f, int s, const Eigen::Vector3cd& val) {
    int idx = s * 6;
    f[idx]     = val[0].real();
    f[idx + 1] = val[0].imag();
    f[idx + 2] = val[1].real();
    f[idx + 3] = val[1].imag();
    f[idx + 4] = val[2].real();
    f[idx + 5] = val[2].imag();
}


// ============================================================================
// ORIGINAL FIELD INDEXING FUNCTIONS (KEPT FOR COMPATIBILITY)
// ============================================================================
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
    int s = (((index[0] * Spatial_Size + index[1]) * Spatial_Size + index[2]) * temporal_size + index[3]);
    return extract_color_vector_s(fermion_field, s);
}

void set_fermion_field_value_at_lattice_site(std::vector<double>& fermion_field, const lattice_index& index, const Eigen::Vector3cd& value) {
    int s = (((index[0] * Spatial_Size + index[1]) * Spatial_Size + index[2]) * temporal_size + index[3]);
    set_color_vector_s(fermion_field, s, value);
}

complex generateGaussianComplex()
{
    static const double stddev = 1.0/std::sqrt(2.0);

    return complex(get_normal_random(0.0,stddev),get_normal_random(0.0,stddev));
}


// ============================================================================
// OPTIMIZED DIRAC OPERATORS
// ============================================================================
inline Eigen::Vector3cd apply_dirac_operator_at_site_s(const std::vector<double>& fermion_field, const Link_array& arr, int s) {
    Eigen::Vector3cd color_vector = extract_color_vector_s(fermion_field, s);
    Eigen::Vector3cd result_vector = fermion_mass * color_vector;
    const lattice_index& index = tables.site_coords[s];

    for (int d = 0; d < 4; ++d) {
        int phase = tables.phase[s * 4 + d];

        // Forward hop
        link_index l_idx = {index[0], index[1], index[2], index[3], d};
        SU3 U = get_SU3_at_link(arr, l_idx); 
        int up_s = tables.up_site[s * 4 + d];
        int bc_phase_up = tables.bc_up[s * 4 + d];
        Eigen::Vector3cd neighbor_color_vector_eigen = extract_color_vector_s(fermion_field, up_s);
        
        result_vector += (0.5 * phase * bc_phase_up) * (U * neighbor_color_vector_eigen);

        // Backward hop
        int dn_s = tables.dn_site[s * 4 + d];
        int bc_phase_down = tables.bc_dn[s * 4 + d];
        const lattice_index& dn_index = tables.site_coords[dn_s];
        link_index l_dn_idx = {dn_index[0], dn_index[1], dn_index[2], dn_index[3], d};
        
        SU3 U_dn = get_SU3_at_link(arr, l_dn_idx).adjoint();
        neighbor_color_vector_eigen = extract_color_vector_s(fermion_field, dn_s);
        
        result_vector -= (0.5 * phase * bc_phase_down) * (U_dn * neighbor_color_vector_eigen);
    }
    return result_vector;
}

Eigen::Vector3cd apply_dirac_operator_at_lattice_site(const std::vector<double>& fermion_field, const Link_array& arr, const lattice_index& index) {
    int s = (((index[0] * Spatial_Size + index[1]) * Spatial_Size + index[2]) * temporal_size + index[3]);
    return apply_dirac_operator_at_site_s(fermion_field, arr, s);
}

inline Eigen::Vector3cd apply_dirac_dagger_at_site_s(const std::vector<double>& fermion_field, const Link_array& arr, int s) {
    Eigen::Vector3cd color_vector = extract_color_vector_s(fermion_field, s);
    Eigen::Vector3cd result_vector = fermion_mass * color_vector;
    const lattice_index& index = tables.site_coords[s];

    for (int d = 0; d < 4; ++d) {
        int phase = tables.phase[s * 4 + d];

        // Forward hop
        link_index l_idx = {index[0], index[1], index[2], index[3], d};
        SU3 U = get_SU3_at_link(arr, l_idx); 
        int up_s = tables.up_site[s * 4 + d];
        int bc_phase_up = tables.bc_up[s * 4 + d];
        Eigen::Vector3cd neighbor_forward = extract_color_vector_s(fermion_field, up_s);
        
        result_vector -= (0.5 * phase * bc_phase_up) * (U * neighbor_forward);

        // Backward hop
        int dn_s = tables.dn_site[s * 4 + d];
        int bc_phase_down = tables.bc_dn[s * 4 + d];
        const lattice_index& dn_index = tables.site_coords[dn_s];
        link_index l_dn_idx = {dn_index[0], dn_index[1], dn_index[2], dn_index[3], d};
        
        SU3 U_dn = get_SU3_at_link(arr, l_dn_idx).adjoint();
        Eigen::Vector3cd neighbor_backward = extract_color_vector_s(fermion_field, dn_s);
        
        result_vector += (0.5 * phase * bc_phase_down) * (U_dn * neighbor_backward);
    }
    return result_vector;
}

Eigen::Vector3cd apply_dirac_dagger_at_site(const std::vector<double>& fermion_field, const Link_array& arr, const lattice_index& index) {
    int s = (((index[0] * Spatial_Size + index[1]) * Spatial_Size + index[2]) * temporal_size + index[3]);
    return apply_dirac_dagger_at_site_s(fermion_field, arr, s);
}


// ============================================================================
// FIELD GENERATION & GELL-MANN MATRICES
// ============================================================================
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

    int V = tables.V;
    #pragma omp parallel for
    for (int s = 0; s < V; ++s) {
        Eigen::Vector3cd phi_n = apply_dirac_operator_at_site_s(chi, arr, s);
        set_color_vector_s(phi_field, s, phi_n);
    }
    return phi_field;
}

int staggered_phase_factor(const lattice_index& index, int direction) {
    int s = (((index[0] * Spatial_Size + index[1]) * Spatial_Size + index[2]) * temporal_size + index[3]);
    return tables.phase[s * 4 + direction];
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
    static const auto gellman_matrices = get_gell_mann_matrices();
    int V = tables.V;
    
    // REMOVE the #pragma omp parallel for here!
    for (int s = 0; s < V; ++s) {
        const lattice_index& site = tables.site_coords[s];
        for (int d = 0; d < 4; d++) {
            SU3 U = SU3::Zero();
            for (int m = 0; m < 8; m++) {
                double p = get_normal_random(0.0,1.0);
                U += 0.5*p*gellman_matrices[m];
            }
            set_link_SU3(P_field, {site[0], site[1], site[2], site[3], d}, U); 
        } 
    }
    return P_field;
}

Eigen::Matrix3cd proj_su3(const Eigen::Matrix3cd& M) {
    Eigen::Matrix3cd A = 0.5 * (M - M.adjoint());
    complex trace_third = (A.trace() / 3.0);
    return A - trace_third * Eigen::Matrix3cd::Identity();
}


// ============================================================================
// GAUGE & FERMION FORCE CALCULATIONS (PRESERVING SYMANZIK IMPROVEMENT)
// ============================================================================
Eigen::Matrix3cd compute_gauge_force_at_link(const Link_array& arr, const link_index& idx, double beta) {
    SU3 U = get_SU3_at_link(arr, idx);
    SU3 V = compute_full_staple_sum_symanzik_at_link(arr, idx); 
    
    Eigen::Matrix3cd Q = U * V;
    
    // This is Hermitian, but NOT traceless
    Eigen::Matrix3cd F_untraced = -complex(0, 1) * (beta / 12.0) * (Q - Q.adjoint());
    
    // Explicitly project out the trace to stay in the SU(3) Lie algebra
    complex trace = F_untraced.trace();
    return F_untraced - (trace / 3.0) * Eigen::Matrix3cd::Identity();
}


// Eigen::Matrix3cd compute_fermion_force_at_link(
//     const std::vector<double>& X_field,
//     const std::vector<double>& Y_field,
//     const Link_array& arr,
//     const link_index& idx)
// {
//     const int d = idx[4];

//     const int s =
//         (((idx[0] * Spatial_Size + idx[1])
//           * Spatial_Size + idx[2])
//           * temporal_size + idx[3]);

//     const int phase =
//         tables.phase[s * 4 + d];

//     const int up_s =
//         tables.up_site[s * 4 + d];

//     const int bc_phase_up =
//         tables.bc_up[s * 4 + d];

//     // Forward staggered coefficient
//     const double coeff =
//         0.5 * static_cast<double>(phase * bc_phase_up);

//     const SU3 U =
//         get_SU3_at_link(arr, idx);

//     const Eigen::Vector3cd X_k =
//         extract_color_vector_s(X_field, s);

//     const Eigen::Vector3cd Y_up =
//         extract_color_vector_s(Y_field, up_s);

//     // A = U Y_{x+mu} X_x^\dagger
//     const Eigen::Matrix3cd A =
//         (U * Y_up) * X_k.adjoint();

//     const std::complex<double> I(0.0, 1.0);

//     // F = -i c (A - A^\dagger)
//     Eigen::Matrix3cd F =
//         -I * coeff * (A - A.adjoint());

//     // Project onto su(3) traceless Hermitian algebra
//     const std::complex<double> trace =
//         F.trace();

//     F -=
//         (trace / 3.0)
//         * Eigen::Matrix3cd::Identity();

//     return F;
// }

Eigen::Matrix3cd compute_fermion_force_at_link(
    const std::vector<double>& X_field,
    const std::vector<double>& Y_field,
    const Link_array& arr,
    const link_index& idx)
{
    const int d = idx[4];

    const int s =
        (((idx[0] * Spatial_Size + idx[1])
          * Spatial_Size + idx[2])
          * temporal_size + idx[3]);

    const int up_s = tables.up_site[s * 4 + d];

    const double eta =
        static_cast<double>(tables.phase[s * 4 + d]);

    const double bc =
        static_cast<double>(tables.bc_up[s * 4 + d]);

    const double c = 0.25 * eta * bc;

    const SU3 U = get_SU3_at_link(arr, idx);

    const Eigen::Vector3cd X_x =
        extract_color_vector_s(X_field, s);

    const Eigen::Vector3cd X_up =
        extract_color_vector_s(X_field, up_s);

    const Eigen::Vector3cd Y_x =
        extract_color_vector_s(Y_field, s);

    const Eigen::Vector3cd Y_up =
        extract_color_vector_s(Y_field, up_s);

    // ------------------------------------------------------------
    // Forward piece:
    //
    //       + c U Y(x+mu)
    //
    // ------------------------------------------------------------
    const Eigen::Matrix3cd A_forward =
        (U * Y_up) * X_x.adjoint();

    // ------------------------------------------------------------
    // Backward piece:
    //
    // At x+mu:
    //
    //       - c U^\dagger Y(x)
    //
    // ------------------------------------------------------------
    const Eigen::Matrix3cd A_backward =
        Y_x * X_up.adjoint() * U.adjoint();

    const std::complex<double> I(0.0, 1.0);

    Eigen::Matrix3cd F =
        -I * c *
        (
            (A_forward - A_forward.adjoint())
          + (A_backward - A_backward.adjoint())
        );

    // su(3) projection
    F -=
        (F.trace() / 3.0)
        * Eigen::Matrix3cd::Identity();

    return F;
}





double field_inner_product_real(const std::vector<double>& A, const std::vector<double>& B) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < Staggered_Fermion_field_indices; ++i) {
        sum += A[i] * B[i];
    }
    return sum;
}

void apply_DDdagger(const std::vector<double>& in_field, const Link_array& arr, std::vector<double>& out_field) {
    std::vector<double> temp(Staggered_Fermion_field_indices, 0.0);
    int V = tables.V;

    // Step 1: Apply D^\dagger
    #pragma omp parallel for
    for (int s = 0; s < V; ++s) {
        Eigen::Vector3cd vec = apply_dirac_dagger_at_site_s(in_field, arr, s);
        set_color_vector_s(temp, s, vec);
    }

    // Step 2: Apply D
    #pragma omp parallel for
    for (int s = 0; s < V; ++s) {
        Eigen::Vector3cd vec = apply_dirac_operator_at_site_s(temp, arr, s);
        set_color_vector_s(out_field, s, vec);
    }
}


// ============================================================================
// CONJUGATE GRADIENT & FIELD SOLVERS
// ============================================================================
std::vector<double> solve_CG_DDdagger(const std::vector<double>& phi_field, const Link_array& arr) {
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

        double p_dot_v = field_inner_product_real(p,v);
        if (!std::isfinite(p_dot_v) || std::abs(p_dot_v) < 1e-30)
        {
            throw std::runtime_error(
                "CG breakdown: p_dot_v = " + std::to_string(p_dot_v));
        }

        double alpha = r_sq / p_dot_v;

        if (!std::isfinite(alpha))
        {
            throw std::runtime_error(
                "CG produced non-finite alpha.");
        }

        #pragma omp parallel for
        for (size_t i = 0; i < Staggered_Fermion_field_indices; ++i) {
            X[i] += alpha * p[i];
            r[i] -= alpha * v[i];
        }

        double r_sq_new = field_inner_product_real(r, r);
        double beta_cg = r_sq_new / r_sq;
        if (!std::isfinite(beta_cg))
        {
            throw std::runtime_error(
                "CG produced non-finite beta.");
        }
        r_sq = r_sq_new;

        #pragma omp parallel for
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
    int V = tables.V;

    #pragma omp parallel for
    for (int s = 0; s < V; ++s) {
        Eigen::Vector3cd Y_n = apply_dirac_dagger_at_site_s(X_field, arr, s);
        set_color_vector_s(Y_field, s, Y_n);
    }
    return Y_field;
}






std::vector<double> compute_full_force_field(const std::vector<double>& X_field, 
                                             const std::vector<double>& Y_field, 
                                             const Link_array& arr, double beta) {   
    std::vector<double> force_field(conjugate_field_array_size, 0.0);
    int V = tables.V;
    #pragma omp parallel for collapse(2) 
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};

            SU3 F_gauge   = compute_gauge_force_at_link(arr, link_index_array, beta);
            SU3 F_fermion = compute_fermion_force_at_link(X_field, Y_field, arr, link_index_array);
            SU3 F_total   = F_gauge + F_fermion;
            
            set_link_SU3(force_field, link_index_array, F_total); 
        }
    }
    return force_field;
}


double compute_Tr_P2(const std::vector<double>& P_field) {
    double computation = 0.0;
    int V = tables.V;
    #pragma omp parallel for collapse(2) reduction(+:computation)
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            // Get the 4D coordinate for the current site using the lookup table
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};
            SU3 P = get_SU3_at_link(P_field, link_index_array);
            SU3 P_squared = P * P;
            double trace_P_squared = P_squared.trace().real();
            computation += trace_P_squared;
        }
    }
    return computation;
}


// ============================================================================
// HMC LEAPFROG INTEGRATION & MONTE CARLO STEPS
// ============================================================================
void apply_initial_step_to_conjugate_field(std::vector<double>& P_field, double beta, const std::vector<double>& phi_field, const Link_array& arr) {
    std::vector<double> X_field = solve_X_field(phi_field,arr);
    std::vector<double> Y_field = solve_Y_field(X_field,arr);

    std::vector<double> Force_field = compute_full_force_field(X_field, Y_field, arr, beta);
    // double maxP = 0;
     #pragma omp parallel for 
    // #pragma omp parallel for reduction(max:maxP)
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_field[i] -= epsilon * 0.5 * Force_field[i];
        // maxP = std::max(maxP, std::abs(P_field[i]));
    }
    // std::cout << "Max P norm: " << maxP << std::endl;
}

void apply_intermediate_update_to_gauge_field_and_conjugate_field(std::vector<double>& P_field, double beta, const std::vector<double>& phi_field, Link_array& arr) {
    int V = tables.V;
   
    for (int n = 1; n < number_of_steps_fermion; n++) {
        #pragma omp parallel for collapse(2)
        for (int s = 0; s < V; ++s) {
            for (int d = 0; d < 4; ++d) {
                const lattice_index& site = tables.site_coords[s];
                link_index link_index_array = {site[0], site[1], site[2], site[3], d};
                SU3 U_pre = get_SU3_at_link(arr, link_index_array);
                SU3 P_pre = epsilon * get_SU3_at_link(P_field, link_index_array);
                
                
                SU3 check_hermitian = P_pre - P_pre.adjoint();
                complex check_trace = P_pre.trace();
                if (check_hermitian.norm() > 1e-10 || std::abs(check_trace) > 1e-10) {
                    throw std::runtime_error("P_pre is not Hermitian or not traceless");
                }
                U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                
                
            
                set_link_SU3(arr, link_index_array, U_pre);
            }
        } 
        // check_unitarity(arr);

        std::vector<double> X_field = solve_X_field(phi_field,arr);
        std::vector<double> Y_field = solve_Y_field(X_field,arr);
        std::vector<double> force_field_full = compute_full_force_field(X_field, Y_field, arr, beta);
        // double maxP = 0;
        #pragma omp parallel for 
        // #pragma omp parallel for reduction(max:maxP)
        for (int i = 0; i < conjugate_field_array_size; i++) {
            P_field[i] -= epsilon * force_field_full[i];
            // maxP = std::max(maxP, std::abs(P_field[i]));
        }
        // std::cout << "Max P norm: " << maxP << std::endl;
    } 
}

std::vector<double> apply_final_step(std::vector<double>& P_field, double beta, const std::vector<double>& phi_field, Link_array& arr) {
    int V = tables.V;

    #pragma omp parallel for collapse(2)
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};
            SU3 U_pre = get_SU3_at_link(arr, link_index_array);
            SU3 P_pre = epsilon * get_SU3_at_link(P_field, link_index_array);


             SU3 check_hermitian = P_pre - P_pre.adjoint();
            complex check_trace = P_pre.trace();
            if (check_hermitian.norm() > 1e-10 || std::abs(check_trace) > 1e-10) {
                throw std::runtime_error("P_pre is not Hermitian or not traceless");
                }
            U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
           
            set_link_SU3(arr, link_index_array, U_pre);
        }
    }
    // check_unitarity(arr);
    std::vector<double> X_field = solve_X_field(phi_field,arr);
    std::vector<double> Y_field = solve_Y_field(X_field,arr);
    std::vector<double> force_field_full = compute_full_force_field(X_field, Y_field, arr, beta);
    
    #pragma omp parallel for 
    // double maxP = 0;
    // #pragma omp parallel for reduction(max:maxP)
    for (int i = 0; i < conjugate_field_array_size; i++) {
        P_field[i] = P_field[i] - epsilon * 0.5 * force_field_full[i];
        // maxP = std::max(maxP, std::abs(P_field[i]));
    }
    // std::cout << "Max P norm: " << maxP << std::endl;
    return X_field; 
}

void Monte_carlo_step(
    double initial_Tr_P2,
    double initial_phi_field_action,
    const std::vector<double>& P_field_final,
    const std::vector<double>& phi_field,
    double gauge_field_initial_action,
    Link_array& gauge_field_initial,
    Link_array& gauge_field_final,
    const std::vector<double>& X_field, 
    double beta) 
{
    double r = get_uniform_random();

    double final_Tr_P2 = compute_Tr_P2(P_field_final);
    double final_gauge_action = compute_action(gauge_field_final, beta);
    double final_phi_field_action = field_inner_product_real(phi_field, X_field);
    double computation = initial_Tr_P2 - final_Tr_P2
                       + gauge_field_initial_action - final_gauge_action 
                       + initial_phi_field_action - final_phi_field_action;

    std::cout << "Initial Tr(P^2): " << initial_Tr_P2 << std::endl;
    std::cout << "Final Tr(P^2): " << final_Tr_P2 << std::endl;
    std::cout << "Initial Gauge Action: " << gauge_field_initial_action << std::endl;
    std::cout << "Final Gauge Action: " << final_gauge_action << std::endl;
    std::cout << "Initial Fermion Action: " << initial_phi_field_action << std::endl;
    std::cout << "Final Fermion Action: " << final_phi_field_action << std::endl;
    std::cout << computation << std::endl;
    
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
    std::vector<double> X_field_initial = solve_X_field(phi_field, gauge_field);
    double initial_fermion_action = field_inner_product_real(phi_field, X_field_initial);
    std::vector<double> gauge_field_copy = gauge_field;

    apply_initial_step_to_conjugate_field(P_field, beta, phi_field, gauge_field_copy);
    apply_intermediate_update_to_gauge_field_and_conjugate_field(P_field, beta, phi_field, gauge_field_copy);
    std::vector<double> X_field = apply_final_step(P_field, beta, phi_field, gauge_field_copy);
    
    Monte_carlo_step(initial_conjugate_action, initial_fermion_action, P_field, phi_field, gauge_action, gauge_field, gauge_field_copy, X_field, beta);
}


// ============================================================================
// STABLE MATRIX EXPONENTIAL FOR SU(3)
// ============================================================================
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
    double c = std::abs(Q_cubed.trace().imag());
    
    if (c > 1e-10) {
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


// ============================================================================
// THE REVERSIBILITY TEST
// ============================================================================
void run_reversibility_test(Link_array gauge_field_initial, double beta) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "        STARTING REVERSIBILITY TEST           " << std::endl;
    std::cout << "==============================================\n" << std::endl;

    // 1. Generate starting state
    std::vector<double> phi_field = generate_phi_field(gauge_field_initial);
    std::vector<double> P_field_initial = generate_conjugate_field_configuration();
    
    Link_array U_test = gauge_field_initial;
    std::vector<double> P_test = P_field_initial;

    // 2. Integrate FORWARD
    std::cout << "Integrating trajectory forward..." << std::endl;
    apply_initial_step_to_conjugate_field(P_test, beta, phi_field, U_test);
    apply_intermediate_update_to_gauge_field_and_conjugate_field(P_test, beta, phi_field, U_test);
    apply_final_step(P_test, beta, phi_field, U_test);

    // 3. FLIP Momenta (Time Reversal)
    std::cout << "Flipping momenta for backward trajectory..." << std::endl;
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_test[i] = -1.0 * P_test[i];
    }

    // 4. Integrate BACKWARD
    std::cout << "Integrating trajectory backward..." << std::endl;
    apply_initial_step_to_conjugate_field(P_test, beta, phi_field, U_test);
    apply_intermediate_update_to_gauge_field_and_conjugate_field(P_test, beta, phi_field, U_test);
    apply_final_step(P_test, beta, phi_field, U_test);

    // 5. Evaluate Violations
    double max_u_diff = 0.0;
    for (int i = 0; i < U_test.size(); ++i) {
        double diff = std::abs(U_test[i] - gauge_field_initial[i]);
        if (diff > max_u_diff) max_u_diff = diff;
    }

    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "Maximum Gauge Link Violation: " << max_u_diff << std::endl;
    if (max_u_diff < 1e-10) {
        std::cout << "Result: PASS. Leapfrog integrator is mathematically perfect." << std::endl;
    } else if (max_u_diff < 1e-5) {
        std::cout << "Result: WARNING. Minor drift. Check CG tolerance." << std::endl;
    } else {
        std::cout << "Result: FAIL. Reversibility broken. HMC will reject." << std::endl;
    }
    std::cout << "----------------------------------------------\n" << std::endl;
}




// ============================================================================
// FORCE / STEP-SIZE SCALING TEST
// ============================================================================
// Parameterized copies of the three leapfrog stages, taking eps/n_steps as
// arguments instead of the global consts, so we can compare step sizes
// without touching the production integrator.

void local_initial_half_step(std::vector<double>& P_field, double beta,
                              const std::vector<double>& phi_field,
                              const Link_array& arr, double eps) {
    std::vector<double> X_field = solve_X_field(phi_field,arr);
    std::vector<double> Y_field = solve_Y_field(X_field,arr);
    std::vector<double> Force_field = compute_full_force_field(X_field, Y_field, arr, beta);
    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_field[i] -= eps * 0.5 * Force_field[i];
    }
}

void local_leapfrog_steps(std::vector<double>& P_field, double beta,
                           const std::vector<double>& phi_field,
                           Link_array& arr, double eps, int n_steps) {
    int V = tables.V;
    for (int n = 1; n < n_steps; n++) {
        #pragma omp parallel for collapse(2)
        for (int s = 0; s < V; ++s) {
            for (int d = 0; d < 4; ++d) {
                const lattice_index& site = tables.site_coords[s];
                link_index link_index_array = {site[0], site[1], site[2], site[3], d};
                SU3 U_pre = get_SU3_at_link(arr, link_index_array);
                SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
                U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                set_link_SU3(arr, link_index_array, U_pre);
            }
        }
        std::vector<double> X_field = solve_X_field(phi_field,arr);
        std::vector<double> Y_field = solve_Y_field(X_field,arr);
        std::vector<double> force_field_full = compute_full_force_field(X_field, Y_field, arr, beta);
        #pragma omp parallel for
        for (int i = 0; i < conjugate_field_array_size; i++) {
            P_field[i] -= eps * force_field_full[i];
        }
    }
}

std::vector<double> local_final_step(std::vector<double>& P_field, double beta,
                                      const std::vector<double>& phi_field,
                                      Link_array& arr, double eps) {
    int V = tables.V;
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};
            SU3 U_pre = get_SU3_at_link(arr, link_index_array);
            SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
            U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
            set_link_SU3(arr, link_index_array, U_pre);
        }
    }
    std::vector<double> X_field = solve_X_field(phi_field,arr);
    std::vector<double> Y_field = solve_Y_field(X_field,arr);
    std::vector<double> force_field_full = compute_full_force_field(X_field, Y_field, arr, beta);
    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; i++) {
        P_field[i] -= eps * 0.5 * force_field_full[i];
    }
    return X_field;
}



// Runs one full trajectory from a given (gauge, phi, P) start.
// gauge_field and P_field are taken by value so each call starts fresh
// from an identical, untouched copy.
TrajectoryResult run_trajectory(Link_array gauge_field, double beta,
                                 const std::vector<double>& phi_field,
                                 std::vector<double> P_field,
                                 double eps, int n_steps) {
    double initial_Tr_P2 = compute_Tr_P2(P_field);
    
    // GAUGE ONLY TEST:
    double initial_gauge_action = compute_action(gauge_field, beta);
    double H_initial = initial_Tr_P2 + initial_gauge_action; 
    // double initial_fermion_action = field_inner_product_real(phi_field, X0);
    // double H_initial = initial_Tr_P2 + initial_gauge_action + initial_fermion_action;

    local_initial_half_step(P_field, beta, phi_field, gauge_field, eps);
    local_leapfrog_steps(P_field, beta, phi_field, gauge_field, eps, n_steps);
    std::vector<double> X_final = local_final_step(P_field, beta, phi_field, gauge_field, eps);

    double final_Tr_P2 = compute_Tr_P2(P_field);
    
    
    double final_gauge_action = compute_action(gauge_field, beta);
    double final_fermion_action = field_inner_product_real(phi_field, X_final);
    double H_final = final_Tr_P2 + final_gauge_action + final_fermion_action;

    return { H_final - H_initial, H_initial, H_final };
}

void run_force_scaling_test(const Link_array& gauge_field_initial, double beta) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "        FORCE / STEP-SIZE SCALING TEST         " << std::endl;
    std::cout << "==============================================\n" << std::endl;

    // Generate stochastic fields ONCE so both trajectories start from
    // bit-identical phi, P, and gauge configurations - no RNG reseed needed.
    std::vector<double> phi_field = generate_phi_field(gauge_field_initial);
    std::vector<double> P_field_initial = generate_conjugate_field_configuration();

    double eps_full = epsilon;
    int n_steps_full = number_of_steps_fermion;

    double eps_half = epsilon * 0.5;
    int n_steps_half = number_of_steps_fermion * 2; // keep eps*n_steps fixed

    std::cout << "Running trajectory at eps = " << eps_full
              << " (" << n_steps_full << " steps)..." << std::endl;
    TrajectoryResult res_full = run_trajectory(gauge_field_initial, beta, phi_field,
                                                P_field_initial, eps_full, n_steps_full);

    std::cout << "Running trajectory at eps = " << eps_half
              << " (" << n_steps_half << " steps)..." << std::endl;
    TrajectoryResult res_half = run_trajectory(gauge_field_initial, beta, phi_field,
                                                P_field_initial, eps_half, n_steps_half);

    double abs_dH_full = std::abs(res_full.delta_H);
    double abs_dH_half = std::abs(res_half.delta_H);
    double ratio = (abs_dH_half > 1e-300) ? abs_dH_full / abs_dH_half
                                           : std::numeric_limits<double>::infinity();

    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "  |Delta H| at eps         : " << abs_dH_full << std::endl;
    std::cout << "  |Delta H| at eps/2        : " << abs_dH_half << std::endl;
    std::cout << "  Ratio (expect ~4 if force is correct): " << ratio << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    if (ratio > 3.5 && ratio < 4.5) {
        std::cout << "Result: PASS. O(eps^2) scaling confirmed." << std::endl;
    } else if (ratio > 2.0) {
        std::cout << "Result: WARNING. Rough scaling seen, not clean 4x. "
                     "Could be CG tol (1e-12) noise floor, or not asymptotic yet." << std::endl;
    } else {
        std::cout << "Result: FAIL. No O(eps^2) scaling. Force is not the "
                     "true gradient of the action (sign/factor bug)." << std::endl;
    }
    std::cout << "----------------------------------------------\n" << std::endl;
}




// 1. Force computation containing ONLY the pseudofermion contribution
std::vector<double> compute_fermion_only_force_field(
    const std::vector<double>& X_field,
    const std::vector<double>& Y_field,
    const Link_array& arr) 
{   
    std::vector<double> force_field(conjugate_field_array_size, 0.0);
    int V = tables.V;

    #pragma omp parallel for collapse(2) 
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};

            // Evaluate fermion force only (gauge force omitted)
            SU3 F_fermion = compute_fermion_force_at_link(X_field, Y_field, arr, link_index_array);
            set_link_SU3(force_field, link_index_array, F_fermion); 
        }
    }
    return force_field;
}

// 2. Local Leapfrog Half-Step (Fermion Only)
void local_initial_half_step_fermion(std::vector<double>& P_field,
                                     const std::vector<double>& phi_field,
                                     const Link_array& arr, double eps) {
    std::vector<double> X_field = solve_X_field(phi_field, arr);
    std::vector<double> Y_field = solve_Y_field(X_field, arr);
    std::vector<double> Force_field = compute_fermion_only_force_field(X_field, Y_field, arr);

    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_field[i] -= eps * 0.5 * Force_field[i];
    }
}

// 3. Local Leapfrog Intermediate Steps (Fermion Only)
void local_leapfrog_steps_fermion(std::vector<double>& P_field,
                                  const std::vector<double>& phi_field,
                                  Link_array& arr, double eps, int n_steps) {
    int V = tables.V;
    for (int n = 1; n < n_steps; n++) {
        #pragma omp parallel for collapse(2)
        for (int s = 0; s < V; ++s) {
            for (int d = 0; d < 4; ++d) {
                const lattice_index& site = tables.site_coords[s];
                link_index link_index_array = {site[0], site[1], site[2], site[3], d};
                SU3 U_pre = get_SU3_at_link(arr, link_index_array);
                SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
                U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                set_link_SU3(arr, link_index_array, U_pre);
            }
        }
        std::vector<double> X_field = solve_X_field(phi_field, arr);
        std::vector<double> Y_field = solve_Y_field(X_field, arr);
        std::vector<double> force_field = compute_fermion_only_force_field(X_field, Y_field, arr);

        #pragma omp parallel for
        for (int i = 0; i < conjugate_field_array_size; i++) {
            P_field[i] -= eps * force_field[i];
        }
    }
}

// 4. Local Leapfrog Final Step (Fermion Only)
std::vector<double> local_final_step_fermion(std::vector<double>& P_field,
                                             const std::vector<double>& phi_field,
                                             Link_array& arr, double eps) {
    int V = tables.V;
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};
            SU3 U_pre = get_SU3_at_link(arr, link_index_array);
            SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
            U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
            set_link_SU3(arr, link_index_array, U_pre);
        }
    }
    std::vector<double> X_field = solve_X_field(phi_field, arr);
    std::vector<double> Y_field = solve_Y_field(X_field, arr);
    std::vector<double> force_field = compute_fermion_only_force_field(X_field, Y_field, arr);

    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; i++) {
        P_field[i] -= eps * 0.5 * force_field[i];
    }
    return X_field;
}

// Runs one trajectory tracking H_fermion = Tr(P^2) + S_fermion
TrajectoryResult run_fermion_trajectory(Link_array gauge_field,
                                        const std::vector<double>& phi_field,
                                        std::vector<double> P_field,
                                        double eps, int n_steps) {
    double initial_Tr_P2 = compute_Tr_P2(P_field);
    std::vector<double> X0 = solve_X_field(phi_field, gauge_field);
    double initial_fermion_action = field_inner_product_real(phi_field, X0);
    
    // Hamiltonian strictly includes kinetic + fermion action
    double H_initial = initial_Tr_P2 + initial_fermion_action;

    local_initial_half_step_fermion(P_field, phi_field, gauge_field, eps);
    local_leapfrog_steps_fermion(P_field, phi_field, gauge_field, eps, n_steps);
    std::vector<double> X_final = local_final_step_fermion(P_field, phi_field, gauge_field, eps);

    double final_Tr_P2 = compute_Tr_P2(P_field);
    double final_fermion_action = field_inner_product_real(phi_field, X_final);
    
    double H_final = final_Tr_P2 + final_fermion_action;

    return { H_final - H_initial, H_initial, H_final };
}

// Executes the fermion-only scaling test
void run_fermion_force_scaling_test(const Link_array& gauge_field_initial) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "     FERMION-ONLY FORCE SCALING TEST          " << std::endl;
    std::cout << "==============================================\n" << std::endl;

    std::vector<double> phi_field = generate_phi_field(gauge_field_initial);
    std::vector<double> P_field_initial = generate_conjugate_field_configuration();

    double eps_full = epsilon;
    int n_steps_full = number_of_steps_fermion;

    double eps_half = epsilon * 0.5;
    int n_steps_half = number_of_steps_fermion * 2;

    std::cout << "Running trajectory at eps = " << eps_full
              << " (" << n_steps_full << " steps)..." << std::endl;
    TrajectoryResult res_full = run_fermion_trajectory(gauge_field_initial, phi_field,
                                                        P_field_initial, eps_full, n_steps_full);

    std::cout << "Running trajectory at eps = " << eps_half
              << " (" << n_steps_half << " steps)..." << std::endl;
    TrajectoryResult res_half = run_fermion_trajectory(gauge_field_initial, phi_field,
                                                        P_field_initial, eps_half, n_steps_half);

    double abs_dH_full = std::abs(res_full.delta_H);
    double abs_dH_half = std::abs(res_half.delta_H);
    double ratio = (abs_dH_half > 1e-300) ? abs_dH_full / abs_dH_half
                                           : std::numeric_limits<double>::infinity();

    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "  |Delta H| at eps         : " << abs_dH_full << std::endl;
    std::cout << "  |Delta H| at eps/2        : " << abs_dH_half << std::endl;
    std::cout << "  Ratio (expect ~4 if force is correct): " << ratio << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    if (ratio > 3.5 && ratio < 4.5) {
        std::cout << "Result: PASS. Fermion force O(eps^2) scaling confirmed." << std::endl;
    } else if (ratio > 2.0) {
        std::cout << "Result: WARNING. Check CG solver tolerance (tol = " << tol << ")." << std::endl;
    } else {
        std::cout << "Result: FAIL. Fermion force is not the true gradient of S_f." << std::endl;
    }
    std::cout << "----------------------------------------------\n" << std::endl;
}



// 1. Force computation containing ONLY the gauge contribution
std::vector<double> compute_gauge_only_force_field(const Link_array& arr, double beta) {   
    std::vector<double> force_field(conjugate_field_array_size, 0.0);
    int V = tables.V;

    #pragma omp parallel for collapse(2) 
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};

            // Evaluate gauge force only
            SU3 F_gauge = compute_gauge_force_at_link(arr, link_index_array, beta);
            set_link_SU3(force_field, link_index_array, F_gauge); 
        }
    }
    return force_field;
}

// 2. Local Leapfrog Half-Step (Gauge Only)
void local_initial_half_step_gauge(std::vector<double>& P_field, double beta,
                                   const Link_array& arr, double eps) {
    std::vector<double> Force_field = compute_gauge_only_force_field(arr, beta);

    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; ++i) {
        P_field[i] -= eps * 0.5 * Force_field[i];
    }
}

// 3. Local Leapfrog Intermediate Steps (Gauge Only)
void local_leapfrog_steps_gauge(std::vector<double>& P_field, double beta,
                                Link_array& arr, double eps, int n_steps) {
    int V = tables.V;
    for (int n = 1; n < n_steps; n++) {
        #pragma omp parallel for collapse(2)
        for (int s = 0; s < V; ++s) {
            for (int d = 0; d < 4; ++d) {
                const lattice_index& site = tables.site_coords[s];
                link_index link_index_array = {site[0], site[1], site[2], site[3], d};
                SU3 U_pre = get_SU3_at_link(arr, link_index_array);
                SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
                U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
                set_link_SU3(arr, link_index_array, U_pre);
            }
        }
        std::vector<double> force_field = compute_gauge_only_force_field(arr, beta);

        #pragma omp parallel for
        for (int i = 0; i < conjugate_field_array_size; i++) {
            P_field[i] -= eps * force_field[i];
        }
    }
}

// 4. Local Leapfrog Final Step (Gauge Only)
void local_final_step_gauge(std::vector<double>& P_field, double beta,
                            Link_array& arr, double eps) {
    int V = tables.V;
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < V; ++s) {
        for (int d = 0; d < 4; ++d) {
            const lattice_index& site = tables.site_coords[s];
            link_index link_index_array = {site[0], site[1], site[2], site[3], d};
            SU3 U_pre = get_SU3_at_link(arr, link_index_array);
            SU3 P_pre = eps * get_SU3_at_link(P_field, link_index_array);
            U_pre = numerically_stable_matrix_exponential(P_pre) * U_pre;
            set_link_SU3(arr, link_index_array, U_pre);
        }
    }
    std::vector<double> force_field = compute_gauge_only_force_field(arr, beta);

    #pragma omp parallel for
    for (int i = 0; i < conjugate_field_array_size; i++) {
        P_field[i] -= eps * 0.5 * force_field[i];
    }
}

// Runs one trajectory tracking H_gauge = Tr(P^2) + S_gauge
TrajectoryResult run_gauge_trajectory(Link_array gauge_field, double beta,
                                      std::vector<double> P_field,
                                      double eps, int n_steps) {
    double initial_Tr_P2 = compute_Tr_P2(P_field);
    double initial_gauge_action = compute_action(gauge_field, beta);
    
    // Hamiltonian strictly includes kinetic + gauge action
    double H_initial = initial_Tr_P2 + initial_gauge_action;

    local_initial_half_step_gauge(P_field, beta, gauge_field, eps);
    local_leapfrog_steps_gauge(P_field, beta, gauge_field, eps, n_steps);
    local_final_step_gauge(P_field, beta, gauge_field, eps);

    double final_Tr_P2 = compute_Tr_P2(P_field);
    double final_gauge_action = compute_action(gauge_field, beta);
    
    double H_final = final_Tr_P2 + final_gauge_action;

    return { H_final - H_initial, H_initial, H_final };
}

// Executes the gauge-only scaling test
void run_gauge_force_scaling_test(const Link_array& gauge_field_initial, double beta) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "       GAUGE-ONLY FORCE SCALING TEST          " << std::endl;
    std::cout << "==============================================\n" << std::endl;

    std::vector<double> P_field_initial = generate_conjugate_field_configuration();

    double eps_full = epsilon;
    int n_steps_full = number_of_steps_fermion;

    double eps_half = epsilon * 0.5;
    int n_steps_half = number_of_steps_fermion * 2;

    std::cout << "Running trajectory at eps = " << eps_full
              << " (" << n_steps_full << " steps)..." << std::endl;
    TrajectoryResult res_full = run_gauge_trajectory(gauge_field_initial, beta,
                                                      P_field_initial, eps_full, n_steps_full);

    std::cout << "Running trajectory at eps = " << eps_half
              << " (" << n_steps_half << " steps)..." << std::endl;
    TrajectoryResult res_half = run_gauge_trajectory(gauge_field_initial, beta,
                                                      P_field_initial, eps_half, n_steps_half);

    double abs_dH_full = std::abs(res_full.delta_H);
    double abs_dH_half = std::abs(res_half.delta_H);
    double ratio = (abs_dH_half > 1e-300) ? abs_dH_full / abs_dH_half
                                           : std::numeric_limits<double>::infinity();

    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "  |Delta H| at eps         : " << abs_dH_full << std::endl;
    std::cout << "  |Delta H| at eps/2        : " << abs_dH_half << std::endl;
    std::cout << "  Ratio (expect ~4 if force is correct): " << ratio << std::endl;
    std::cout << "----------------------------------------------\n" << std::endl;

    if (ratio > 3.5 && ratio < 4.5) {
        std::cout << "Result: PASS. Gauge force O(eps^2) scaling confirmed." << std::endl;
    } else {
        std::cout << "Result: FAIL. Gauge force is not the true gradient of S_g." << std::endl;
    }
    std::cout << "----------------------------------------------\n" << std::endl;
}