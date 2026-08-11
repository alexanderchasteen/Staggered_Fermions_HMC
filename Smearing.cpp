    #include "Smearing.h"
    #include "SU3_Sampling.h"
    #include "Parameters.h"
    #include "Lattice.h"    
    #include <Eigen/Dense>
    #include <complex>
    #include "Fermions.h"
    #include <iostream>





int flat_index_smearing_3_tensor(const int mu, const int nu, const int rho, const int x, const int y, const int z, const int t) {
    return ((4 * mu + nu) * 4 + rho)*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size + x*Spatial_Size*Spatial_Size*temporal_size + y*Spatial_Size*temporal_size + z*temporal_size + t;
}

std::array<int, 7> tensor_index_smearing_3_tensor(int flat_index) {
    int t = flat_index % temporal_size;
    flat_index /= temporal_size;
    int z = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int y = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int x = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int rho = flat_index % 4;
    flat_index /= 4;
    int nu = flat_index % 4;
    flat_index /= 4;
    int mu = flat_index % 4;

    return {mu,nu,rho,x,y,z,t};
}

int flat_index_smearing_2_tensor(const int mu, const int nu, const int x, const int y, const int z, const int t) {
    return ((4 * mu + nu)*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size + x*Spatial_Size*Spatial_Size*temporal_size + y*Spatial_Size*temporal_size + z*temporal_size + t);
}

std::array<int, 6> tensor_index_smearing_2_tensor(int flat_index) {
    int t = flat_index % temporal_size;
    flat_index /= temporal_size;
    int z = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int y = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int x = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int nu = flat_index % 4;
    flat_index /= 4;
    int mu = flat_index % 4;

    return {mu,nu,x,y,z,t};
}


int flat_index_smearing_1_tensor(const int mu, const int x, const int y, const int z, const int t) {
    return ((mu)*Spatial_Size*Spatial_Size*Spatial_Size*temporal_size + x*Spatial_Size*Spatial_Size*temporal_size + y*Spatial_Size*temporal_size + z*temporal_size + t);
}

std::array<int, 5> tensor_index_smearing_1_tensor(int flat_index) {
    int t = flat_index % temporal_size;
    flat_index /= temporal_size;
    int z = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int y = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int x = flat_index % Spatial_Size;
    flat_index /= Spatial_Size;
    int mu = flat_index % 4;

    return {mu,x,y,z,t};
}


std::vector<SU3> apply_step_1(const Link_array& U) {
    std::vector<SU3> Gamma1_mu_nu_rho(smearing_size_3_index);
    // index in the 64 is given by (4*mu+nu)*4+rho, where mu,nu,rho are the directions of the staples
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                            for (int nu = 0; nu < 4; nu++) {
                                for (int rho = 0; rho < 4; rho++) {
                                    SU3 Gamma_1;
                                    Gamma_1.setZero(); 
                                    for (int sigma = 0; sigma < 4; sigma++) {
                                        if (mu != sigma &&nu != sigma && rho != sigma) {
                                            lattice_index lattice_index_array;
                                            lattice_index_array[0] = x;
                                            lattice_index_array[1] = y;
                                            lattice_index_array[2] = z;
                                            lattice_index_array[3] = t;
                                            SU3 up_part = get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],sigma});
                                            moveup(lattice_index_array, sigma);
                                            up_part = up_part * get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],mu});
                                            movedown(lattice_index_array, sigma);
                                            moveup(lattice_index_array, mu);
                                            up_part = up_part * get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],sigma}).adjoint();
                                            movedown(lattice_index_array, mu);
                                            movedown(lattice_index_array, sigma);
                                            SU3 down_part = get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],sigma}).adjoint();
                                            down_part = down_part * get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],mu});
                                            moveup(lattice_index_array, mu);
                                            down_part = down_part * get_SU3_at_link(U, {lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3],sigma});
                                            // moveup(lattice_index_array, sigma);
                                            // movedown(lattice_index_array, mu);
                                            // assert(lattice_index_array[0] == x && lattice_index_array[1] == y && lattice_index_array[2] == z && lattice_index_array[3] == t);
                                            Gamma_1 += up_part + down_part;
                                        }
                                    }
                                    int index = flat_index_smearing_3_tensor(mu, nu, rho, x, y, z, t);
                                    Gamma1_mu_nu_rho[index] = Gamma_1;
                                }
                            }
                        }
                    }
                }     
            }
        }
        return Gamma1_mu_nu_rho;
    }
    

SU3 projection(const SU3& matrix) {
    SU3 projected_matrix = 0.5 * (matrix - matrix.adjoint()) - (1.0 / 3.0) * (0.5 * (matrix - matrix.adjoint())).trace() * SU3::Identity();
    return projected_matrix;
}


std::vector<SU3> apply_step_2(std::vector<SU3> Gamma1_mu_nu_rho, const Link_array& U_array) {
    std::vector<SU3> V1_mu_nu_rho(smearing_size_3_index);
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                            for (int nu = 0; nu < 4; nu++) {
                                for (int rho = 0; rho < 4; rho++) {
                                    SU3 V1;
                                    SU3 U = get_SU3_at_link(U_array, {x,y,z,t,mu});
                                    V1 = Gamma1_mu_nu_rho[flat_index_smearing_3_tensor(mu, nu, rho, x, y, z, t)]*U.adjoint();
                                    V1 = projection(V1)*alpha3 * 0.5 ;
                                    V1 = numerically_stable_matrix_exponential(V1) * U;
                                    V1_mu_nu_rho[flat_index_smearing_3_tensor(mu, nu, rho, x, y, z, t)] = V1;
                                }
                            }
                       }
                    }
                }
            }
        }
    return V1_mu_nu_rho;
}
 
std::vector<SU3> apply_step_3(std::vector<SU3> V1_mu_nu_rho) {
    std::vector<SU3> Gamma2_mu_nu(smearing_size_2_index);
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                            for (int nu = 0; nu < 4; nu++) {
                                SU3 Gamma_2;
                                Gamma_2.setZero(); 
                                for (int sigma = 0; sigma < 4; sigma++) {
                                    if (mu != sigma && nu != sigma) {
                                        lattice_index lattice_index_array;
                                        lattice_index_array[0] = x;
                                        lattice_index_array[1] = y;
                                        lattice_index_array[2] = z;
                                        lattice_index_array[3] = t;


                                        SU3 up_part = V1_mu_nu_rho[flat_index_smearing_3_tensor(sigma, mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                        moveup(lattice_index_array, sigma);

                                        // FIX: Middle link is in dir 'mu', excluded 'nu' and 'sigma'. So indices are (mu, nu, sigma).
                                        up_part = up_part * V1_mu_nu_rho[flat_index_smearing_3_tensor(mu, nu, sigma, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                        movedown(lattice_index_array, sigma);
                                        moveup(lattice_index_array, mu);

                                        up_part = up_part * V1_mu_nu_rho[flat_index_smearing_3_tensor(sigma, mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])].adjoint();
                                        movedown(lattice_index_array, mu);

                                        // --- DOWN STAPLE ---
                                        movedown(lattice_index_array, sigma);
                                        SU3 down_part = V1_mu_nu_rho[flat_index_smearing_3_tensor(sigma, mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])].adjoint();

                                        down_part = down_part * V1_mu_nu_rho[flat_index_smearing_3_tensor(mu, nu, sigma, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                        moveup(lattice_index_array, mu);

                                        down_part = down_part * V1_mu_nu_rho[flat_index_smearing_3_tensor(sigma, mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                       

                                        Gamma_2 += up_part + down_part;
                                    }
                                }
                                int index = flat_index_smearing_2_tensor(mu, nu, x, y, z, t);
                                Gamma2_mu_nu[index] = Gamma_2;
                            }
                        }
                    }
                }
            }
        }
        return Gamma2_mu_nu;
    }



std::vector<SU3> apply_step_4(std::vector<SU3> Gamma2_mu_nu, const Link_array& U_array) {
    std::vector<SU3> V2_mu_nu(smearing_size_2_index);
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                            for (int nu = 0; nu < 4; nu++) {
                                SU3 V2;
                                SU3 U = get_SU3_at_link(U_array, {x,y,z,t,mu});
                                V2 = Gamma2_mu_nu[flat_index_smearing_2_tensor(mu, nu, x, y, z, t)]*U.adjoint();
                                V2 = projection(V2)*alpha2 * 0.5 ;
                                V2 = numerically_stable_matrix_exponential(V2) * U;
                                V2_mu_nu[flat_index_smearing_2_tensor(mu, nu, x, y, z, t)] = V2;
                            }
                       }
                    }
                }
            }
        }
    return V2_mu_nu;
}

std::vector<SU3> apply_step_5(std::vector<SU3> V2_mu_nu){
    std::vector<SU3> Gamma_3_mu(smearing_size_1_index);
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                            SU3 Gamma_3;
                            for (int nu = 0; nu < 4; nu++) {
                                if (mu != nu) {
                                    lattice_index lattice_index_array;
                                    lattice_index_array[0] = x;
                                    lattice_index_array[1] = y;
                                    lattice_index_array[2] = z;
                                    lattice_index_array[3] = t;
                                    SU3 up_part = V2_mu_nu[flat_index_smearing_2_tensor(nu, mu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                    moveup(lattice_index_array, nu);
                                    up_part = up_part * V2_mu_nu[flat_index_smearing_2_tensor(mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                    movedown(lattice_index_array, nu);
                                    moveup(lattice_index_array, mu);
                                    up_part = up_part * V2_mu_nu[flat_index_smearing_2_tensor(nu, mu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])].adjoint();
                                    movedown(lattice_index_array, mu);

                                    // --- DOWN STAPLE ---
                                    movedown(lattice_index_array, nu);
                                    SU3 down_part = V2_mu_nu[flat_index_smearing_2_tensor(nu, mu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])].adjoint();
                                    down_part = down_part * V2_mu_nu[flat_index_smearing_2_tensor(mu, nu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];
                                    moveup(lattice_index_array, mu);
                                    down_part = down_part * V2_mu_nu[flat_index_smearing_2_tensor(nu, mu, lattice_index_array[0],lattice_index_array[1],lattice_index_array[2],lattice_index_array[3])];


                                
                                    Gamma_3 += up_part + down_part;
                                }
                            }
                            int index = flat_index_smearing_1_tensor(mu, x, y, z, t);
                            Gamma_3_mu[index] = Gamma_3;
                        }
                    }
                }
            }
        }
    return Gamma_3_mu;
    }
    
    
void apply_step_6(std::vector<SU3> Gamma_3_mu, Link_array& U_array) {
    std::vector<SU3> V_mu(smearing_size_1_index);
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                for (int t = 0; t < temporal_size; t++) {
                    for (int mu = 0; mu < 4; mu++) {
                        SU3 V3;
                        SU3 U = get_SU3_at_link(U_array, {x,y,z,t,mu});
                        V3 = Gamma_3_mu[flat_index_smearing_1_tensor(mu, x, y, z, t)]*U.adjoint();
                        V3 = projection(V3)*alpha1 * 0.5 ;
                        V3 = numerically_stable_matrix_exponential(V3) * U;
                        set_link_SU3(U_array, {x,y,z,t,mu}, V3);
                    }
                }
            }
        }
    }
    return;
}


void Apply_HEX_smearing(Link_array& U_array) {
    std::vector<SU3> Gamma1_mu_nu_rho = apply_step_1(U_array);
    std::vector<SU3> V1_mu_nu_rho = apply_step_2(Gamma1_mu_nu_rho, U_array);
    std::vector<SU3> Gamma2_mu_nu = apply_step_3(V1_mu_nu_rho);
    std::vector<SU3> V2_mu_nu = apply_step_4(Gamma2_mu_nu, U_array);
    std::vector<SU3> Gamma3_mu = apply_step_5(V2_mu_nu);
    apply_step_6(Gamma3_mu, U_array);
}

void Apply_4HEX_smearing(Link_array& U_array) {
    for (int i = 0; i < 4; i++) {
        Apply_HEX_smearing(U_array);
    }
    return;
}

std::vector<Link_array> Apply_4HEX_smearing_with_history(const Link_array& U_thin) {
    std::vector<Link_array> history;
    history.push_back(U_thin);
    
    Link_array U_current = U_thin;
    for (int step = 0; step < 4; ++step) {
        Apply_HEX_smearing(U_current); // Modifies U_current to the next level
        history.push_back(U_current);
    }
    return history;
}

