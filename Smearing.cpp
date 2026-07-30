    #include "Smearing.h"
    #include "SU3_Sampling.h"
    #include "Parameters.h"
    #include "Lattice.h"    
    #include <iostream>


    // STILL UNDER WORKS, LOTS OF BUGS

    double sinx_over_x_stable(double x) {
        if (std::abs(x) <= 0.05) {
            return 1.0-1/6*x*x*(1-1/20*x*x*(1- 1/42*x*x)); // Use the limit value for small x
        } else {
            return std::sin(x) / x;
        }
    }


    SU3 numerically_stable_matrix_exponential(matrix_3by3 Q){
        matrix_3by3 Q_cubed = Q*Q*Q;
        matrix_3by3 Q_squared = Q*Q;
        double c0 = 1/3*(Q_cubed.trace().real());

        if (Q_cubed.trace().imag() != 0) {
            std::cerr << "Error: Q cubed matrix has non-zero imaginary trace." << std::endl;
            // Placeholder return value
            return;
        }

        double c1 = 1/2*(Q_squared.trace().real());
        if (c1 < 0) {
            std::cerr << "Error: Q squared matrix has negative trace." << std::endl;
            // Placeholder return value
            return;
        }
        std::complex<double> f0, f1, f2;
        if (c0>=0){
            double c0_max = 2*std::sqrt(c1*c1*c1/27);
            double theta = std::acos(c0/c0_max);
            double w = std::sqrt(c1)*std::sin(theta/3);
            double u = std::sqrt(c1/3)*std::cos(theta/3);

    
            // computing the h functions
            std::complex<double> h0 = (u*u-w*w)*std::exp(std::complex<double>(0,2)*u) + std::exp(std::complex<double>(0,-1)*u)*(8*u*u*std::cos(w)+std::complex<double>(0,2)*u*(3*u*u+w*w)*sinx_over_x_stable(w));
            std::complex<double> h1 = 2.0*u*std::exp(std::complex<double>(0,2)*u)- std::exp(std::complex<double>(0,-1)*u)*(2*u*std::cos(w)+std::complex<double>(0,-1)*(3*u*u-w*w)*sinx_over_x_stable(w));
            std::complex<double> h2 = std::exp(std::complex<double>(0,2)*u)- std::exp(std::complex<double>(0,-1)*u)*(std::cos(w)+std::complex<double>(0,3)*u*sinx_over_x_stable(w));
            // computing the f functions
            double denom = 9*u*u - w*w;
            f0 = h0/denom;
            f1 = h1/denom;
            f2 = h2/denom;
        }

        if (c0<0){
            c0 = -c0;
            double c0_max = 2*std::sqrt(c1*c1*c1/27);
            double theta = std::acos(c0/c0_max);
            double w = std::sqrt(c1)*std::sin(theta/3);
            double u = std::sqrt(c1/3)*std::cos(theta/3);

    
            // computing the h functions
            std::complex<double> h0 = (u*u-w*w)*std::exp(std::complex<double>(0,2)*u) + std::exp(std::complex<double>(0,-1)*u)*(8*u*u*std::cos(w)+std::complex<double>(0,2)*u*(3*u*u+w*w)*sinx_over_x_stable(w));
            std::complex<double> h1 = 2.0*u*std::exp(std::complex<double>(0,2)*u)- std::exp(std::complex<double>(0,-1)*u)*(2*u*std::cos(w)+std::complex<double>(0,-1)*(3*u*u-w*w)*sinx_over_x_stable(w));
            std::complex<double> h2 = std::exp(std::complex<double>(0,2)*u)- std::exp(std::complex<double>(0,-1)*u)*(std::cos(w)+std::complex<double>(0,3)*u*sinx_over_x_stable(w));
            // computing the f functions
            double denom = 9*u*u - w*w;
            f0 = std::conj(h0/denom);
            f1 = -std::conj(h1/denom);
            f2 = std::conj(h2/denom);
        }

        SU3 exponential_of_iQ = f0*Eigen::Matrix3d::Identity() + f1*Q + f2*Q_squared;
        return exponential_of_iQ; 
    } 


    SU3 compute_weighted_staplesum_at_link(const Link_array& arr, const link_index& link_index_array){
        lattice_index lattice_index_array = {link_index_array[0], link_index_array[1], link_index_array[2], link_index_array[3]};
        int d=link_index_array[4];
        link_index local_link_index_array;
        SU3 staple;
        staple.setZero();
        SU3 staple_sum; 
        staple_sum.setZero();
        for (int dperp=0;dperp<4;dperp++){
            if (dperp!=d){
                // make temperoray lattice_index_array storage because each time we loop we want to start back at the beginning point 
                lattice_index tmp = lattice_index_array;
                // Bottom staple
            
                //Let us grab U_{ν}(n-ν)
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array); 
                
                //Let us grab U_{-μ}(n+μ-ν)=U_{μ}^dagger(n-ν)
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                //Let us grab U_{-ν}(n+μ)=U_{ν}^dagger}(n+μ-ν)
                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;
                
                staple_sum += staple;

                // Top staple
                //Let us grab U_{-ν}(n+ν)=U_{ν}(n)^dagger
                moveup(tmp,dperp);
                movedown(tmp,d);
                
                assert(tmp == lattice_index_array);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint(); 
                
                //Let us grab U_{-μ}(n+μ+ν)
                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                //Let us grab U_{ν}(n+μ)
                moveup(tmp,d);
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;

                staple_sum += staple;
                movedown(tmp,d);
                assert(tmp == lattice_index_array);


                staple_sum *= rho(d,dperp);
            }
        }
        return staple_sum;
    }

    matrix_3by3 compute_Q_matrix(Link_array& arr, const link_index& link_index_array){
        SU3 U = get_SU3_at_link(arr,link_index_array);
        SU3 C = compute_weighted_staplesum_at_link(arr,link_index_array);
        matrix_3by3 Omega = C*U.adjoint();
        matrix_3by3 Q = (Omega - Omega.adjoint())/(2.0*std::complex<double>(0,1)) + (Omega.adjoint()-Omega).trace()/(6.0*std::complex<double>(0,1))*Eigen::Matrix3d::Identity();
        return Q;
    }


    void smear_single_link(Link_array& arr, const link_index& link_index_array, double rho){
        SU3 U = get_SU3_at_link(arr,link_index_array);
        matrix_3by3 Q = compute_Q_matrix(arr,link_index_array);
        SU3 exp_iQ = numerically_stable_matrix_exponential(std::complex<double>(0,rho)*Q);
        SU3 smeared_link = exp_iQ*U;
        // Do something with the smeared link, e.g., update the link in the array
        set_link_SU3(arr, link_index_array, smeared_link);
    }

    void smear_lattice(Link_array& arr, double rho){
        for (int x=0;x<Spatial_Size;x++){
            for (int y=0;y<Spatial_Size;y++){
                for (int z=0;z<Spatial_Size;z++){
                    for (int t=0;t<temporal_size;t++){
                        lattice_index lattice_index_array = {x,y,z,t};
                        for (int d=0;d<4;d++){
                            link_index link_index_array = combine_lattice_index_with_direction(lattice_index_array,d);
                            smear_single_link(arr,link_index_array,rho);
                        }
                    }
                }
            }
        }
    }

    void smear_lattice_multiple_times(Link_array& arr, double rho, int num_smearings){
        for (int i=0;i<num_smearings;i++){
            smear_lattice(arr,rho);
        }
    }

    