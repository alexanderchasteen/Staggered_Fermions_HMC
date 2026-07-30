    #include "Lattice.h"
    #include "SU3_Sampling.h"
    #include "Parameters.h"
    #include <iostream>



    // Gives the index in an array given the indices in our 7 index tensor for the lattice_data information (at a singular beta)
    // 4 of the indices are the spatiotemporal positions on the lattice, the 5th index the (positive) link direction (of 4) 
    // 6th runs over each of the 18 indices of SU(3) (9 for real, 9 for imaginary). 
    int flat_index(tensor_index tensor_index_array) {
        int i1=tensor_index_array[0];
        int i2=tensor_index_array[1];
        int i3=tensor_index_array[2];
        int i4=tensor_index_array[3];
        int i5=tensor_index_array[4];
        int i6=tensor_index_array[5];
        return (((((i1 * Spatial_Size + i2) * Spatial_Size + i3) * temporal_size + i4) *4  + i5) * 18 + i6);
    }


    // Inverts the flat index back to the position on the tensor
    tensor_index tensor_index_array(int index){
        int i6= index % 18;
        index /= 18;
        int i5 = index % 4;
        index /= 4;
        int i4 = index % temporal_size;
        index /= temporal_size;
        int i3 = index % Spatial_Size;
        index /= Spatial_Size;
        int i2 = index % Spatial_Size;
        index /= Spatial_Size;
        int i1 = index;
        return {i1,i2,i3,i4,i5,i6};
    }


    void set_array_value(Link_array& arr, tensor_index tensor_index_array, double value) {
        int idx = flat_index(tensor_index_array);
        arr[idx] = value;
        return;
    }

    double get_array_value(const Link_array& arr, tensor_index tensor_index_array) {
        int idx = flat_index(tensor_index_array);
        return arr[idx];
    }


    // Now we want to use these functions to get the matrix at a certain link along the lattice. This will take in a 5 index tensor location and we will loop over 
    // the 18 possibilities for the 6th array. Then we will put these in a matrix.


    // First let us come up with functions that both FLATTEN, and MATRIXify information. That is, from SU(3) to array with 18 components and vice versa

    SU3_array SU3_matrix_to_array(const SU3& matrix){
        SU3_array array={};
        int index = 0;
        for (int i=0;i<3; i++){
            for (int j=0;j<3;j++){
                double real_part = matrix(i,j).real();
                array[index] = real_part;
                index += 1;
                double imaginary_part = matrix(i,j).imag();
                array[index] = imaginary_part;
                index += 1;
            }
        }
        return array;
    }

    SU3 SU3_array_to_matrix(const SU3_array& array){
        SU3 SU3_matrix;
        int index=0;
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                double real_part=array[index];
                index += 1;
                double imaginary_part=array[index];
                index += 1;
                SU3_matrix(i,j)=complex (real_part,imaginary_part);
            }
        }
        return SU3_matrix;
    }


    // Now we want code that gets the SU3 matrix at a given link. The matrix information is stored as an SU3_array and we want to convert it to a matrix. 
    // We use our array --> matrix converter 

    // SU3 get_SU3_at_link(const Link_array&arr, link_index link_index_array){
    //     // extract the real component
    //     // Initialize a tensor index_array. The algorithm is to append the value 0<=i<=17 int to the link_index 
    //     // Put the link location in the first 5 elements of the total tensor index array
    //     tensor_index tensor_index_array = {};
    //     for (int i=0; i<5; i++){
    //         tensor_index_array[i] = link_index_array[i];
    //     }

    //     // Put the 6th link location in the last element of the total tensor index array (for some general index). 
    //     // But let us find the real and the imaginary index at the same time and add it Then find this element and add it to an array
    //     SU3_array SU3array={};
    //     for (int i=0; i<18;i+=2){
    //         // Get the real part
    //         tensor_index_array[5] = i;
    //         double real_part=get_array_value(arr,tensor_index_array);
    //         SU3array[i]=real_part;

    //         // Get the imaginary part
    //         tensor_index_array[5] = i+1;
    //         double imaginary_part=get_array_value(arr,tensor_index_array);
    //         SU3array[i+1]=imaginary_part;
    //     }
    //     SU3 SU3_matrix=SU3_array_to_matrix(SU3array);
    //     return SU3_matrix;
    // }

    // void set_link_SU3(Link_array& arr, link_index link_index_array, const SU3& SU3_matrix){
    //     tensor_index tensor_index_array = {};
    //     for (int i=0; i<5; i++){
    //         tensor_index_array[i] = link_index_array[i];
    //     }
    //     SU3_array SU3array = SU3_matrix_to_array(SU3_matrix);
    //     for (int i=0; i<18;i++){
    //         tensor_index_array[5] = i;
    //         set_array_value(arr, tensor_index_array, SU3array[i]);
    //     }
    //     return;
    // }


    SU3 get_SU3_at_link(const Link_array& arr, link_index link_index_array) {
        // Calculate base offset directly for all 18 elements of this link
        int base_idx = (((((link_index_array[0] * Spatial_Size + link_index_array[1]) 
                        * Spatial_Size + link_index_array[2]) 
                        * temporal_size + link_index_array[3]) 
                        * 4 + link_index_array[4]) * 18);

        SU3 SU3_matrix;
        int idx = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                SU3_matrix(i, j) = complex(arr[base_idx + idx], arr[base_idx + idx + 1]);
                idx += 2;
            }
        }
        return SU3_matrix;
    }

    void set_link_SU3(Link_array& arr, link_index link_index_array, const SU3& SU3_matrix) {
        // Calculate base offset directly for all 18 elements of this link
        int base_idx = (((((link_index_array[0] * Spatial_Size + link_index_array[1]) 
                        * Spatial_Size + link_index_array[2]) 
                        * temporal_size + link_index_array[3]) 
                        * 4 + link_index_array[4]) * 18);

        int idx = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                arr[base_idx + idx]     = SU3_matrix(i, j).real();
                arr[base_idx + idx + 1] = SU3_matrix(i, j).imag();
                idx += 2;
            }
        }
    }










    // Want to start our lattice cold. This means we want to initlaize all the links to I \in SU(3)
    // This is because the action is β/3 Σ_{n ∈ Λ} Σ_{μ < ν} Re(Tr(1-U_{μν}(n))) so at the identity S=0
    // Since in the path integral it is e^-S as the statisticall factor, this correlates to a 0 energy state whichh exists at high temperatures

    void cold_start_array(Link_array& arr){
        for (int i1 = 0; i1 < Spatial_Size; i1++)
        for (int i2 = 0; i2 < Spatial_Size; i2++)
        for (int i3 = 0; i3 < Spatial_Size; i3++)
        for (int i4 = 0; i4 < temporal_size; i4++)
        for (int i5 = 0; i5 < 4; i5++){
            link_index link_index_array = {i1,i2,i3,i4,i5};
            SU3 I;
            I.setIdentity();
            set_link_SU3(arr, link_index_array,I);
        }
        return;
    }


    void moveup(lattice_index& lattice_index_array, int link_direction){
        lattice_index_array[link_direction]+=1;
        if (link_direction==3){
            if (lattice_index_array[link_direction]>=temporal_size) lattice_index_array[link_direction] -= temporal_size;
        }
        else{
            if (lattice_index_array[link_direction]>=Spatial_Size) lattice_index_array[link_direction] -= Spatial_Size;
        }
        return;
    }


    void movedown(lattice_index& lattice_index_array, int link_direction){
        if (lattice_index_array[link_direction] == 0) {
            if (link_direction == 3)
                lattice_index_array[link_direction] = temporal_size - 1;
            else
                lattice_index_array[link_direction] = Spatial_Size - 1;
        } else {
            lattice_index_array[link_direction] -= 1;
        }
        return;
    }



    link_index combine_lattice_index_with_direction(const lattice_index& latice_index_array, int direction){
        link_index total_link_index_array;
        for (int i=0; i<4; i++){
            total_link_index_array[i]=latice_index_array[i];
        }
        total_link_index_array[4]=direction;
        return total_link_index_array;
    }
    

    SU3 compute_staple_sum_at_link(const Link_array& arr, const link_index& link_index_array){
        // The staple sum is given in gattringer. However note that for the bottom part of the loop, this is the staple sum BUT the staple we compute is actually the dagger of the "correct" staple
        // This does not matter since we will take the real part and the trace. So it is equivalent to use this matrix. 
        // SO IMPORTANT NOTE this is physically unmeaningful without includiong the real part of the trace


        // Seperate into the lattice indices (4) and the direction of the link
        lattice_index lattice_index_array = {link_index_array[0], link_index_array[1], link_index_array[2], link_index_array[3]};
        int d=link_index_array[4];
        link_index local_link_index_array;

        // Takes care of the staplesum calculation
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
            }
        }
        return staple_sum;
    }

    SU3 compute_rectangle_staple_sum_at_link(const Link_array& arr, const link_index& link_index_array){
        // This will be used for the Symanzik imrpovement to the lattice action to get rid of o(a^6) lattice artifacts. 
        // See "Renormalization Group Flow of SU(3) Gauge Theory" by Y. Iwasaki, Nucl. Phys. B258 (1985) 141-156 for more details. 
        lattice_index lattice_index_array = {link_index_array[0], link_index_array[1], link_index_array[2], link_index_array[3]};
        int d=link_index_array[4];
        link_index local_link_index_array;

        // Takes care of the staplesum calculation
        SU3 staple;
        staple.setZero();
        SU3 rectangle_staple_sum; 
        rectangle_staple_sum.setZero();
        for (int dperp=0;dperp<4;dperp++){
            if (dperp!=d){
                lattice_index tmp = lattice_index_array;
        
                // Bottom rectangle: Type 1: The link is on the short side of the rectangle.
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array); 
                
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;

            
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;


                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;
                
                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;
                



                rectangle_staple_sum += staple;

                // Top rectangle: Type 2: The link is on the short side of the rectangle.
                moveup(tmp,dperp);
                movedown(tmp,d);
                
                assert(tmp == lattice_index_array);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint(); 
                

                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;

                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                moveup(tmp,d);
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;

                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;

                rectangle_staple_sum += staple;
                movedown(tmp,d);
                assert(tmp == lattice_index_array);


                // Weird rectangles where the link is on the long side of the rectangle.
                // Type 3 and Type 4: The link is on the long side of the rectangle. Looks like Z block tetris
                //Type 3: The rectangle is below the link
                movedown(tmp,dperp); 
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array);
                
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;
                
                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;

                moveup(tmp,dperp);
                movedown(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;

                movedown(tmp,d);
                rectangle_staple_sum += staple;
                assert(tmp == lattice_index_array);

                //Type 4: The rectangle is above the link
                movedown(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array);

                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;

                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;
                
                moveup(tmp,d);
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;
                movedown(tmp,d);
                rectangle_staple_sum += staple;
                assert(tmp == lattice_index_array);

                // Type 5 and Type 6: The link is on the long side of the rectangle. Looks like reverse Z block tetris
                //Type 5: The rectangle is below the link

                movedown(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array);

                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple =  get_SU3_at_link(arr,local_link_index_array)*staple;

                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;

                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;
                
                moveup(tmp,d); 
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;

                moveup(tmp,dperp);
                movedown(tmp,d);
                rectangle_staple_sum += staple;
                assert(tmp == lattice_index_array);

                //Type 6: The rectangle is above the link
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint();

                moveup(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple =  (get_SU3_at_link(arr,local_link_index_array).adjoint())*staple;  

                moveup(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array).adjoint()*staple;

                moveup(tmp,d);
                movedown(tmp,dperp);
                local_link_index_array = combine_lattice_index_with_direction(tmp,dperp);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;
                movedown(tmp,d);
                local_link_index_array = combine_lattice_index_with_direction(tmp,d);
                staple = get_SU3_at_link(arr,local_link_index_array)*staple;
                movedown(tmp,d);
                rectangle_staple_sum += staple;
                assert(tmp == lattice_index_array);

            }
        }
        return rectangle_staple_sum;
    }


    SU3 compute_full_staple_sum_symanzik_at_link(const Link_array& arr, const link_index& link_index_array){
        SU3 staple_sum = c_11 * compute_staple_sum_at_link(arr,link_index_array);
        SU3 rectangle_staple_sum = c_12 * compute_rectangle_staple_sum_at_link(arr,link_index_array);
        SU3 full_staple_sum = staple_sum + rectangle_staple_sum;
        return full_staple_sum;
    }

    SU2 R_block(const SU3& SU3_matrix){
        SU2 block = SU3_matrix.block<2,2>(0,0);
        return block;
    }

    SU3 R_block_to_SU3(const SU2& SU2_matrix){
        SU3 matrix;
        matrix(0,0) = SU2_matrix(0,0);
        matrix(0,1) = SU2_matrix(0,1);
        matrix(0,2) = 0.0;
        matrix(1,0) = SU2_matrix(1,0);
        matrix(1,1) = SU2_matrix(1,1);
        matrix(1,2) = 0.0;
        matrix(2,0) = 0.0;
        matrix(2,1) = 0.0;
        matrix(2,2) = 1.0; 
        return matrix;    
    }

    SU2 S_block(const SU3& SU3_matrix){
        SU2 block;
        block(0,0) = SU3_matrix(0,0);
        block(0,1) = SU3_matrix(0,2);
        block(1,0) = SU3_matrix(2,0);
        block(1,1) = SU3_matrix(2,2);
        return block;
    }

    SU3 S_block_to_SU3(const SU2& SU2_matrix){
        SU3 matrix;
        matrix(0,0) = SU2_matrix(0,0);
        matrix(0,1) = 0;
        matrix(0,2) = SU2_matrix(0,1);
        matrix(1,0) = 0.0;
        matrix(1,1) = 1.0;
        matrix(1,2) = 0.0;
        matrix(2,0) = SU2_matrix(1,0);
        matrix(2,1) = 0.0;
        matrix(2,2) = SU2_matrix(1,1); 
        return matrix;    
    }

    SU2 T_block(const SU3& SU3_matrix){
        SU2 block = SU3_matrix.block<2,2>(1,1);
        return block;
    }

    SU3 T_block_to_SU3(const SU2& SU2_matrix){
        SU3 matrix;
        matrix(0,0) = 1.0;
        matrix(0,1) = 0;
        matrix(0,2) = 0;
        matrix(1,0) = 0.0;
        matrix(1,1) = SU2_matrix(0,0);
        matrix(1,2) = SU2_matrix(0,1);
        matrix(2,0) = 0.0;
        matrix(2,1) = SU2_matrix(1,0);
        matrix(2,2) = SU2_matrix(1,1); 
        return matrix;    
    }

    SU2 R_block_to_2by2_unitary(const SU3& SU3_matrix){
        // Note that since we want to sample our random matrix with respect to the 2 by 2 block that we have to project to a unitary matrix, we need to compute the determinant of the projection
        // The projection conserves the real trace so it is purely mathematical trick


        // First extract the two by two block
        SU2 block = R_block(SU3_matrix);
        SU2 Hermetian_part = (block+block.adjoint())/2.0;
        SU2 anti_hermetian_part = (block-block.adjoint())/2.0;
        complex b0 = (Hermetian_part.trace())/2.0;
        complex b1 = (anti_hermetian_part*Pauli::sigma_x).trace() / 2.0;
        complex b2 = (anti_hermetian_part*Pauli::sigma_y).trace() / 2.0;
        complex b3 = (anti_hermetian_part*Pauli::sigma_z).trace() / 2.0;
        SU2 matrix = b0*Pauli::I + b1*Pauli::sigma_x + b2*Pauli::sigma_y +b3*Pauli::sigma_z;
        return matrix; 
    }
    // Similarly for S and T type

    SU2 S_block_to_2by2_unitary(const SU3& SU3_matrix){
        SU2 block = S_block(SU3_matrix);
        SU2 Hermetian_part = (block+block.adjoint())/2.0;
        SU2 anti_hermetian_part = (block-block.adjoint())/2.0;
        complex b0 = (Hermetian_part.trace())/2.0;
        complex b1 = (anti_hermetian_part*Pauli::sigma_x).trace() / 2.0;
        complex b2 = (anti_hermetian_part*Pauli::sigma_y).trace() / 2.0;
        complex b3 = (anti_hermetian_part*Pauli::sigma_z).trace() / 2.0;
        SU2 matrix = b0*Pauli::I + b1*Pauli::sigma_x + b2*Pauli::sigma_y +b3*Pauli::sigma_z;
        return matrix; 
    }

    SU2 T_block_to_2by2_unitary(const SU3& SU3_matrix){
        SU2 block = T_block(SU3_matrix);
        SU2 Hermetian_part = (block+block.adjoint())/2.0;
        SU2 anti_hermetian_part = (block-block.adjoint())/2.0;
        complex b0 = (Hermetian_part.trace())/2.0;
        complex b1 = (anti_hermetian_part*Pauli::sigma_x).trace() / 2.0;
        complex b2 = (anti_hermetian_part*Pauli::sigma_y).trace() / 2.0;
        complex b3 = (anti_hermetian_part*Pauli::sigma_z).trace() / 2.0;
        SU2 matrix = b0*Pauli::I + b1*Pauli::sigma_x + b2*Pauli::sigma_y +b3*Pauli::sigma_z;
        return matrix; 
    }



    SU3 type_R_heatbath(const Link_array& arr, link_index link_index_array, double beta, SU3 U, SU3 A){
        constexpr double small = 1e-14;
        // We want to compute the staple sum A once and we will feed into it
        // The link U to update ach time. However after a type R heatbath, we change the link U. So let us have this function
        // Give out the new link U and keep the same format for type S and T so that we can just after each update, get the new link,
        // use the already computed staplesum, and after we do T we will set the link to the last one. 

        // That is 
        // SU3 A = compute_staple_sum_at_link(arr, link_index_array);
        // SU3 U = get_SU3_at_link(arr, link_index_array);
        SU3 UA = U*A;
        SU2 W_block = R_block_to_2by2_unitary(UA);
        double a = std::sqrt(W_block.determinant().real());

        SU2 R_2by2;
        SU3 R_3by3;
        if (a <= small){
            R_2by2 = Random_SU2_generator();
            R_3by3 = R_block_to_SU3(R_2by2);
            U = R_3by3*U; 
        }

        else{
            W_block = W_block/a;
            double adjusted_beta = 2*beta/3;
            SU2 X = SU2_generator(a,adjusted_beta);
            R_2by2 = X*W_block.adjoint();
            R_3by3 = R_block_to_SU3(R_2by2);    
            U = R_3by3 * U; 
        }    
        return U ;
    }

    SU3 type_S_heatbath(const Link_array& arr, link_index link_index_array, double beta, SU3 U, SU3 A){
        constexpr double small = 1e-14;
        SU3 UA = U*A;
        SU2 W_block = S_block_to_2by2_unitary(UA);
        double a = std::sqrt(W_block.determinant().real());
        
        SU2 S_2by2;
        SU3 S_3by3;
        if (a <= small){
            S_2by2 = Random_SU2_generator();
            S_3by3 = S_block_to_SU3(S_2by2);
            U = S_3by3*U; 
        }

        else{
            W_block = W_block/a;
            double adjusted_beta = 2*beta/3;
            SU2 X = SU2_generator(a,adjusted_beta);
            S_2by2 = X*W_block.adjoint();
            S_3by3 = S_block_to_SU3(S_2by2);    
            U = S_3by3 * U; 
        }    
        return U ;
    }

    SU3 type_T_heatbath(const Link_array& arr, link_index link_index_array, double beta, SU3 U, SU3 A){
        constexpr double small = 1e-14;
        SU3 UA = U*A;
        SU2 W_block = T_block_to_2by2_unitary(UA);
        double a = std::sqrt(W_block.determinant().real());

        SU2 T_2by2;
        SU3 T_3by3;
        if (a <= small){
            T_2by2 = Random_SU2_generator();
            T_3by3 = T_block_to_SU3(T_2by2);
            U = T_3by3*U; 
        }

        else{
            W_block = W_block/a;
            double adjusted_beta = 2*beta/3;
            SU2 X = SU2_generator(a,adjusted_beta);
            T_2by2 = X*W_block.adjoint();
            T_3by3 = T_block_to_SU3(T_2by2);    
            U = T_3by3 * U; 
        }    
        return U ;
    }
    std::pair<double, double> single_link_heatbath(Link_array& arr, const link_index& link_index_array, double beta) {
        // 1. Store old link
        SU3 U_old = get_SU3_at_link(arr, link_index_array);
        
        // 2. Compute staples
        SU3 A1 = compute_staple_sum_at_link(arr, link_index_array);
        SU3 A2 = compute_rectangle_staple_sum_at_link(arr, link_index_array);
        SU3 A = c_11 * A1 + c_12 * A2;

        // 3. Update link across SU(2) subgroups
        SU3 U_new = U_old;
        U_new = type_R_heatbath(arr, link_index_array, beta, U_new, A); 
        U_new = type_S_heatbath(arr, link_index_array, beta, U_new, A);
        U_new = type_T_heatbath(arr, link_index_array, beta, U_new, A);
        
        // Save updated link back to array
        set_link_SU3(arr, link_index_array, U_new);

        // 4. Local trace quantities for the updated link
        // Plaquette contribution: 1/3 * Re(Tr(U * A1))
        double plaq_val = ((U_new * A1).trace().real()) / 3.0;

        // Symanzik gauge action per link: S_link = - (beta / 3) * Re(Tr(U * A))
        double action_val = - (beta / 3.0) * ((U_new * A).trace().real());

        return {plaq_val, action_val}; 
    }


    std::pair<double, double> heatbath_update(Link_array& arr, double beta) {
        double total_plaq_sum = 0.0;
        double total_action = 0.0;

        for (int i1 = 0; i1 < Spatial_Size; ++i1) {
            for (int i2 = 0; i2 < Spatial_Size; ++i2) {
                for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                    for (int i4 = 0; i4 < temporal_size; ++i4) {
                        for (int d = 0; d < 4; ++d) {
                            link_index link_index_array = {i1, i2, i3, i4, d};
                            
                            auto [plaq, action] = single_link_heatbath(arr, link_index_array, beta);
                            
                            total_plaq_sum += plaq;   // 1/3 Re Tr(U * A1)
                            total_action   += action; // - (beta/3) Re Tr(U * A)
                        }
                    }
                }
            }
        }

        double volume = static_cast<double>(Spatial_Size * Spatial_Size * Spatial_Size * temporal_size);
        double num_plaquettes = 6.0 * volume;

        // total_plaq_sum overcounts total plaquette trace sum by a factor of 4
        double avg_plaq = (total_plaq_sum / 4.0) / num_plaquettes;

        // total_action overcounts lattice action by a factor of 4
        double total_lattice_action = total_action / 4.0;
        double avg_action_density = total_lattice_action / volume;

        return {avg_plaq, avg_action_density};  
    }

    double compute_action(const Link_array& arr, double beta){
        double total_action = 0.0;
        for (int i1 = 0; i1 < Spatial_Size; ++i1) {
            for (int i2 = 0; i2 < Spatial_Size; ++i2) {
                for (int i3 = 0; i3 < Spatial_Size; ++i3) {
                    for (int i4 = 0; i4 < temporal_size; ++i4) {
                        for (int d = 0; d < 4; ++d) {
                            link_index link_index_array = {i1, i2, i3, i4, d};
                            SU3 U = get_SU3_at_link(arr, link_index_array);
                            SU3 A1 = compute_staple_sum_at_link(arr, link_index_array);
                            SU3 A2 = compute_rectangle_staple_sum_at_link(arr, link_index_array);
                            SU3 A = c_11 * A1 + c_12 * A2;
                            
                            // Division by 4.0 accounts for each plaquette/rectangle being 
                            // counted 4 times when iterating over all links.
                            double action = - (beta / 3.0) * ((U * A).trace().real()) / 4.0;
                            total_action += action;
                        }
                    }
                }
            }
        }
        return total_action;   
    }


   bool save_all_lattice_configs(const std::array<Link_array, CONFIG>& links_at_coupling, const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return false;
        }

        for (int i = 0; i < CONFIG; i++) {
            out.write(reinterpret_cast<const char*>(links_at_coupling[i].data()), 
                    links_at_coupling[i].size() * sizeof(double));
        }

        if (!out) {
            std::cerr << "Error writing complete binary data to: " << filename << "\n";
            return false;
        }
        return true;
    }

    // Loads all CONFIG lattices sequentially from a single binary file
    bool load_all_lattice_configs(std::array<Link_array, CONFIG>& links_at_coupling, const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "Error opening file for reading: " << filename << "\n";
            return false;
        }

        for (int i = 0; i < CONFIG; i++) {
            in.read(reinterpret_cast<char*>(links_at_coupling[i].data()), 
                    links_at_coupling[i].size() * sizeof(double));
        }

        if (!in) {
            std::cerr << "Error reading complete binary data from: " << filename << "\n";
            return false;
        }
        return true;
    }

complex poly_loop_at_spat_coord(const Link_array& arr, int x, int y, int z) {
    int T = 3;
    SU3 wl_matrix;
    wl_matrix.setIdentity();
    lattice_index lat = {x, y, z, 0};
    
    for (int i = 0; i < temporal_size; i++) {
        link_index idx = combine_lattice_index_with_direction(lat, T);
        wl_matrix = wl_matrix * get_SU3_at_link(arr, idx); // RIGHT multiply
        moveup(lat, T);
        // Periodic boundary handled by moveup
    }
    std::complex<double> tr = wl_matrix.trace();
    
    // Fix: Normalize by N_c = 3 for SU(3)
    return tr; 
}

// Helper function to precompute P(x, y, z) across the spatial grid ONCE
// Fix: Flattened into a 1D vector for optimal OpenMP cache locality
std::vector<complex> precompute_polyakov_grid(const Link_array& arr) {
    std::vector<complex> grid(Spatial_Size * Spatial_Size * Spatial_Size);
    
    #pragma omp parallel for collapse(3)
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                int idx = (x * Spatial_Size + y) * Spatial_Size + z;
                grid[idx] = poly_loop_at_spat_coord(arr, x, y, z);
            }
        }
    }
    return grid;
}

// Fast correlator calculation using precomputed P(x, y, z) grid
// Fix: Accepts r_squared to allow measuring off-axis correlation
complex correlator_over_fixed_distance_fast(const std::vector<complex>& P_grid, int r_squared) {
    // 1. Find valid displacement vectors (Delta_x, Delta_y, Delta_z) ONCE per r_squared
    struct Disp { int dx, dy, dz; };
    std::vector<Disp> displacements;
    
    // Calculate the maximum integer displacement that could contribute to r_squared
    int max_d = std::min(static_cast<int>(std::sqrt(r_squared)), Spatial_Size / 2);
    
    for (int dx = -max_d; dx <= max_d; dx++) {
        // Fix: Prevent double-counting exact L/2 periodic boundary reflections
        if (dx == Spatial_Size / 2 && Spatial_Size % 2 == 0) continue; 
        for (int dy = -max_d; dy <= max_d; dy++) {
            if (dy == Spatial_Size / 2 && Spatial_Size % 2 == 0) continue;
            for (int dz = -max_d; dz <= max_d; dz++) {
                if (dz == Spatial_Size / 2 && Spatial_Size % 2 == 0) continue;
                
                if (dx * dx + dy * dy + dz * dz == r_squared) {
                    displacements.push_back({dx, dy, dz});
                }
            }
        }
    }

    // If no valid integer coordinates sum to this squared distance, skip.
    if (displacements.empty()) return complex(0.0, 0.0);

    complex polyakov_data = 0.0;
    int N = 0;

    // 2. Loop over spatial grid m (L^3) and valid displacements only
    for (int x = 0; x < Spatial_Size; x++) {
        for (int y = 0; y < Spatial_Size; y++) {
            for (int z = 0; z < Spatial_Size; z++) {
                
                int idx_m = (x * Spatial_Size + y) * Spatial_Size + z;
                complex P_m = P_grid[idx_m];
                
                for (const auto& disp : displacements) {
                    // Fast bounds checking
                    int nx = x + disp.dx;
                    if (nx < 0) nx += Spatial_Size;
                    else if (nx >= Spatial_Size) nx -= Spatial_Size;

                    int ny = y + disp.dy;
                    if (ny < 0) ny += Spatial_Size;
                    else if (ny >= Spatial_Size) ny -= Spatial_Size;

                    int nz = z + disp.dz;
                    if (nz < 0) nz += Spatial_Size;
                    else if (nz >= Spatial_Size) nz -= Spatial_Size;

                    int idx_n = (nx * Spatial_Size + ny) * Spatial_Size + nz;
                    complex P_n = P_grid[idx_n];
                    
                    polyakov_data += P_m * std::conj(P_n);
                    N++;
                }
            }
        }
    }

    return (N > 0) ? (polyakov_data / static_cast<double>(N)) : complex(0.0, 0.0);
}
     
void check_unitarity(const Link_array& arr) {
    constexpr double eps = 1e-12; // tolerance for numerical errors
    for (int i1 = 0; i1 < Spatial_Size; i1++)
        for (int i2 = 0; i2 < Spatial_Size; i2++)
            for (int i3 = 0; i3 < Spatial_Size; i3++)
                for (int i4 = 0; i4 < temporal_size; i4++)
                    for (int d = 0; d < 4; d++) {
                        link_index idx = {i1, i2, i3, i4, d};
                        SU3 U = get_SU3_at_link(arr, idx);

                        // Check determinant
                        std::complex<double> det = U.determinant();
                        if (std::abs(det - 1.0) > eps) {
                            std::cerr << "WARNING: det(U) != 1 at link "
                                    << "(" << i1 << "," << i2 << "," << i3 << "," << i4
                                    << "), dir=" << d 
                                    << " det(U) = " << det << std::endl;
                        }
                        assert(std::abs(det - 1.0) < eps);

                        // Check unitarity: U^dagger * U = I
                        SU3 identity_check = U.adjoint() * U;
                        for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++) {
                            std::complex<double> expected = (i == j) ? 1.0 : 0.0;
                            if (std::abs(identity_check(i, j) - expected) > eps) {
                                std::cerr << "WARNING: U^dagger * U != I at link "
                                        << "(" << i1 << "," << i2 << "," << i3 << "," << i4
                                        << "), dir=" << d 
                                        << " element (" << i << "," << j << ") = " 
                                        << identity_check(i, j) << std::endl;
                            }
                            assert(std::abs(identity_check(i, j) - expected) < eps);
                        }
        }
        std::cout << "All links are unitary and have determinant 1 within tolerance." << std::endl;
    }
