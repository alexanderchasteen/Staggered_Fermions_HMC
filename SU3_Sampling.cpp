#include "SU3_Sampling.h"
#include "rng.h"


// Initialize our uniform distribution between [0,1)
std::uniform_real_distribution<double> dist(0.0, 1.0);


// Initialize another uniform distribution between [-1,1)
std::uniform_real_distribution<double> dist2(-1.0, 1.0);


// SU(2) Generator
// Want to generate SU(2) Matrices according to the distribution. Do this according to Gattringer
// Each SU(2) matrix is a 4 vector (x0,\vec{x}) as generated according to the following code
// First generatr x0
double generate_x0(double a,double beta){
    // First we need to generate 3 numbers in (0,1]. So use rng to generate in r_i in [0,1) and do 1-r_i
    while (true){ 
        double r_1=1.0-dist(rng);
        double r_2=1.0-dist(rng);
        double r_3=1.0-dist(rng);
        // As in gatringer this parameter is defined
        double lambda_squared=-1/(2*a*beta) * (std::log(r_1)+(std::cos(2*pi*r_2))*(std::cos(2*pi*r_2))*std::log(r_3));
        double x=1-lambda_squared;
        double r=dist(rng);
        if (r * r <= x && x >= -1 && x <= 1) return 1-2*lambda_squared; 
    }
}


// used to run the generation of x0 and do a statistics check in the respective .py script to see that is obeys the correct distribution
void sample_x0_check(double a,double b,int j){
    std::vector<double> array= {};
    std::ofstream outfile("output.csv");

    // Generate numbers and write them to CSV
    for (int i = 0; i < j; i++) {
        double x = generate_x0(a, b);
        array.push_back(x);
        outfile << x << "\n";   // Write each x on a new line
    }

    outfile.close();
}

// gets the norm of x0 used to generate the vectore x
double get_norm_x(double x0){
    return std::sqrt(1-x0*x0);
}

// to get x1,x2,x3, generate random unit 3 vector and then give it length as calculated above. First we generate unit norm vector
std::array<double,3> generate_random_unit_3_vector(){
    while (true){     
        double r_1=dist2(rng);
        double r_2=dist2(rng);
        double r_3=dist2(rng);
        double length=std::sqrt(r_1*r_1+r_2*r_2+r_3*r_3);
        std::array<double,3> array ={r_1,r_2,r_3};
        // auto& means it will alter the values of the array
        if (length <= 1.0){
            for (auto& x:array){
                x /= length;
            }
            return array;
        }  
    }    
}

// Now we generate the proper normed vector
std::array<double,3> generate_3_vector_norm_x(double x0){
    std::array<double,3> vector=generate_random_unit_3_vector();
    double norm=get_norm_x(x0);
    for (auto&x : vector){
        x *= norm;
    }
    return vector;
}



four_complex_array SU_2_matrix_as_array(double a, double beta){   
    double x0 = generate_x0(a,beta);
    std::array<double,3> vector = generate_3_vector_norm_x(x0);
    double x1=vector[0];
    double x2=vector[1];
    double x3=vector[2];

    std::complex<double> component00 = std::complex<double>(x0,x3);
    std::complex<double> component01 = std::complex<double>(x2,x1);
    std::complex<double> component10 = std::complex<double>(-x2,x1);
    std::complex<double> component11 = std::complex<double>(x0,-x3);
    four_complex_array SU2 = {component00,component01 ,component10,component11};
    return SU2;
}



// Now we want to embeed these SU(2) matrices as the respecive SU(3) subgroups. Let us do this by using the following association of types of matrices: 
//    (u_11 u_12 0)            (1     0     0 )            (u_11    0  u_12)
// R= (u_21 u_22 0)         S= (0   u_11  u_12)         T= (  0     1     0)
//    (  0    0  1)            (0   u_21  u_22)            (u_21    0  u_22)

SU3 SU3_TypeR_generator(double a,double beta){
    four_complex_array SU2_generated = SU_2_matrix_as_array(a,beta);
    SU3 U;
    U(0,0) = SU2_generated[0];  
    U(0,1) = SU2_generated[1]; 
    U(0,2) = std::complex<double>(0.0, 0.0);

    U(1,0) = SU2_generated[2];  
    U(1,1) = SU2_generated[3];
    U(1,2) = std::complex<double>(0.0,0.0);


    U(2,0) = std::complex<double>(0, 0);  
    U(2,1) = std::complex<double>(0, 0); 
    U(2,2) = std::complex<double>(1.0, 0.0);
    return U;
}


SU3 SU3_TypeS_generator(double a,double beta){
    four_complex_array SU2_generated = SU_2_matrix_as_array(a,beta);
    SU3 U;
    U(0,0) = complex (1.0,0.0);  
    U(0,1) = complex (0.0,0.0); 
    U(0,2) = complex (0.0,0.0);

    U(1,0) = complex (0.0,0.0);  
    U(1,1) = SU2_generated[0];
    U(1,2) = SU2_generated[1];


    U(2,0) = complex (0.0,0.0);  
    U(2,1) = SU2_generated[2]; 
    U(2,2) = SU2_generated[3];
    return U;
}


SU3 SU3_TypeT_generator(double a,double beta){
    four_complex_array SU2_generated = SU_2_matrix_as_array(a,beta);
    SU3 U;
    U(0,0) = SU2_generated[0];  
    U(0,1) = complex (0.0,0.0); 
    U(0,2) = SU2_generated[1];

    U(1,0) = complex (0.0,0.0);  
    U(1,1) = complex (1.0,0.0);
    U(1,2) = complex (0.0,0.0);


    U(2,0) = SU2_generated[2];  
    U(2,1) = complex (0.0,0.0); 
    U(2,2) = SU2_generated[3];
    return U;
}


SU2 SU2_generator(double a,double beta){
    four_complex_array SU2_generated = SU_2_matrix_as_array(a,beta);
    SU2 U;
    U(0,0) = SU2_generated[0];  
    U(0,1) = SU2_generated[1]; 
    U(1,0) = SU2_generated[2];  
    U(1,1) = SU2_generated[3];
  
    return U;
}




std::array<double,4> generate_random_unit_4_vector(){
    while (true){     
        double r_1=dist2(rng);
        double r_2=dist2(rng);
        double r_3=dist2(rng);
        double r_4=dist2(rng);
        double length=std::sqrt(r_1*r_1+r_2*r_2+r_3*r_3+r_4*r_4);
        std::array<double,4> array ={r_1,r_2,r_3,r_4};
        // auto& means it will alter the values of the array
        if (length <= 1.0){
            for (auto& x:array){
                x /= length;
            }
            return array;
        }  
    }    
}

SU2 Random_SU2_generator(){
    std::array<double,4> array = generate_random_unit_4_vector();
    SU2 I;
    I.setIdentity();
    SU2 random_SU2 = array[0] * I + array[1] * complex(0,1)* Pauli::sigma_x + array[2] * complex(0,1)* Pauli::sigma_y + array[3] * complex(0,1)* Pauli::sigma_z;
    return random_SU2;
}


