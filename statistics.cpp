#include "statistics.h"





double mean(const std::array<double,autocorrelation_sweeps>& arr) {
    double s = 0.0;
    for (int i = 0; i < autocorrelation_sweeps;i++){
        s += arr[i];
    }    
    double N = autocorrelation_sweeps;
    return s / N;
}

double variance(const std::array<double,autocorrelation_sweeps>& arr){
    double m=mean(arr);
    double s = 0.0;
    for (int i = 0; i < autocorrelation_sweeps; i++) {
        double d = arr[i] - m;
        s += d * d;
    }
     double N = autocorrelation_sweeps;
    return s / N;
}

std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& arr){
    double m = mean(arr);
    double var = variance(arr);
    const double eps = 1e-12;
    double N = autocorrelation_sweeps;
    std::array<double,maxlag> rho; 
    if (std::abs(var) < eps){
        for (int t=0; t<maxlag; t++) rho[t] = (t==0)?1.0:0.0;
        return rho;
    }

    for (int t = 0; t < maxlag; t++) {
        double c = 0.0;
        for (int i = 0; i < N - t; i++) {
            c += (arr[i] - m) * (arr[i + t] - m);
        }
        c /= (N - t);   
        rho[t] = c / var;
    }
    return rho;
}

int tau_int(const std::array<double,autocorrelation_sweeps>& arr) {
    std::array<double,maxlag> rho = autocorr(arr);
    double tau = 1;
    int tau_integer;
    for (int t = 1; t < maxlag; t++) {
     tau += 2*  rho[t];
    }
    if (std::abs(tau) < 1) tau_integer = 1;
    else tau_integer = std::ceil(tau);
    return tau_integer;
}

