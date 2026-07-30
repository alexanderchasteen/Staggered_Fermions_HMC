#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <array>
#include <utility>
#include <omp.h>
#include "Lattice.h"
#include "SU3_Sampling.h"
#include "rng.h"
#include "Parameters.h"
#include "statistics.h"

int main() {
    // --- Configuration Flags for Checkpointing ---
    bool load_checkpoint = false; // Set to true to skip thermalization and load saved state
    bool save_checkpoint = true;  // Set to true to save final state to disk at the end
    const std::string checkpoint_filename = "final_lattice_configs.bin";

    // 1. Seed every thread's RNG once at startup
    #pragma omp parallel
    {
        seed_thread_rng();
    }

    // --- 2. File Initialization ---
    std::string filename1 = "Thermalization_Data.csv";
    std::ofstream Thermfile(filename1);
    if (!Thermfile) { std::cerr << "Error creating file!\n"; return 1; }
    Thermfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < CONFIG; i++) {
        Thermfile << coupling_constants[i] << (i < CONFIG - 1 ? "," : "");
    }
    Thermfile << "\n"; Thermfile.flush();

    std::string filename2 = "RawMC_Data.csv";
    std::ofstream RawMCfile(filename2);
    if (!RawMCfile) { std::cerr << "Error creating file!\n"; return 1; }
    RawMCfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < CONFIG; i++) {
        RawMCfile << coupling_constants[i] << (i < CONFIG - 1 ? "," : "");
    }
    RawMCfile << "\n"; RawMCfile.flush();

    std::string filename_action = "Action_Data.csv";
    std::ofstream Actionfile(filename_action);
    if (!Actionfile) { std::cerr << "Error creating file!\n"; return 1; }
    Actionfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < CONFIG; i++) {
        Actionfile << coupling_constants[i] << (i < CONFIG - 1 ? "," : "");
    }
    Actionfile << "\n"; Actionfile.flush();

    std::string filename3 = "Poly_Corr_Real_Data.csv";
    std::ofstream poly(filename3);
    if (!poly) { std::cerr << "Error creating file!\n"; return 1; }
    poly << std::fixed << std::setprecision(6);
    poly << "sweep,coupling_idx,";
    for (int i = 1; i <= maxdistance; i++) {
        poly << "r_" << i << (i < maxdistance ? "," : "");
    }
    poly << "\n"; poly.flush();

    std::string filename4 = "Poly_Corr_Imag_Data.csv";
    std::ofstream polycomplex(filename4);
    if (!polycomplex) { std::cerr << "Error creating file!\n"; return 1; }
    polycomplex << std::fixed << std::setprecision(6);
    polycomplex << "sweep,coupling_idx,";
    for (int i = 1; i <= maxdistance; i++) {
        polycomplex << "r_" << i << (i < maxdistance ? "," : "");
    }
    polycomplex << "\n"; polycomplex.flush();


    // --- 3. Lattice Setup & Loading ---
    std::array<Link_array, CONFIG> links_at_coupling;
    for (int i = 0; i < CONFIG; i++) {
        links_at_coupling[i] = Link_array(array_size, 0.0);
    }

    bool successfully_loaded = false;

    if (load_checkpoint) {
        std::cout << "Attempting to load saved configurations from " << checkpoint_filename << "..." << std::endl;
        successfully_loaded = load_all_lattice_configs(links_at_coupling, checkpoint_filename);
        if (successfully_loaded) {
            std::cout << "All " << CONFIG << " lattice configurations successfully restored." << std::endl;
        } else {
            std::cerr << "Failed to load checkpoint file. Falling back to cold start.\n";
        }
    }

    // Default to cold start if load_checkpoint is false or reading failed
    if (!load_checkpoint || !successfully_loaded) {
        std::cout << "Initializing lattices with cold start..." << std::endl;
        for (int i = 0; i < CONFIG; i++) {
            cold_start_array(links_at_coupling[i]);
        }
    }


    // --- 4. Thermalization Phase ---
    if (load_checkpoint && successfully_loaded) {
        std::cout << "Skipping thermalization phase (loaded existing state)." << std::endl;
    } else {
        std::cout << "Starting thermalization phase..." << std::endl;
        std::vector<double> therm_plaq(CONFIG);

        for (int j = 0; j < thermal_sweeps; j++) {
            bool log_this_sweep = (j % 10 == 0);
            
            if (log_this_sweep) {
                std::cout << "Thermalization Sweep: " << j << std::endl;
            }
            
            #pragma omp parallel for
            for (int i = 0; i < CONFIG; i++) {
                auto [avg_plaq_local, avg_action_local] = heatbath_update(links_at_coupling[i], coupling_constants[i]);
                therm_plaq[i] = avg_plaq_local;
            }

            for (int i = 0; i < CONFIG; i++) {
                Thermfile << therm_plaq[i] << (i < CONFIG - 1 ? "," : "");
            }
            Thermfile << "\n";

            if (log_this_sweep) {
                for (int i = 0; i < CONFIG; i++) {
                    std::cout << "  Coupling: " << coupling_constants[i] 
                              << ", Avg Plaquette: " << therm_plaq[i] << std::endl;
                }
            }
        }
    }


    // --- 5. Measurement Phase ---
    std::cout << "Starting measurement phase..." << std::endl;
    std::vector<double> sweep_plaq(CONFIG);
    std::vector<double> sweep_action(CONFIG);
    std::vector<std::vector<double>> sweep_poly_real(CONFIG, std::vector<double>(maxdistance));
    std::vector<std::vector<double>> sweep_poly_imag(CONFIG, std::vector<double>(maxdistance));

    for (int j = 0; j < measurment_sweeps; j++) {
        if (j % 10 == 0) {
            std::cout << "Measurement Sweep: " << j << std::endl;
        }

        #pragma omp parallel for
        for (int i = 0; i < CONFIG; i++) {
            auto [plaq, action] = heatbath_update(links_at_coupling[i], coupling_constants[i]);
            sweep_plaq[i]   = plaq;
            sweep_action[i] = action;

            // 2. Precompute Polyakov grid 
            std::vector<complex> P_grid = precompute_polyakov_grid(links_at_coupling[i]);

            // 3. Compute Correlator
            for (int r2 = 1; r2 <= maxdistance; r2++) { // Note: maxdistance should now represent r^2
                complex C_r = correlator_over_fixed_distance_fast(P_grid, r2);
                
                // C_r will be 0.0 + 0.0i if there are no integer points corresponding to the r^2 value.
                sweep_poly_real[i][r2 - 1] = C_r.real();
                sweep_poly_imag[i][r2 - 1] = C_r.imag();
            }    
        } // <-- CORRECTLY CLOSED 'i' LOOP HERE

        // --- FILE OUTPUT (Inside the 'j' loop) ---
        for (int i = 0; i < CONFIG; i++) {
            RawMCfile  << sweep_plaq[i]   << (i < CONFIG - 1 ? "," : "");
            Actionfile << sweep_action[i] << (i < CONFIG - 1 ? "," : "");
        }
        RawMCfile  << "\n";
        Actionfile << "\n";

        for (int i = 0; i < CONFIG; i++) {
            poly        << j << "," << i << ",";
            polycomplex << j << "," << i << ",";

            for (int r = 1; r <= maxdistance; r++) {
                poly        << sweep_poly_real[i][r - 1] << (r < maxdistance ? "," : "");
                polycomplex << sweep_poly_imag[i][r - 1] << (r < maxdistance ? "," : "");
            }
            poly        << "\n";
            polycomplex << "\n";
        }
    } // <-- CORRECTLY CLOSED 'j' (MEASUREMENT SWEEP) LOOP HERE


    // --- 6. Save Final Lattice Configurations ---
    if (save_checkpoint) {
        std::cout << "Saving final lattice configurations to " << checkpoint_filename << "..." << std::endl;
        if (save_all_lattice_configs(links_at_coupling, checkpoint_filename)) {
            std::cout << "Final configurations saved successfully." << std::endl;
        }
    }

    Thermfile.close();
    RawMCfile.close();
    Actionfile.close();
    poly.close();
    polycomplex.close();
    return 0;
}