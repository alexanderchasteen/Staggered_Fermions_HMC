#!/bin/bash
#SBATCH --job-name=lqcd_su3_heatbath
#SBATCH --output=lqcd_%j.out
#SBATCH --error=lqcd_%j.err
#SBATCH --account=open    # e.g., open, or your group account
#SBATCH --partition=basic                   # 'open' or standard submission queue
#SBATCH --nodes=1                           # Kept on a single node for OpenMP shared memory
#SBATCH --ntasks=1                          # Single execution process
#SBATCH --cpus-per-task=16                  # Number of OpenMP threads
#SBATCH --mem=16GB                          # Memory requested
#SBATCH --time=24:00:00                     # Walltime limit (HH:MM:SS)

# --- Roar Collab Module Setup ---
module purge
module load gcc

# Try loading Eigen module if provided by cluster environment
module load eigen 2>/dev/null || true

# --- OpenMP Environment Setup ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Increase thread stack size to prevent stack overflows on large lattice volumes
export OMP_STACKSIZE=512M

echo "=================================================="
echo "Running on Roar node: $(hostname)"
echo "Date: $(date)"
echo "Allocated OpenMP Threads: $OMP_NUM_THREADS"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=================================================="

cd $SLURM_SUBMIT_DIR

# --- Compilation with Eigen ---
# Include common system/module path locations for Eigen header files
EIGEN_INC="-I./eigen/eigen-5.0.0"

echo "Compiling C++ source code with Eigen support..."
g++ -O3 -fopenmp -std=c++17 $EIGEN_INC Main.cpp Lattice.cpp SU3_Sampling.cpp rng.cpp statistics.cpp -o lqcd_sim

if [ $? -ne 0 ]; then
    echo "Compilation failed! Check stderr log."
    exit 1
fi

# --- Execution ---
echo "Starting simulation..."
./lqcd_sim

echo "=================================================="
echo "Job completed successfully at $(date)"
echo "=================================================="