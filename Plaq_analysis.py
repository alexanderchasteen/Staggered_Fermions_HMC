import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

Spatial_size = 16
Temporal_size = 8
Nplaq = Spatial_size*Spatial_size*Spatial_size*Temporal_size*6
df = pd.read_csv('RawMC_Data.csv')

beta_array = df.columns.to_numpy()
beta_array = beta_array.astype(float)
CONFIGS = len(beta_array)

# ---------------------------------------------------------
# Observable Functions
# ---------------------------------------------------------
def fourth_moment(array):
    m = np.mean(array)
    s = 0.0
    for x in array:
        diff = x - m
        s += diff * diff * diff * diff
    s /= len(array)
    return s

def binder_cum(array):
    fourth_mom = fourth_moment(array)
    second_mom = np.var(array)
    # Protection against divide by zero if variance is too small
    if second_mom == 0:
        return np.nan
    binder_cum = 1 - (fourth_mom) / (3 * second_mom * second_mom)
    return binder_cum

# ---------------------------------------------------------
# Moving Block Bootstrap
# ---------------------------------------------------------
def moving_block_bootstrap(array, block_size, num_bootstraps, stat_func):
    """
    Estimates the standard error of a statistic using Moving Block Bootstrap.
    Returns the statistic evaluated on the original array, and the bootstrap standard deviation.
    """
    n = len(array)
    num_blocks_to_draw = n // block_size
    valid_starts = n - block_size + 1
    
    bootstrap_stats = []
    for _ in range(num_bootstraps):
        # Draw random starting indices for overlapping blocks
        starts = np.random.randint(0, valid_starts, size=num_blocks_to_draw)
        
        # Concatenate drawn blocks into a new bootstrap sample
        bootstrap_sample = np.concatenate([array[start:start + block_size] for start in starts])
        
        # Compute the observable on the bootstrap sample
        bootstrap_stats.append(stat_func(bootstrap_sample))
        
    # Central value is best estimated from the full original dataset
    val = stat_func(array)
    
    # Error is the standard deviation of the bootstrap distribution
    err = np.std(bootstrap_stats, ddof=1)
    
    return val, err

# ---------------------------------------------------------
# Main Analysis Loop
# ---------------------------------------------------------
blocks = 50
num_bootstraps = 200  # Number of bootstrap resamples (adjust as needed)

avg_plaq, SD_avg_plaq = [], []
plaq_sus, SD_plaq_sus = [], []
binder, SD_Binder = [], []

for i in range(CONFIGS):
    array = df.iloc[:, i].to_numpy()
    
    # Set block size based on the desired number of blocks
    block_size = len(array) // blocks
    
    # 1. Average Plaquette
    m_plaq, err_plaq = moving_block_bootstrap(array, block_size, num_bootstraps, np.mean)
    avg_plaq.append(m_plaq)
    SD_avg_plaq.append(err_plaq)

    # 2. Plaquette Susceptibility
    m_sus, err_sus = moving_block_bootstrap(array, block_size, num_bootstraps, lambda x: Nplaq * np.var(x))
    plaq_sus.append(m_sus)
    SD_plaq_sus.append(err_sus)

    # 3. Binder Cumulant
    m_binder, err_binder = moving_block_bootstrap(array, block_size, num_bootstraps, binder_cum)
    binder.append(m_binder)
    SD_Binder.append(err_binder)

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.errorbar(beta_array, avg_plaq, yerr=SD_avg_plaq, fmt='o', label='Average Plaquette', color='blue', ecolor='red', elinewidth=3)
plt.xlabel('Beta')
plt.ylabel('Average Plaquette')
plt.legend()
plt.title('Average Plaquette vs Beta')
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(beta_array, plaq_sus, yerr=SD_plaq_sus, fmt='o', label='Plaquette Susceptibility', color='blue', ecolor='red', elinewidth=3)
plt.xlabel('Beta')
plt.ylabel('Susceptibility')
plt.legend()
plt.title('Susceptibility vs Beta')
plt.show()

plt.figure(figsize=(8, 5))
plt.errorbar(beta_array, binder, yerr=SD_Binder, fmt='o', label='Binder Cumulant', color='blue', ecolor='red', elinewidth=3)
plt.xlabel('Beta')
plt.ylabel('Binder Cumulant')
plt.legend()
plt.title('Binder Cumulant vs Beta')
plt.show()

# ---------------------------------------------------------
# Saving Data
# ---------------------------------------------------------
folder = './Analyzed_Data/'
# Ensure the folder exists in your environment before saving
import os
os.makedirs(folder, exist_ok=True)

montecarlo_data = np.transpose(np.array([
    beta_array, 
    avg_plaq, 
    plaq_sus, 
    binder, 
    SD_avg_plaq, 
    SD_plaq_sus, 
    SD_Binder
]))

filename = folder + 'Spatial_size_' + str(Spatial_size) + '_Temporal_size_' +str(Temporal_size) + '_analysis.csv'

header = ["Beta", "Avg Plaq", "Plaq Sus", "Binder Cumulant", "Bootstrap Plaq SD", "Bootstrap Sus SD", "Bootstrap Binder SD"]
with open(filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header)
    csv_writer.writerows(montecarlo_data)

print(f"CSV file '{filename}' created successfully.")