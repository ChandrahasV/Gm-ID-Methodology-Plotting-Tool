import numpy as np
from scipy import io
from lookup import lookup
from lookup_vgs import lookup_vgs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the .mat data file
    data = io.loadmat('pch_18.mat')
    pch_data = data['pch']
    
    # Get data values and transpose
    gm_ID = lookup(pch_data, 'GM_ID', 'L', [0.6, 0.7, 0.8, 0.9])
    noise_coefficient = lookup(pch_data, 'STH_GM', 'L',[0.6, 0.7, 0.8, 0.9] ) / (4 * 1.3806488e-23 * 300)

    # Create a figure
    plt.figure(figsize=(12, 5))    
    
    # Create the plot on the current axes
    plt.plot(gm_ID,noise_coefficient, 'k-', linewidth=1)
    plt.grid(True)
    plt.xlim([0, 30])
    plt.ylim([0,1])
    plt.xlabel('$g_m/I_D$ (S/A)')
    plt.ylabel('$𝛾_{P}$')
    plt.title('$𝛾_{P}$ vs $g_m/I_D$')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
