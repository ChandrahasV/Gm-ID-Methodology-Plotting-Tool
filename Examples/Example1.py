import numpy as np
from scipy import io
from lookup import lookup
from lookup_vgs import lookup_vgs
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the .mat data file
    data = io.loadmat('nch_18.mat')
    nch_data = data['nch']
    
    # Get data values
    VGS_values = nch_data['VGS'][0, 0].flatten()
    VDS = np.arange(0.6, 1.5,0.3)
    gm_ID = lookup(nch_data, 'GM_ID', 'VDS', VDS, 'L', 0.6)
    gm_ID = np.transpose(gm_ID)    
    
    JD = lookup(nch_data, 'ID_W', 'VDS', VDS, 'L', 0.6)
    JD = np.transpose(JD)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # First subplot: VGS vs gm_ID
    ax1.plot(VGS_values, gm_ID, 'k-', linewidth=1)
    ax1.grid(True)
    ax1.set_xlim([0, 1.2])
    ax1.set_ylim([0, 35])
    ax1.set_xlabel('$V_{GS}$ (V)\n(a)')
    ax1.set_ylabel('$g_m/I_D$ (S/A)')
    ax1.set_title('VGS vs $g_m/I_D$')

    # Second subplot: log10(JD) vs gm_ID
    ax2.plot(np.log10(JD), gm_ID, 'k-', linewidth=1)
    ax2.grid(True)
    ax2.set_xlim([-10, -2])
    ax2.set_ylim([0, 35])
    ax2.set_xlabel('$log_{10}(J_D$ (A/µm))\n(b)')
    ax2.set_ylabel('$g_m/I_D$ (S/A)')
    ax2.set_title('$log_{10}(J_D)$ vs $g_m/I_D$')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
