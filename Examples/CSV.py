import numpy as np
from scipy import io
from lookup import lookup
from lookup_vgs import lookup_vgs

if __name__ == "__main__":
    # Load the .mat data file
    data = io.loadmat('nch_18.mat')
    nch_data = data['nch']
    
    #To save the data in a CSV File, use the np.savetxt function
    ex1 = lookup(nch_data, 'GM_CGG', 'GM_GDS', 50.9738, 'L', np.arange(0.5, 1.7, 0.1))
    print("Example 1:\n", ex1)
    np.savetxt("ex1.csv", ex1,  delimiter= ",")
    
    ex2 = lookup_vgs(nch_data, ID_W=1e-4, VDB=0.6, VGB=1, L=0.3)
    print("Example 2:", ex2)
    np.savetxt("ex2.csv", ex2,  delimiter= ",")
    
    ex3 = lookup(nch_data, 'GM_GDS', 'L', np.arange(0.8, 1.4, 0.2))
    ex3 = np.transpose(ex3)
    print("Example 3:\n", ex3)
    np.savetxt("ex3.csv", ex3,  delimiter= ",")
