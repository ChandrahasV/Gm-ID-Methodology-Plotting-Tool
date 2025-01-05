import numpy as np
from scipy import interpolate
from scipy import io

def safe_divide(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.asarray(a)
        b = np.asarray(b)
        
        if a.shape != b.shape:
            if a.shape[0] == 1:
                a = np.repeat(a, b.shape[0], axis=0)
            if b.shape[0] == 1:
                b = np.repeat(b, a.shape[0], axis=0)
        
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result[b == 0] = np.nan
        elif b == 0:
            result = np.nan
    return result

def lookup(nch_data, outvar, *args, **kwargs):
    # Debug flag
    DEBUG = True
    
    # Extract base arrays
    try:
        L_values = nch_data['L'][0, 0].flatten()
        VGS_values = nch_data['VGS'][0, 0].flatten()
        VDS_values = nch_data['VDS'][0, 0].flatten()
        VSB_values = np.array([0]) if 'VSB' not in nch_data.dtype.names else nch_data['VSB'][0, 0].flatten()
        W = float(nch_data['W'][0, 0].flatten()[0])
    except Exception as e:
        print(f"Error extracting values: {e}")
        return None

    # Default parameters
    params = {
        'L': np.min(L_values),
        'VGS': VGS_values,
        'VDS': np.max(VDS_values)/2,
        'VSB': 0,
        'METHOD': 'pchip',
        'WARNING': 'on'
    }
    
    # Process args into kwargs
    i = 0
    while i < len(args):
        if isinstance(args[i], str) and i + 1 < len(args):
            kwargs[args[i]] = args[i + 1]
            i += 2
        else:
            i += 1
    
    # Update params with kwargs and ensure arrays
    for key, value in kwargs.items():
        if key in params:
            params[key] = np.atleast_1d(value)

    # Determine mode
    out_ratio = '_' in outvar
    var_ratio = len(args) > 0 and isinstance(args[0], str) and '_' in args[0]
    mode = 3 if (out_ratio and var_ratio) else (2 if out_ratio else 1)

    if DEBUG: print(f"Mode: {mode}")

    # Process output variable
    if out_ratio:
        numerator, denominator = outvar.split('_')
        if denominator == 'W':
            ydata = nch_data[numerator][0, 0] / W
        elif numerator == 'W':
            ydata = W / nch_data[denominator][0, 0]
        else:
            ydata = safe_divide(nch_data[numerator][0, 0], nch_data[denominator][0, 0])
    else:
        ydata = nch_data[outvar][0, 0]

    # Mode 3: Cross-lookup
    if mode == 3:
        try:
            # Parse arguments
            ratio_var = args[0]
            xdesired = np.atleast_1d(args[1])
            
            # Determine which parameter is being swept
            sweep_param = None
            sweep_values = None
            for param in ['L', 'VDS', 'VGS', 'VSB']:
                if param in kwargs and len(np.atleast_1d(kwargs[param])) > 1:
                    sweep_param = param
                    sweep_values = np.atleast_1d(kwargs[param])
                    break
            
            if sweep_param is None:
                sweep_param = 'L'
                sweep_values = np.array([params['L'][0] if isinstance(params['L'], np.ndarray) else params['L']])
            
            # Initialize output array with proper shape
            output = np.full((len(sweep_values), len(xdesired)), np.nan)
            
            # Process input ratio
            num, den = ratio_var.split('_')
            if den == 'W':
                xdata = nch_data[num][0, 0] / W
            elif num == 'W':
                xdata = W / nch_data[den][0, 0]
            else:
                xdata = safe_divide(nch_data[num][0, 0], nch_data[den][0, 0])

            # For each sweep value
            for idx, sweep_val in enumerate(sweep_values):
                # Update params for current sweep value
                current_params = params.copy()
                current_params[sweep_param] = sweep_val
                
                # Get parameter values, handling both scalar and array cases
                L = current_params['L'][0] if isinstance(current_params['L'], np.ndarray) else current_params['L']
                VDS = current_params['VDS'][0] if isinstance(current_params['VDS'], np.ndarray) else current_params['VDS']
                VSB = current_params['VSB'][0] if isinstance(current_params['VSB'], np.ndarray) else current_params['VSB']
                
                L_idx = np.abs(L_values - L).argmin()
                VDS_idx = np.abs(VDS_values - VDS).argmin()
                VSB_idx = np.abs(VSB_values - VSB).argmin()
                
                # Extract curves
                x_curves = []
                y_curves = []
                
                for vgs_idx in range(len(VGS_values)):
                    x = xdata[L_idx, vgs_idx, VDS_idx, VSB_idx]
                    y = ydata[L_idx, vgs_idx, VDS_idx, VSB_idx]
                    
                    if np.isfinite(x) and np.isfinite(y):
                        x_curves.append(x)
                        y_curves.append(y)
                
                if len(x_curves) > 0:
                    x_curves = np.array(x_curves)
                    y_curves = np.array(y_curves)
                    
                    # Sort and process curves
                    sort_idx = np.argsort(x_curves)
                    x_curves = x_curves[sort_idx]
                    y_curves = y_curves[sort_idx]
                    
                    # Special case handling
                    if num == 'GM' and den == 'ID':
                        peaks = np.where(np.diff(x_curves) < 0)[0]
                        if len(peaks) > 0:
                            max_idx = peaks[0]
                            x_curves = x_curves[max_idx:]
                            y_curves = y_curves[max_idx:]
                    elif num == 'GM' and (den == 'CGG' or den == 'CGS'):
                        peaks = np.where(np.diff(x_curves) < 0)[0]
                        if len(peaks) > 0:
                            max_idx = peaks[0]
                            x_curves = x_curves[:max_idx+1]
                            y_curves = y_curves[:max_idx+1]
                    
                    # Remove duplicates
                    unique_mask = np.concatenate(([True], np.diff(x_curves) != 0))
                    x_curves = x_curves[unique_mask]
                    y_curves = y_curves[unique_mask]
                    
                    if len(x_curves) >= 2:
                        try:
                            if params['METHOD'] == 'pchip':
                                interpolator = interpolate.PchipInterpolator(x_curves, y_curves, extrapolate=False)
                            else:
                                interpolator = interpolate.interp1d(x_curves, y_curves,
                                                                  kind=params['METHOD'],
                                                                  bounds_error=False,
                                                                  fill_value=np.nan)
                            
                            mask = (xdesired >= np.min(x_curves)) & (xdesired <= np.max(x_curves))
                            output[idx, mask] = interpolator(xdesired[mask])
                            
                        except Exception as e:
                            if DEBUG: print(f"Interpolation error: {e}")
                    elif len(x_curves) == 1:
                        exact_matches = np.isclose(xdesired, x_curves[0], rtol=1e-10)
                        output[idx, exact_matches] = y_curves[0]
            
            # Ensure output is always at least 1D array
            squeezed = output.squeeze()
            return np.atleast_1d(squeezed)

        except Exception as e:
            if DEBUG: print(f"Mode 3 error: {e}")
            raise

    # Modes 1 and 2
    else:
        points = (L_values, VGS_values, VDS_values, VSB_values)
        
        for key in ['L', 'VGS', 'VDS', 'VSB']:
            params[key] = np.atleast_1d(params[key])
            
        xi = np.array(np.meshgrid(params['L'], params['VGS'], params['VDS'], params['VSB'],
                                 indexing='ij')).reshape(4, -1).T

        output = interpolate.interpn(points, ydata, xi, method='linear')
        output = output.reshape(len(params['L']), len(params['VGS']), 
                              len(params['VDS']), len(params['VSB']))
        output = np.squeeze(output)
        
        if output.ndim > 1:
            output = np.transpose(output)
        if output.ndim == 1:
            output = output.reshape(-1, 1)
            
        # Ensure output is always at least 1D array
        return np.atleast_1d(output)
    
if __name__ == "__main__":
    # Load the .mat data file
    data = io.loadmat('nch_18.mat')
    nch_data = data['nch']
    
    print("Available fields in data:", nch_data.dtype.names)
    
    # Test Case 1: Lookup 'ID' with specified L range and a fixed VGS
    print("\n--- Test Case 1: Lookup 'ID' ---")
    print("Inputs: L=np.arange(0.4, 1.8, 0.2), VGS=0.5")
    result1 = lookup(nch_data, 'ID', 'L', np.arange(0.4, 1.8, 0.2), 'VGS', 0.5)
    print("Result:\n", result1)

    # Test Case 2: Lookup 'ID_W' with specific values for GM_ID and L
    print("\n--- Test Case 2: Lookup 'ID_W' ---")
    print("Inputs: L=0.257, GM_ID=15")
    result2 = lookup(nch_data, 'ID_W', 'GM_ID', 15, 'L', 0.257)
    print("Result:\n", result2)

    # Test Case 3: Lookup 'GM_ID' using default parameters
    print("\n--- Test Case 3: Lookup 'GM_ID' ---")
    print("Inputs: Default parameters")
    result3 = lookup(nch_data, 'GM_ID')
    print("Result:\n", result3)

    # Test Case 4: Lookup 'GM_CGG' for a range of GM_ID values
    print("\n--- Test Case 4: Lookup 'GM_CGG' ---")
    print("Inputs: GM_ID=np.arange(5, 20.1, 0.1)")
    result4 = lookup(nch_data, 'GM_CGG', 'GM_ID', np.arange(5, 20.1, 0.1))
    print("Result:\n", result4)

    # Test Case 5: Lookup 'GM_CGG' with specific GM_GDS and L
    print("\n--- Test Case 5: Lookup 'GM_CGG' with additional parameter ---")
    print("Inputs: GM_GDS=50.9738, L=1.748")
    result5 = lookup(nch_data, 'GM_CGG', 'GM_GDS', 50.9738, 'L', 1.748)
    print("Result:\n", result5)

    # Test Case 6: Lookup 'ID' for a fixed L value
    print("\n--- Test Case 6: Lookup 'ID' ---")
    print("Inputs: L=0.257")
    result6 = lookup(nch_data, 'ID', 'L', 0.257)
    print("Result:\n", result6)

    # Test Case 7: Lookup 'STH_GM' with VGS and L range, normalized by a factor
    print("\n--- Test Case 7: Lookup 'STH_GM' ---")
    print("Inputs: VGS=np.arange(0.2, 0.9, 25e-3), L=np.arange(0.4, 1.8, 0.2)")
    result7 = lookup(nch_data, 'STH_GM', 'VGS', np.arange(0.2, 0.9, 25e-3), 'L', np.arange(0.4, 1.8, 0.2)) / (4 * 1.3806488e-23 * 300)
    print("Result:\n", result7)

    # Test Case 8: Lookup 'CGS_W' for specific GM_ID and L
    print("\n--- Test Case 8: Lookup 'CGS_W' ---")
    print("Inputs: GM_ID=20, L=0.35")
    result8 = lookup(nch_data, 'CGS_W', 'GM_ID', 20, 'L', 0.35)
    print("Result:\n", result8)

    # Test Case 9: Lookup 'GM_CGG' for a range of ID_W values
    print("\n--- Test Case 9: Lookup 'GM_CGG' ---")
    print("Inputs: ID_W=[50.9738e-09, 60.9738e-09]")
    result9 = lookup(nch_data, 'GM_CGG', 'ID_W', [50.9738e-09, 60.9738e-09])
    print("Result:\n", result9)

    # Test Case 10: Lookup 'CGS_CGG' for specific ID_W and L
    print("\n--- Test Case 10: Lookup 'CGS_CGG' ---")
    print("Inputs: ID_W=50.9738e-09, L=1.748")
    result10 = lookup(nch_data, 'CGS_CGG', 'ID_W', 50.9738e-09, 'L', 1.748)
    print("Result:\n", result10)

    # Test Case 11: Lookup 'GM_ID' with a range of L values
    print("\n--- Test Case 11: Lookup 'GM_ID' ---")
    print("Inputs: L=np.arange(0.4, 2, 0.2)")
    result11 = lookup(nch_data, 'GM_ID', 'L', np.arange(0.4, 2, 0.2))
    print("Result:\n", result11)

    # Test Case 12: Lookup 'ID' with a range of VGS and VDS values
    print("\n--- Test Case 12: Lookup 'ID' ---")
    print("Inputs: VGS=np.arange(0, 1.1, 0.1), VDS=np.arange(0, 1.1, 0.1)")
    result12 = lookup(nch_data, 'ID', 'VGS', np.arange(0, 1.1, 0.1), 'VDS', np.arange(0, 1.1, 0.1))
    print("Result:\n", result12)

    # Test Case 13: Diagonal elements from a 2D result matrix
    print("\n--- Test Case 13: Lookup 'ID' diagonal ---")
    print("Inputs: VGS=np.arange(0, 1.1, 0.1), VDS=np.arange(0, 1.1, 0.1)")
    result13a = lookup(nch_data, 'ID', 'VGS', np.arange(0, 1.1, 0.1), 'VDS', np.arange(0, 1.1, 0.1))
    result13 = result13a.diagonal()
    print("Result:\n", result13)

    # Additional test cases
    print("\n--- Additional Test Cases ---")
    result14 = lookup(nch_data, 'GM_ID', 'L', np.arange(0.8, 1.4, 0.2))
    result15 = lookup(nch_data, 'GM_GDS', 'L', np.arange(0.8, 1.4, 0.2))
    result16 = lookup(nch_data, 'ID_W', 'L', np.arange(0.8, 1.4, 0.2))
    print("GM_ID Results:\n", result14)
    print("GM_GDS Results:\n", result15)
    print("ID_W Results:\n", result16)

    # Test Case 18: Lookup 'GM_CGG' with a range of L values
    print("\n--- Test Case 18: Lookup 'GM_CGG' with additional parameter ---")
    print("Inputs: GM_GDS=50.9738, L=np.arange(0.5, 1.7, 0.1)")
    result18 = lookup(nch_data, 'GM_CGG', 'GM_GDS', 50.9738, 'L', np.arange(0.5, 1.7, 0.1))
    print("Result:\n", result18)

    # Test Case 19: Lookup 'GM_CGG' with GM_ID and VDS range
    print("\n--- Test Case 19: Lookup 'GM_CGG' with additional parameter ---")
    print("Inputs: GM_ID=15, VDS=np.arange(0.2, 0.5, 0.1)")
    result19 = lookup(nch_data, 'GM_CGG', 'GM_ID', 15, 'VDS', np.arange(0.2, 0.5, 0.1))
    print("Result:\n", result19)

    # Test Case 20: Lookup 'ID_W' with GM_ID and specific L values
    print("\n--- Test Case 20: Lookup 'ID_W' ---")
    print("Inputs: GM_ID=0.973, L=[0.4, 0.5]")
    result20 = lookup(nch_data, 'ID_W', 'GM_ID', 0.973, 'L', [0.4, 0.5])
    print("Result:\n", result20)
