import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from lookup import lookup

# Note: Please ignore the "Mode" in the output while using the lookup_vgs function. It refers to the mode used by the lookup function when it is called.

def lookup_vgs(nch_data, **kwargs):
    debug = kwargs.pop('debug', False)
    
    try:
        L_values = nch_data['L'][0, 0].flatten()
        VGS_values = nch_data['VGS'][0, 0].flatten()
        VDS_values = nch_data['VDS'][0, 0].flatten()
        if 'VSB' in nch_data.dtype.names:
            VSB_values = nch_data['VSB'][0, 0].flatten()
        else:
            VSB_values = np.array([0])
            
        if debug:
            print("\nValue ranges:")
            print(f"L: {np.min(L_values):.3f} to {np.max(L_values):.3f}")
            print(f"VGS: {np.min(VGS_values):.3f} to {np.max(VGS_values):.3f}")
            print(f"VDS: {np.min(VDS_values):.3f} to {np.max(VDS_values):.3f}")
            print(f"VSB: {np.min(VSB_values):.3f} to {np.max(VSB_values):.3f}")
            
    except Exception as e:
        print(f"Error extracting values: {e}")
        print("Available fields in data:", nch_data.dtype.names)
        return np.array([])

    defaults = {
        'L': np.min(L_values),
        'VDS': np.max(VDS_values) / 2,
        'VDB': np.nan,
        'VGB': np.nan,
        'GM_ID': np.nan,
        'ID_W': np.nan,
        'VSB': 0,
        'METHOD': 'pchip'
    }

    params = {**defaults, **kwargs}

    if debug:
        print("\nInput parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")

    # Determine usage mode
    if np.isnan(params['VGB']) and np.isnan(params['VDB']):
        mode = 1
    elif not np.isnan(params['VGB']) and not np.isnan(params['VDB']):
        mode = 2
    else:
        print('Invalid syntax or usage mode! Please check the documentation.')
        return np.array([])
    
    if debug:
        print(f"\nOperating in mode {mode}")

    # Check whether GM_ID or ID_W was passed
    has_gm_id = isinstance(params['GM_ID'], (np.ndarray, list)) or not np.isnan(params['GM_ID'])
    has_id_w = isinstance(params['ID_W'], (np.ndarray, list)) or not np.isnan(params['ID_W'])

    if has_id_w:
        ratio_string = 'ID_W'
        ratio_data = np.atleast_1d(params['ID_W'])
    elif has_gm_id:
        ratio_string = 'GM_ID'
        ratio_data = np.atleast_1d(params['GM_ID'])
    else:
        print('Invalid syntax or usage mode! Please check the documentation.')
        return np.array([])

    if mode == 1:
        VGS = VGS_values
        ratio = lookup(nch_data, ratio_string, 
                      VGS=VGS, 
                      VDS=params['VDS'], 
                      VSB=params['VSB'], 
                      L=params['L'])
        
        if ratio is None:
            print("Error: lookup function returned None")
            return np.array([])
            
    else:  # mode 2
        step = VGS_values[1] - VGS_values[0]
        VSB = np.arange(np.min(VSB_values), np.max(VSB_values) + step, step)
        VGS = params['VGB'] - VSB
        VDS = params['VDB'] - VSB
        
        # Filter valid operating points
        valid_vsb = (VSB >= np.min(VSB_values)) & (VSB <= np.max(VSB_values))
        valid_vgs = (VGS >= np.min(VGS_values)) & (VGS <= np.max(VGS_values))
        valid_vds = (VDS >= np.min(VDS_values)) & (VDS <= np.max(VDS_values))
        valid_points = valid_vsb & valid_vds & valid_vgs
        
        if debug:
            print("\nMode 2 filtering details:")
            print(f"VSB range: {np.min(VSB):.3f} to {np.max(VSB):.3f}")
            print(f"VGS range: {np.min(VGS):.3f} to {np.max(VGS):.3f}")
            print(f"VDS range: {np.min(VDS):.3f} to {np.max(VDS):.3f}")
            print(f"Valid VSB points: {np.sum(valid_vsb)} / {len(VSB)}")
            print(f"Valid VGS points: {np.sum(valid_vgs)} / {len(VGS)}")
            print(f"Valid VDS points: {np.sum(valid_vds)} / {len(VDS)}")
            print(f"Total valid points: {np.sum(valid_points)} / {len(VGS)}")
        
        if np.sum(valid_points) == 0:
            print("Error: No valid operating points found within device limits")
            return np.array([])
        
        # Apply filtering
        VGS = VGS[valid_points]
        VSB = VSB[valid_points]
        VDS = VDS[valid_points]
        
        # Create L array matching the size of filtered points
        L_array = np.full_like(VGS, params['L'])
        
        if debug:
            print("\nLookup input ranges after filtering:")
            print(f"VGS: {np.min(VGS):.3f} to {np.max(VGS):.3f}")
            print(f"VDS: {np.min(VDS):.3f} to {np.max(VDS):.3f}")
            print(f"VSB: {np.min(VSB):.3f} to {np.max(VSB):.3f}")
            print(f"L: {params['L']}")
        
        # Get ratio values for valid points
        ratio = lookup(nch_data, ratio_string,
                      VGS=VGS,
                      VDS=VDS,
                      VSB=VSB,
                      L=L_array)
                      
        if ratio is None:
            print("Error: lookup function returned None")
            return np.array([])

        if debug:
            print("\nRatio array details:")
            print(f"Shape: {ratio.shape}")
            print(f"Finite values: {np.sum(np.isfinite(ratio))} / {ratio.size}")
            if np.sum(np.isfinite(ratio)) > 0:
                print(f"Range: {np.nanmin(ratio):.3e} to {np.nanmax(ratio):.3e}")
        
        # Extract diagonal elements for mode 2
        if len(ratio.shape) == 4:
            idx = np.arange(len(VGS))
            ratio = ratio[idx, idx, idx, idx]
        
        valid_idx = np.isfinite(ratio)
        ratio = ratio[valid_idx]
        VGS = VGS[valid_idx]
        
        if debug:
            print("\nAfter extracting valid ratios:")
            print(f"Valid points: {len(ratio)}")
            if len(ratio) > 0:
                print(f"Ratio range: {np.min(ratio):.3e} to {np.max(ratio):.3e}")
                print(f"VGS range: {np.min(VGS):.3f} to {np.max(VGS):.3f}")
        
        if len(ratio) == 0:
            print("Error: No valid ratio values after filtering")
            return np.array([])

    # Ensure ratio is 2D
    ratio = np.atleast_2d(ratio)
    if ratio.shape[0] == 1:
        ratio = ratio.T

    # Remove NaN values and sort for interpolation
    valid_idx = ~np.isnan(ratio.flatten())
    ratio_range = ratio.flatten()[valid_idx]
    VGS_range = VGS[valid_idx] if mode == 2 else VGS_values[valid_idx]

    if debug:
        print("\nInterpolation setup:")
        print(f"Ratio points: {len(ratio_range)}")
        print(f"Target ratio: {ratio_data[0]:.3e}")
        if len(ratio_range) > 0:
            print(f"Ratio range: {np.min(ratio_range):.3e} to {np.max(ratio_range):.3e}")
            print(f"VGS range: {np.min(VGS_range):.3f} to {np.max(VGS_range):.3f}")

    if len(ratio_range) < 2:
        print("Error: Not enough valid points for interpolation")
        return np.array([])

    # Sort arrays to ensure monotonic interpolation
    sort_idx = np.argsort(ratio_range)
    ratio_range = ratio_range[sort_idx]
    VGS_range = VGS_range[sort_idx]

    # Handle extrapolation for ID_W
    if ratio_string == 'ID_W' and ratio_data[0] > np.max(ratio_range):
        # Use the last two points for linear extrapolation
        slope = (VGS_range[-1] - VGS_range[-2]) / (ratio_range[-1] - ratio_range[-2])
        delta_ratio = ratio_data[0] - ratio_range[-1]
        result = VGS_range[-1] + slope * delta_ratio
        
        if debug:
            print("\nPerforming linear extrapolation:")
            print(f"Slope: {slope:.6f}")
            print(f"Delta ratio: {delta_ratio:.3e}")
            print(f"Extrapolated result: {result:.6f}")
            
        return np.array([result])

    # Normal interpolation for other cases
    try:
        if params['METHOD'].lower() == 'pchip':
            interpolator = PchipInterpolator(ratio_range, VGS_range, extrapolate=True)
        else:
            interpolator = interp1d(ratio_range, VGS_range, 
                                  kind=params['METHOD'], 
                                  fill_value='extrapolate')
        result = interpolator(ratio_data)
        if debug:
            print(f"\nInterpolation result: {result}")
        return np.array(result)
    except Exception as e:
        print(f"Interpolation error: {e}")
        return np.array([])

if __name__ == "__main__":
    from scipy.io import loadmat
    
    # Load the MATLAB data
    data = loadmat('nch_18.mat')
    nch_data = data['nch']
  
    print("\nTest Case 1:")
    result1 = lookup_vgs(nch_data, GM_ID=10, VDS=0.6, VSB=0.1, L=0.18)
    print("Result 1:", result1)

    print("\nTest Case 2:")
    result2 = lookup_vgs(nch_data, GM_ID=np.arange(10,15,1), VDS=0.6, VSB=0.1, L=0.18)
    print("Result 2:", result2)

    print("\nTest Case 3:")
    result3 = lookup_vgs(nch_data, ID_W=1e-4, VDS=0.6, VSB=0.1, L=0.18)
    print("Result 3:", result3)

    print("\nTest Case 4:")
    result4 = lookup_vgs(nch_data, ID_W=1e-4, VDB=0.6, VGB=1, L=0.3)
    print("Result 4:", result4)
    
    result5 = lookup_vgs(nch_data, GM_ID= 12, VDB=0.6, VGB=1, L=1.8)
    print("Result 5:", result5)
