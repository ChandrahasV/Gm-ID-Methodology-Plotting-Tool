import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_array(*arrays, canvas=None, ax1=None, ax2=None, x_label="", y1_label="", y2_label="", x_scale="", y1_scale="", y2_scale=""):
    """
    Generate a plot with dynamically colored axis labels matching the graphs .

    Parameters:
        *arrays: Input data arrays for plotting.
        canvas, ax1, ax2: Canvas and axes for embedding in GUI.
        x_label, y1_label, y2_label: Labels for x-axis, left y-axis (y1), and right y-axis (y2).
        x_scale, y1_scale, y2_scale: Scales for the respective axes.
    """
    arrays = [np.squeeze(np.array(arr)) for arr in arrays]
    x = arrays[0]
    y1 = arrays[1] if len(arrays) > 1 else None
    y2 = arrays[2] if len(arrays) > 2 else None

    # Clear axes for replotting
    if canvas and ax1 and ax2:
        ax1.clear()
        ax2.clear()

    # Set axis scales
    if x_scale == "log":
        ax1.set_xscale('log')
    if y1_scale == "log":
        ax1.set_yscale('log')
    if y2_scale == "log":
        ax2.set_yscale('log')

    # Plot data
    if y1 is not None:
        ax1.plot(x, y1, marker='o', color='red')  # Red for left y-axis
        ax1.set_ylabel(y1_label, color='red')  # Match axis label color
        ax1.tick_params(axis='y', colors='red')  # Match tick color

    if y2 is not None:
        ax2.plot(x, y2, marker='o', color='blue')  # Blue for right y-axis
        ax2.set_ylabel(y2_label, color='blue')  # Match axis label color
        ax2.tick_params(axis='y', colors='blue')  # Match tick color

    # X-axis settings
    ax1.set_xlabel(x_label)
    ax1.tick_params(axis='x')

    # Adjust layout for readability
    if y2 is None:
        ax1.figure.tight_layout()
        ax1.figure.subplots_adjust(bottom=0.2)
    else:
        ax1.figure.tight_layout()

    # Redraw canvas
    if canvas:
        canvas.draw()
    else:
        plt.show()

### Ignore the best plot function
def best_plot(x, y):
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    # Filter out non-positive values for log plots
    valid_log_mask = (x > 0) & (y > 0)
    x_log = x[valid_log_mask]
    y_log = y[valid_log_mask]
    
    # Initialize errors dictionary
    errors = {}
    
    # Linear plot fit
    coeffs_linear = np.polyfit(x, y, 1)
    y_fit_linear = np.polyval(coeffs_linear, x)
    errors['linear'] = mean_squared_error(y, y_fit_linear)
    
    # Log-log plot fit
    if len(x_log) > 0 and len(y_log) > 0:
        coeffs_loglog = np.polyfit(np.log(x_log), np.log(y_log), 1)
        y_fit_loglog = np.exp(np.polyval(coeffs_loglog, np.log(x_log)))
        errors['loglog'] = mean_squared_error(y_log, y_fit_loglog)
    
    # Semi-log x plot fit
    if len(x_log) > 0:
        coeffs_semilogx = np.polyfit(np.log(x_log), y_log, 1)
        y_fit_semilogx = np.polyval(coeffs_semilogx, np.log(x_log))
        errors['semilogx'] = mean_squared_error(y_log, y_fit_semilogx)
    
    # Semi-log y plot fit
    if len(y_log) > 0:
        coeffs_semilogy = np.polyfit(x_log, np.log(y_log), 1)
        y_fit_semilogy = np.exp(np.polyval(coeffs_semilogy, x_log))
        errors['semilogy'] = mean_squared_error(y_log, y_fit_semilogy)
    
    # Return the plot type with the lowest error
    return min(errors, key=errors.get)
