# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:23:30 2026

@author: cramb
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid


# The URL to the CSV file corresponding to paper feds200628
url = r"https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

# Define a standard browser header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# Create function to get NSS forward rate
def get_NSS_forward(m, beta0, beta1, beta2, beta3, tau1, tau2):
    
    # Constant term
    term0 = beta0
    
    # First term (exponential decay)
    term1 = beta1 * np.exp(-m/tau1)
    
    # Second term (first hump)
    term2 = beta2 * (m/tau1) * np.exp(-m/tau1)
    
    # Third term (second hump)
    term3 =  beta3 * (m/tau2) * np.exp(-m/tau2)
    
    return term0 + term1 + term2 + term3
    

def load_fed_data(url, headers, skiprows = 9):
    
    # Read data skipping the description rows (first 9 rows are headers/metadata)
    df = pd.read_csv(url, skiprows = skiprows, storage_options = headers)

    # Convert date column to date time object
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Set the date as the index
    df = df.set_index("Date")
    
    # Get the short-rate
    df["SVENF0"] = get_NSS_forward(0, df["BETA0"], df["BETA1"], df["BETA2"], 
                                   df["BETA3"], df["TAU1"], df["TAU2"])
    
    # Get the six-month rate
    df["SVENF0.5"] = get_NSS_forward(0.5, df["BETA0"], df["BETA1"], df["BETA2"], 
                                   df["BETA3"], df["TAU1"], df["TAU2"])
    
    # Get the 40-year rate; helpful to price long tenor maturies as we roll along curve
    df["SVENF40"] = get_NSS_forward(0.5, df["BETA0"], df["BETA1"], df["BETA2"], 
                                   df["BETA3"], df["TAU1"], df["TAU2"])

    # Extract Instantaneous Forward Rates (prefixed with SVENF)
    forward_cols = [col for col in df.columns if col.startswith("SVENF")]

    # Sort columns by maturity to ensure sequence
    forward_cols.sort(key = lambda x: float(x.replace("SVENF", "")))

    # Change order and drop missing values
    fwd_data = df[forward_cols].dropna()

    # Convert percentage to decimals
    fwd_data = fwd_data / 100.0

    return fwd_data


def get_hjm_volatility(fwd_data, num_components = 0.95, lam = None):
    
    # Calculate change in forward rates
    diff_data = fwd_data.diff().dropna()

    # Fit PCA
    pca = PCA(n_components = num_components)
    pca.fit(diff_data)

    # Get the number of components
    num_factors = pca.n_components_
    
    # Extract numerical maturities directly from the column names
    maturities = np.array([float(col.replace("SVENF", "")) for col in fwd_data.columns])

    # Calculate raw annualized volatilities; scale by sqrt of eigenvalues and sqrt(252) to annualize
    raw_vols = pca.components_.T * np.sqrt(pca.explained_variance_) * np.sqrt(252)
    
    # Store the actual callable objects
    vol_splines = []
    
    for i in range(num_factors):
        
        # Create the spline
        spline = make_smoothing_spline(maturities, raw_vols[:, i], lam = lam)
        
        # Save results
        vol_splines.append(spline)

    return maturities, vol_splines, pca


def get_hjm_drift(maturities, vol_splines):
    """
    Function to calculate HJM no arbitrage drift term.
    """
    
    # Initialize array to hold drift terms
    drift = np.zeros_like(maturities)
    
    for vol_func in vol_splines:
        
        # Get the volatilities
        sigma = vol_func(maturities)
        
        # Use the spline's own integration capability for better precision
        drift += sigma * (vol_func.antiderivative()(maturities) - vol_func.antiderivative()(0))
        
    return drift
    
    
def simulate_hjm(fwd_curve, maturities, vol_splines, dt = 1/252, num_steps = 252):
    """Simulates the evolution of the forward rate curve over time.

    fwd_curve: Initial forward rate curve (1D array)
    maturities: Maturity vector corresponding to the curve 
    vol_splines: volatility components represented as spline functions
    dt: Time step
    (daily) num_steps: Number of steps to simulate
    """
    
    # Get dimensions
    num_factors = len(vol_splines)
    num_maturities = len(maturities)

    # Ensure fwd_curve is a numpy array (handles pandas Series/DataFrame input)
    if hasattr(fwd_curve, "to_numpy"):
        current_curve = fwd_curve.to_numpy().flatten()  
    else:
        current_curve = np.array(fwd_curve).copy()
        
    # Calculate the drift at maturities + dt
    drift = get_hjm_drift(maturities, vol_splines)
    
    # Initialize list to hold simulated curve
    simulated_curves = [current_curve]

    # Monte Carlo simulation
    for i in range(num_steps):
        
        # Generate random normal shocks for the factors
        shocks = np.random.normal(0, 1, num_factors)

        # Initialize array to hold defusion terms
        diffusion = np.zeros(num_maturities)
        
        for i, vol_func in enumerate(vol_splines):
            
            # Evaluate the volatility spline at the specific maturities
            diffusion += vol_func(maturities + dt) * shocks[i] * np.sqrt(dt)

        # Initialize new curve
        new_curve = np.zeros(num_maturities)
        
        # Use linear boundary conditions to handle the long end
        curve_spline = CubicSpline(maturities, current_curve, bc_type = 'natural')
        
        # Interpolate the current curve to "age" it by dt
        rolled_rates = curve_spline(maturities + dt)

        # Rolldown + Drift + Diffusion
        new_curve = rolled_rates + drift * dt + diffusion

        # Append results to simulated curves
        simulated_curves.append(new_curve)
        
        # Update the current curve
        current_curve = new_curve

    return np.array(simulated_curves)


def simulate_hjm_vectorized(fwd_curve, maturities, vol_splines, dt = 1/252, 
                           num_steps = 252, num_sims = 10_000):
    """
    Vectorized HJM simulation.
    Weird AI vectorization techniques.
    Returns:
        results: numpy array of shape (num_steps + 1, num_sims, num_maturities)
    """
    num_factors = len(vol_splines)
    num_maturities = len(maturities)
    
    # Initialize array to hold drift terms
    drift = np.zeros(num_maturities)
            
    for vol_func in vol_splines:
        
        # Get the volatilities
        sigma = vol_func(maturities)
        
        # Use the spline's own integration capability for better precision
        drift += sigma * (vol_func.antiderivative()(maturities) - vol_func.antiderivative()(0))

    # Compute Aging Matrix
    # This transforms the curve from time t to t + dt via cubic interpolation
    basis = np.eye(num_maturities)
    
    W = np.zeros((num_maturities, num_maturities))
    
    for i in range(num_maturities):
        
        spline = CubicSpline(maturities, basis[i], bc_type = 'natural')
        
        W[i, :] = spline(maturities + dt)

    # Prepare Volatility Matrix
    vol_matrix = np.array([vol_func(maturities) for vol_func in vol_splines])

    # Convert fwd_curve to array if it's a Series
    base_curve = np.array(fwd_curve).flatten()
    current_curves = np.tile(base_curve, (num_sims, 1))
    
    # Storage for all paths: (steps+1, sims, mats)
    results = np.zeros((num_steps + 1, num_sims, num_maturities))
    results[0] = current_curves

    # Simulation Loop
    drift_dt = drift * dt
    sqrt_dt = np.sqrt(dt)
    
    for t in range(1, num_steps + 1):
        
        # Generate shocks for this step
        shocks = np.random.normal(0, 1, (num_sims, num_factors))
        
        # Diffusion
        diffusion = (shocks @ vol_matrix) * sqrt_dt
        
        # Vectorized step
        current_curves = (current_curves @ W) + drift_dt + diffusion
        
        # Add to results
        results[t] = current_curves

    return results


def forward_to_spot(maturities, forward_curve):
    """
    Converts an instantaneous forward curve to a spot curve.
    
    maturities: array of years (e.g., [0.08, 0.25, ..., 30])
    forward_curve: array of instantaneous forward rates
    """
    # Calculate the integral using the trapezoidal rule
    integral = cumulative_trapezoid(forward_curve, x = maturities)
    
    # R(t, T) = (1/T) * Integral from t to T of f(t, tau) wrt tau
    spot_curve = integral / maturities[1:]
        
    # Usually, the 0-maturity spot rate equals the 0-maturity forward rate
    spot_curve = np.insert(spot_curve, 0, forward_curve[0])
    
    # Make semiannual rates
    spot_curve = 2 * (np.exp(spot_curve/2) - 1)
    
    return spot_curve


def plot_scree(pca, threshold = 0.95, figsize = (10, 6)):
    """
    Generates a scree plot for a fitted sklearn PCA object.
    """
    # Get variance ratios
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    
    # Create labels for the x-axis
    labels = [f"PC{i+1}" for i in range(len(exp_var))]
    
    fig, ax1 = plt.subplots(figsize = figsize)
    
    # Plot Individual Explained Variance (Bars)
    bars = ax1.bar(labels, exp_var, 
                   alpha = 0.8, 
                   label = 'Individual Variance')
    ax1.set_ylabel('Explained Variance Ratio', fontsize = 12)
    ax1.set_xlabel('Principal Components', fontsize = 12)
    ax1.set_title('Scree Plot: Explained Variance by Component', fontsize = 14)
    
    # Leave space for labels
    ax1.set_ylim(0, 1.1) 
    
    # Add percentage labels on top of bars
    for bar in bars:
        
        height = bar.get_height()
        
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha = 'center', va = 'bottom')

    # Plot Cumulative Explained Variance (Line)
    ax2 = ax1.twinx()
    ax2.plot(labels, cum_var, 
             color = 'navy', 
             marker = 'o', markersize = 8, 
             linewidth = 2, label = 'Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance', fontsize = 12)
    ax2.set_ylim(0, 1.1)
    
    # Add a horizontal line at the 95% threshold
    ax2.axhline(y = threshold, color = 'red', linestyle = '--', alpha = 0.7, 
                label = f'{threshold:0.0%} Threshold')
    
    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc = 'center right')
    
    plt.grid(axis = 'y', linestyle = ':', alpha = 0.7)
    plt.tight_layout()
    plt.show()
   
    
if __name__ == "__main__":
    
    # Set the random seed
    np.random.seed(0)
    
    # Define a threshold
    threshold = 0.95
    
    print("\nFetching data from the Federal Reserve...")
    fwd_data = load_fed_data(url, headers)
    
    print("\nRunning PCA to compute HJM volatility parameters...")
    maturities, vol_splines, pca = get_hjm_volatility(fwd_data, 
                                                      num_components = threshold)
    
    # Add a scree plot
    plot_scree(pca, threshold = threshold)

    # Plot the Volatility Structures (Factor Loadings)
    plt.figure(figsize = (10, 6))
    
    for i, spline in enumerate(vol_splines):
        
        plt.plot(maturities, spline(maturities), label = f"Factor {i+1}")
        
    plt.title("PCA Resolved HJM Volatility Functions (Annualized)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Get last forward curve
    latest_curve = fwd_data.iloc[-1].to_numpy()
    
    
    print(
        f"\nSimulating forward curve evolution starting from {fwd_data.index[-1].date()}..."
    )

    # Run the simulation
    simulation_paths = simulate_hjm(latest_curve, maturities, vol_splines)

    # Plot initial forward curve vs simulated forward curve after 1 year (252 steps)
    plt.figure(figsize = (10, 6))
    
    plt.plot(maturities, latest_curve, label = "Initial Forward Curve")
    plt.plot(
            maturities,
            simulation_paths[-1],
            label = "Simulated Forward Curve (1 Year out)",
            linestyle = "--",
    )
    plt.title("HJM Forward Curve Simulation")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Forward Rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot initial spot curve vs simulated spot curve after 1 year (252 steps)
    plt.figure(figsize = (10, 6))
    plt.plot(maturities, forward_to_spot(maturities, latest_curve), label = "Initial Spot Curve")
    plt.plot(
        maturities,
        forward_to_spot(maturities, simulation_paths[-1]),
        label = "Simulated Spot Curve (1 Year out)",
        linestyle = "--",
    )
    plt.title("HJM Spot Curve Simulation")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Spot Rate")
    plt.legend()
    plt.grid(True)
    plt.show()