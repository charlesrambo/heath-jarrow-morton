# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:33:11 2026

@author: cramb
"""
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import cumulative_trapezoid

def load_cmt_data():
    
    # List of all tenors for a high-res curve
    tenors = ['DFF', 'SOFR', 'DGS1MO', 'DGS3MO', 'DGS6MO', 
              'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
    
    # Get the current date
    end_date = dt.datetime.now()
    
    # Go back tem years
    start_date = end_date - dt.timedelta(days = 10 * 365)
    
    # Make the month January and the day 1
    start_date = start_date.replace(month = 1, day = 1)
    
    # Fetch the last 5 years of daily data
    cmt_data = web.DataReader(tenors, 'fred', start_date, end_date )
    
    # Fill missing SOFR rates with DFF
    cmt_data['SOFR'] = cmt_data['SOFR'].fillna(cmt_data['DFF'])
    
    # Drop column that has done its job
    cmt_data = cmt_data.drop(columns = ['DFF'])
    
    # Drop missing values
    cmt_data = cmt_data.dropna()

    # Convert percentage to decimals
    cmt_data = cmt_data / 100.0
    
    # Rename columns for clarity in your HJM script
    cmt_data.columns = ['0Y', '0.083Y', '0.25Y', '0.5Y', 
                        '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    
    return cmt_data


# We need a way to interpolate spot rates we've already found
def get_spot_interp(t, spot_rates):
    
    keys = sorted(spot_rates.keys())
    
    values = [spot_rates[k] for k in keys]
    
    f = interp1d(keys, values, kind = 'linear', fill_value = "extrapolate")
    
    return float(f(t))


def bootstrap_treasuries(cmt_rates, cmt_maturities):
    """
    cmt_rates: array of yields (e.g., 0.04 for 4%)
    cmt_maturities: array of years [0.083, 0.25, 0.5, 1.0, 2.0, ...]
    """
    # Initialize spot rates dict with the zero-coupon "Bills" section
    spot_rates = {}
    
    for r, T in zip(cmt_rates, cmt_maturities):
        
        # Treasures one year or less don't have coupons
        if T <= 1.0:
            
            spot_rates[T] = r
            
        else:
            
            # Semi-annual coupon payment dates
            coupon_times = np.arange(0.5, T + 0.5, 0.5)
            
            # Par curve so if coupon is rate PV is 100
            coupon_amt = (r / 2) * 100
            
            # Calculate PV of all coupons except the final one
            pv_coupons = 0
            
            for t in coupon_times[:-1]:
                
                # Get the spot rate for t; if not in dictionary then it's interpolated
                z_t = get_spot_interp(t, spot_rates)
                
                # Add discounted coupon value
                pv_coupons += coupon_amt / (1 + z_t / 2)**(2 * t)
                
            # Solve for the final spot rate z(T)
            z_T = 2 * (((100 + coupon_amt) / (100 - pv_coupons))**(1 / (2 * T)) - 1)
            
            spot_rates[T] = z_T
        
    return spot_rates


def get_hjm_forward_data(cmt_data, eval_maturities = None):
    """
    Leverages your bootstrap_treasuries to create a continuous forward curve
    for each historical date in cmt_data.
    """
    
    # Parse maturities from the columns once
    maturities = np.array([float(col.replace('Y', '')) for col in cmt_data.columns])
    
    # Get maturities for forward curve
    if eval_maturities is None:
        eval_maturities = maturities
        
    
    # Initialize list to hold forward curves
    fwd_curves = []

    for i in range(len(cmt_data)):
        
        # Get semi-annual spot rates for row
        spot_dict = bootstrap_treasuries(cmt_data.iloc[i].values, maturities)
        
        # Sort by the maturity (the key) and unpack into two tuples
        sorted_mats, z_sa = zip(*sorted(spot_dict.items()))

        # Convert to arrays for the math
        sorted_mats = np.array(sorted_mats)
        z_sa = np.array(z_sa)
        
        # Convert to continuous compounding
        z_c = 2 * np.log(1 + z_sa / 2)
        
        spline = make_smoothing_spline(sorted_mats, z_c)
        
        # Instantaneous Forward Formula
        f_0_T = spline(eval_maturities) + np.array(eval_maturities) * spline.derivative()(eval_maturities)
        
        # Add forward rates to list
        fwd_curves.append(f_0_T)
        
    # Create new columns
    new_columns = [f"{round(m, 3)}Y" for m in eval_maturities]

    # Rebuild as a DataFrame with your original column names
    fwd_data =  pd.DataFrame(fwd_curves, index = cmt_data.index, 
                        columns = new_columns)
    
    return fwd_data


def get_hjm_volatility(fwd_data, n_components = 0.95, lam = None):
    
    # Calculate change in forward rates
    diff_data = fwd_data.diff().dropna()

    # Fit PCA
    pca = PCA(n_components = n_components)
    pca.fit(diff_data)

    # Get the number of components
    n_factors = pca.n_components_
    
    # Extract numerical maturities directly from the column names
    maturities = np.array([float(col.replace("Y", "")) for col in fwd_data.columns])

    # Calculate raw annualized volatilities; scale by sqrt of eigenvalues and sqrt(252) to annualize
    raw_vols = pca.components_.T * np.sqrt(pca.explained_variance_) * np.sqrt(252)
    
    # Store the actual callable objects
    vol_splines = []
    
    for i in range(n_factors):
        
        # Create the spline
        spline = make_smoothing_spline(maturities, raw_vols[:, i], lam = lam)
        
        # Save results
        vol_splines.append(spline)

    return maturities, vol_splines, pca


def simulate_hjm(fwd_curve, maturities, vol_splines, dt = 1/252, n_steps = 252):
    """Simulates the evolution of the forward rate curve over time.

    fwd_curve: Initial forward rate curve (1D array)
    maturities: Maturity vector corresponding to the curve vols: PCA
    volatility components (shape: n_maturities, n_factors) dt: Time step
    (daily) n_steps: Number of steps to simulate
    """
    
    # Get dimensions
    n_factors = len(vol_splines)
    n_maturities = len(maturities)

    # Ensure fwd_curve is a numpy array (handles pandas Series/DataFrame input)
    if hasattr(fwd_curve, "to_numpy"):
        current_curve = fwd_curve.to_numpy().flatten()  
    else:
        current_curve = np.array(fwd_curve).copy()
    
    # Initialize array to hold drift terms
    drift = np.zeros(n_maturities)

    for m, T in enumerate(maturities):
        
        for vol_func in vol_splines:
            
            # Use the spline's own integration capability for better precision
            # drift = sigma(T) * integral from 0 to T of sigma(s)ds
            sigma_T = vol_func(T)
            integral = vol_func.antiderivative()(T) - vol_func.antiderivative()(0)
            drift[m] += sigma_T * integral

    # Initialize list to hold simulated curve
    simulated_curves = [current_curve]

    # Monte Carlo simulation
    for i in range(n_steps):
        
        # Generate random normal shocks for the factors
        shocks = np.random.normal(0, 1, n_factors)

        # Initialize array to hold defusion terms
        diffusion = np.zeros(n_maturities)
        
        for i, vol_func in enumerate(vol_splines):
            
            # Evaluate the volatility spline at the specific maturities
            diffusion += vol_func(maturities) * shocks[i] * np.sqrt(dt)

        # Initialize new curve
        new_curve = np.zeros(n_maturities)
        
        # Interpolate the current curve to "age" it by dt
        # Use 'natural' or 'clamped' boundary conditions to handle the long end
        curve_spline = CubicSpline(maturities, current_curve, bc_type = 'natural')
        rolled_rates = curve_spline(maturities + dt)

        # Rolldown + Drift + Diffusion
        new_curve = rolled_rates + drift * dt + diffusion

        # Append results to simulated curves
        simulated_curves.append(new_curve)
        
        # Update the current curve
        current_curve = new_curve

    return np.array(simulated_curves)


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


if __name__ == "__main__":
    
    # Set the random seed
    np.random.seed(0)
    
    print("\nFetching data from the Fred...")
    cmt_data = load_cmt_data()
    
    # Specify maturities for forward curve
    eval_maturities = np.append([0, 1/12, 1/4, 1/2, 3/4], np.arange(1, 31))
    
    print("\nConverting par curve data to forward rates...")
    fwd_data = get_hjm_forward_data(cmt_data, eval_maturities)

    print("\nRunning PCA to compute HJM volatility parameters...")
    maturities, vol_splines, pca = get_hjm_volatility(fwd_data, 
                                                      n_components = 0.95)

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