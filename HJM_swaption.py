# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:52:41 2026

@author: cramb
"""
import numpy as np
import time
from scipy.stats import norm
from scipy.interpolate import CubicSpline
import HJM_NSS as hjm


def get_swap_rate(maturities, spot_curve, swap_tenor):
    """
    Calculates the par swap rate for a given tenor from a spot curve,
    using semi-annual compounding conventions.
    """
    
    # Get the payments which we assume are semi-annual
    payment_dates = np.arange(0.5, swap_tenor + 0.5, 0.5)
    
    # Fit the spline to the current spot curve; linear boundary condition
    spot_spline = CubicSpline(maturities, spot_curve, bc_type = 'natural')
    
    # Use spline to interpolate spot rates at payment dates
    z_t = spot_spline(payment_dates)
    
    # Discount factors
    discount_factors = 1/(1 + z_t/2)**(2*payment_dates)
    
    # Calculate annuity value
    annuity = 0.5 * np.sum(discount_factors)
    
    # Par swap rate
    swap_rate = (1 - discount_factors[-1]) / annuity
    
    return swap_rate, annuity


def black_swaption(swap_rate, strike, expiry, vol, annuity):
    """Standard Black formula for a payer swaption."""
    
    # Calculate d1; same as Black-Scholes with risk-free rate of zero
    d1 = (np.log(swap_rate/strike) + 0.5 * vol**2 * expiry)/(vol * np.sqrt(expiry))
    
    # Calculate d2 just like Black-Scholes
    d2 = d1 - vol * np.sqrt(expiry)
    
    # Black formula
    return annuity * (swap_rate * norm.cdf(d1) - strike * norm.cdf(d2))


def estimate_black_vol_mc(latest_curve, maturities, vol_splines, expiry, tenor, 
                          dt = 1/252, num_vols = 10_000):
    """
    Estimates the implied Black volatility by simulating the forward swap rate
    evolution until the option expiry.
    """
    
    # Get the initial spot curve
    initial_spot = hjm.forward_to_spot(maturities, latest_curve)
    
    # Use initial spot curve to get initial swap rate
    initial_swap_rate, _ = get_swap_rate(maturities, initial_spot, 
                                                    tenor)
    
    # Define the number of steps
    steps = int(expiry/dt)
    
    # Recalculate dt
    dt = expiry/steps
    
    # All paths shap
    all_paths = hjm.simulate_hjm_vectorized(latest_curve, maturities, 
                                                 vol_splines, dt = dt, 
                                                 num_steps = steps, 
                                                 num_sims = num_vols)
    
    # Get forward rates at expiry (last time step)
    fwd_expiry = all_paths[-1] 
    
    # Initialize list to hold swap rates
    swaps_expiration = []
    
    # Calculate swap rates 
    for i in range(num_vols):
        
        spot_expiry = hjm.forward_to_spot(maturities, fwd_expiry[i])
        
        swap_rate, _ = get_swap_rate(maturities, spot_expiry, tenor)
        
        swaps_expiration.append(swap_rate)
     
    # Convert to a numpy array
    swaps_expiration = np.array(swaps_expiration)
    
    # Clip negative rates so we can take logs
    swaps_expiration = np.clip(swaps_expiration, a_min = 1e-7, a_max = None)
      
    # Calculate the log returns
    log_returns = np.log(swaps_expiration/initial_swap_rate)
    
    # Use log returns to calculate volatility
    return np.std(log_returns, ddof = 1)/np.sqrt(expiry)


def price_swaption_mc(latest_curve, maturities, vol_splines, expiry = 1.0, 
                      tenor = 5.0, strike = 0.04, num_sims = 10_000, dt = 1/252):
    """
    Prices a European Payer Swaption using Monte Carlo path simulation.
    """
    
    # Define the number of steps
    steps = int(expiry/dt)
    
    # Recalculate dt
    dt = expiry/steps
    
    # Run full simulation 
    all_paths = hjm.simulate_hjm_vectorized(latest_curve, maturities, 
                                                 vol_splines, dt = dt, 
                                                 num_steps = steps, 
                                                 num_sims = num_sims)
    
    # Calculate discount factors 
    discount_factors = np.exp(-np.trapz(all_paths[:, :, 0], dx = dt, axis = 0))
    
    # The last simulated curve
    fwd_at_expiry = all_paths[-1] 
    
    # Initialize list to hold payoffs
    payoffs = []
 
    for i in range(num_sims):
        
        # Convert forward curve to spot curve
        spot_at_expiry = hjm.forward_to_spot(maturities, fwd_at_expiry[i])
        
        # Get swap rate and annuity
        swap_rate, annuity = get_swap_rate(maturities, spot_at_expiry, tenor)
        
        # Calculate discounted payoff
        payoff = annuity * np.max([swap_rate - strike, 0])
        payoffs.append(payoff * discount_factors[i])
        
    return np.mean(payoffs), np.std(payoffs, ddof = 1)/np.sqrt(num_sims)


if __name__ == "__main__":
    
    # Start the clock!
    start_time = time.perf_counter()
    
    # Get the forward rate data
    fwd_data = hjm.load_fed_data(hjm.url, hjm.headers)
    
    # Get the volatilities
    maturities, vol_splines, pca = hjm.get_hjm_volatility(fwd_data)
    
    # Get the last forward curve
    latest_curve = fwd_data.iloc[-1].to_numpy()
    
    # Define parameters
    expiry, tenor, strike = 1.0, 5.0, 0.04
    
    # HJM Price and standard error of estimate
    hjm_price, se = price_swaption_mc(latest_curve, maturities, vol_splines, 
                                          expiry = expiry, tenor = tenor, 
                                          strike = strike, num_sims = 100_000)
    
    # Use the forward curve to get current spot rates
    initial_spot = hjm.forward_to_spot(maturities, latest_curve)
    swap_rate, annuity = get_swap_rate(maturities, initial_spot, tenor)
    
    # Simulate volatility of swap rate using HJM model
    swap_vol = estimate_black_vol_mc(latest_curve, maturities, vol_splines, 
                                     expiry, tenor, num_vols = 100_000)
    
    # Calculate the Black swaption price
    black_price = black_swaption(swap_rate, strike, expiry, swap_vol, 
                                     annuity)

    print(r"--- Results ---")
    print(f"Current {tenor:0.0f}Y Swap Rate: {swap_rate:.4%}")
    print(f"HJM Swaption Price: {hjm_price:.5f}")
    print(f"Black Price: {black_price:.5f}")
    
    print(f'\nThis program took {(time.perf_counter() - start_time)/20:.2f} minutes.')