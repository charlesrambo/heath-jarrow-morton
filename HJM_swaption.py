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
    
    # Get the swap rates at expiration
    swaps_expiration = []
    
    for _ in range(num_vols):
        
        # Simulate path of rates
        path = hjm.simulate_hjm(latest_curve, maturities, vol_splines, dt = dt, 
                                num_steps = steps)
        
        # Get the forward rates at expiration
        fwd_expiry = path[-1]
        
        # Calculate the spot rates at expiration
        spot_expiry = hjm.forward_to_spot(maturities, fwd_expiry)
        
        # Use spot rates to calculate swap rate
        swap_rate, _ = get_swap_rate(maturities, spot_expiry, tenor)
        
        # Add to list
        swaps_expiration.append(swap_rate)
     
    # Convert to a numpy array
    swaps_expiration = np.array(swaps_expiration)
    
    # Clip negative rates so we can take logs
    swaps_expiration = np.clip(swaps_expiration, a_min = 1e-7)
      
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
    
    # Inilize list of payoffs
    payoffs = []
 
    for _ in range(num_sims):
        
        # Simulate forward curve evolution
        path = hjm.simulate_hjm(latest_curve, maturities, vol_splines, 
                            dt = dt, num_steps = steps)
        
        # Get the forward curve in one year
        fwd_at_expiry = path[-1]
        
        # Get spot curve at expiration
        spot_at_expiry = hjm.forward_to_spot(maturities, fwd_at_expiry)
        
        # Calculate realized swap rate and annuity
        swap_rate, annuity = get_swap_rate(maturities, spot_at_expiry, tenor)
        
        # Calculate payoff for payer swaption
        payoff = annuity * np.max([swap_rate - strike, 0])
        
        # Discount back using initial spot curve
        initial_spot = hjm.forward_to_spot(maturities, latest_curve)
        
        # Fit the spline to the current spot curve; linear boundary condition
        spot_spline = CubicSpline(maturities, initial_spot, bc_type = 'natural')
        
        # Use spline to get spot rate at expiration
        z_expiry = spot_spline(expiry)
        
        # Calculate the discount factor
        discount_factor = 1/(1 + z_expiry/2)**(2 * expiry)
        
        # Add discounted payoff to list
        payoffs.append(payoff * discount_factor)
        
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
                                          strike = strike)
    
    # Use the forward curve to get current spot rates
    initial_spot = hjm.forward_to_spot(maturities, latest_curve)
    swap_rate, annuity = get_swap_rate(maturities, initial_spot, tenor)
    
    # Simulate volatility of swap rate using HJM model
    swap_vol = estimate_black_vol_mc(latest_curve, maturities, vol_splines, 
                                     expiry, tenor, num_vols = 10_000)
    
    # Calculate the Black swaption price
    black_price = black_swaption(swap_rate, strike, expiry, swap_vol, 
                                     annuity)

    print(r"--- Results ---")
    print(f"Current {tenor:0.0f}Y Swap Rate: {swap_rate:.4%}")
    print(f"HJM Swaption Price: {hjm_price:.5f}")
    print(f"Black Price: {black_price:.5f}")
    
    print(f'\nThis program took {(time.perf_counter() - start_time)/20:.2f} minutes.')