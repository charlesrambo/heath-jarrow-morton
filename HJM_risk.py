# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:55:26 2026

@author: cramb
"""
import numpy as np
import time
from scipy.interpolate import CubicSpline
import HJM_NSS as hjm
import HJM_swaption as swaption


def calculate_swaption_horizon_risk(latest_curve, maturities, vol_splines, 
                                    expiry = 1.0, tenor = 5.0, strike = 0.04, 
                                    dt = 1/252, horizon = 1/12, alpha = 0.05,
                                    num_sims = 10_000, inner_sims = 500):
    """
    Calculates VaR/ES using HJM for both the path evolution and the re-pricing.
    """
    steps_to_horizon = int(horizon/dt)
    dt = horizon/steps_to_horizon
    
    # Calculate the initial price of the swaption
    price_initial, _ = swaption.price_swaption_mc(latest_curve, maturities, 
                                                  vol_splines, 
                                                  expiry = expiry, 
                                                  tenor = tenor, 
                                                  strike = strike, 
                                                  num_sims = num_sims)
    
    # Simulate HJM paths to horizon
    outer_paths = hjm.simulate_hjm_vectorized(latest_curve, maturities, 
                                              vol_splines, 
                                              dt = dt, 
                                              num_steps = steps_to_horizon, 
                                              num_sims = num_sims)
    
    # Calculate the funding factor for every path
    funding_accrual = np.exp(np.trapz(outer_paths[:, :, 0], dx = dt, axis = 0))
    
    # Only interested in the last path
    fwd_at_horizon = outer_paths[-1] 
    
    # Clear up RAM (could be many intermediate steps)
    del outer_paths
    
    # Initialize array to hold outpout
    horizon_pnl = np.zeros(num_sims)
    
    # Re-pricing
    for i in range(num_sims):
        
        # Calculate the swpation price
        price_at_horizon, _ = swaption.price_swaption_mc(fwd_at_horizon[i], 
                                                         maturities, vol_splines, 
                                                         expiry = expiry - horizon, 
                                                         tenor = tenor, 
                                                         strike = strike, 
                                                         num_sims = inner_sims)
        
        # pnl is price at horizon minus FV initial cost
        horizon_pnl[i] = price_at_horizon - price_initial * funding_accrual[i]

    # Sort values for VaR/ES calculation
    horizon_pnl = np.sort(horizon_pnl)

    # Get the VaR index; take floor to be conservative
    var_idx = int(num_sims * alpha)
    
    return np.mean(horizon_pnl), -horizon_pnl[var_idx], -np.mean(horizon_pnl[:var_idx])


def get_bond_price(forward_curve, maturities, bond_maturity, coupon_rate, current_time = 0.0):
    """
    Calculates bond price given an instantaneous forward curve.
    """
    # Convert forward curve to spot curve
    spot_rates = hjm.forward_to_spot(maturities, forward_curve)
    
    # Create spline using maturities and corresponding spot rates; linear extrapolation
    spot_spline = CubicSpline(maturities, spot_rates, bc_type = 'natural')
    
    # Determine remaining cash flows; assuming semi-annual payments
    cf_dates = np.arange(0.5, bond_maturity + 0.5, 0.5)
    remaining_cfs = cf_dates[cf_dates > current_time] - current_time
    
    # Use spot rates to get discount factors
    z_t = spot_spline(remaining_cfs)
    
    # Get discount factors
    discounts = 1/(1 + z_t/2)**(2 * remaining_cfs)
    
    # Sum up coupons and principal
    coupons = (coupon_rate/2) * 100 * np.sum(discounts)
    principal = 100 * discounts[-1]
    
    return coupons + principal


def calculate_bond_risk_metrics(latest_curve, maturities, vol_splines, 
                               bond_maturity = 10.0, coupon_rate = 0.04, 
                               dt = 1/252, horizon = 1/12, alpha = 0.05, 
                               num_sims = 10_000):
    """
    Calculates P&L VaR and ES for a coupon-bearing bond at a future horizon,
    accounting for carry (funding cost).
    """
    # Get the number of steps
    steps = int(horizon / dt)
    
    # Re calculate the step size
    dt = horizon/steps
    
    # Calculate the initial price
    price_initial = get_bond_price(latest_curve, maturities, bond_maturity, 
                                   coupon_rate, current_time = 0)
    
    # Generate the outer paths
    outer_paths = hjm.simulate_hjm_vectorized(latest_curve, maturities, 
                                              vol_splines, dt = dt, 
                                              num_steps = steps, 
                                              num_sims = num_sims)
    
    # Calculate the funding factor for every path
    funding_accrual = np.exp(np.trapz(outer_paths[:, :, 0], dx = dt, axis = 0))
    
    # Calculate P&L at Horizon
    fwd_at_horizon = outer_paths[-1] 
    
    # Pre-calculate coupon timing (
    cf_dates = np.arange(0.5, bond_maturity + 0.5, 0.5)
    
    # Identify which coupons actually fall within our window
    coupons_in_horizon = cf_dates[cf_dates <= horizon]
    coupon_value = (coupon_rate / 2) * 100
    
    # Initialize array to hold results
    bond_pnl = np.zeros(num_sims)

    for i in range(num_sims):
        
        # Calculate bond value at horizon
        price_at_horizon = get_bond_price(fwd_at_horizon[i], maturities, 
                                          bond_maturity, coupon_rate, 
                                          current_time = horizon)
        
        # Initialize future value of coupons
        coupon_payout_fv = 0
        
        for coupon_date in coupons_in_horizon:
            
            # Calculate exactly which index in outer_paths corresponds to the payment date
            start_step = int(coupon_date/dt)
        
            # Pull the short-rate path from the moment of payment to the end of the horizon
            path_rates = outer_paths[start_step:, i, 0]
        
            # Calculate the reinvestment factor 
            reinvestment_factor = np.exp(np.trapz(path_rates, dx = dt))
            
            # Add future value of coupon payment to future values
            coupon_payout_fv += coupon_value * reinvestment_factor
        
        # pnl is price at horizon minus FV initial cost
        bond_pnl[i] = price_at_horizon + coupon_payout_fv - (price_initial * funding_accrual[i])

    # Sort prices to get VaR and ES
    bond_pnl = np.sort(bond_pnl)
    
    # Get index for VaR; take floor to be conservative
    var_idx = int(num_sims * alpha)
    
    # Return Mean P&L, VaR (as loss), and ES (as loss)
    return np.mean(bond_pnl), -bond_pnl[var_idx], -np.mean(bond_pnl[:var_idx])


if __name__ == "__main__":
    
    # Start the clock!
    start_time = time.perf_counter()
    
    # Get the forward rate data
    fwd_data = hjm.load_fed_data(hjm.url, hjm.headers)
    
    # Get the volatilities
    maturities, vol_splines, pca = hjm.get_hjm_volatility(fwd_data)
    
    # Get the last forward curve
    latest_curve = fwd_data.iloc[-1].to_numpy()
    
    # Define the significance level amd horizon for risk metrics
    alpha, horizon = 0.05, 1/4
    
    # Define parameters for swaption
    tenor, expiry, strike = 5.0, 1.0, 0.04
    
    # Get the swap rate
    swap_rate, _ = swaption.get_swap_rate(maturities, 
                                          hjm.forward_to_spot(maturities, latest_curve), 
                                          swap_tenor = tenor)
    
    # Get the swaption results
    swaption_pnl, swaption_var, swaption_es = calculate_swaption_horizon_risk(
                                                latest_curve, maturities, 
                                                vol_splines, expiry = expiry, 
                                                tenor = tenor, strike = strike, 
                                                horizon = horizon, alpha = alpha)
    
    # Define parameters for bond
    coupon_rate, bond_maturity = 0.04, 10.0
    
    # Get the bond results
    bond_pnl, bond_var, bond_es = calculate_bond_risk_metrics(
                                    latest_curve, maturities, 
                                    vol_splines, bond_maturity = bond_maturity, 
                                    coupon_rate = coupon_rate, horizon = horizon, 
                                    alpha = alpha)

    print("=" * 50)
    print(f"{'HJM RISK MANAGEMENT REPORT':^50}")
    print("=" * 50)
    print(f"Current {tenor:0.0f}Y Swap Rate:  {swap_rate:.4%}")
    print(r"Risk Horizon:          1 Quarter")
    print(r"Confidence Level:      95.0% (Alpha = 0.05)")
    print("-" * 50)
    
    # Compare swaption vs. bond
    print(f"{'Metric':<20} | {'Swaption':<12} | {'10Y Bond':<12}")
    print("-" * 50)
    print(f"{'Mean P&L':<20} | {swaption_pnl:>12.4f} | {bond_pnl:>12.4f}")
    print(f"{'Value-at-Risk':<20} | {swaption_var:>12.4f} | {bond_var:>12.4f}")
    print(f"{'Expected Shortfall':<20} | {swaption_es:>12.4f} | {bond_es:>12.4f}")
    print("-" * 50)
    
    print(f"Simulation completed in {(time.perf_counter() - start_time)/60:.2f} minutes.")
    print("=" * 50)