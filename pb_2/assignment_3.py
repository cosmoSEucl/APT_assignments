import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, f
import matplotlib.pyplot as plt

# -------------------------
# 1) Data load & cleaning
# -------------------------
def load_and_clean(portfolios_csv='data/25_Portfolios_5x5.csv',
                   factors_csv='data/F-F_Research_Data_Factors.csv',
                   start_date='1932-01-01',
                   end_date=None,  # <-- ADDED PARAMETER
                   num_factors=3):
    """
    Load and clean Fama-French portfolio and factor data
    
    Parameters:
    -----------
    ...
    end_date : str or None
        End date for analysis (format: 'YYYY-MM-DD'). If None, uses all data to the end.
    ...
    """
    # Load files exactly as in your original code
    portfolio_SBM = pd.read_csv(portfolios_csv, skiprows=15, nrows=1188)
    fama_french_factors = pd.read_csv(factors_csv, skiprows=3, nrows=1188)

    # Clean data exactly as in your original code
    portfolio_SBM = portfolio_SBM.dropna(how='all')
    portfolio_SBM['date'] = pd.to_datetime(portfolio_SBM.iloc[:,0], format='%Y%m')
    portfolio_SBM.set_index('date', inplace=True)
    
    fama_french_factors['date'] = pd.to_datetime(fama_french_factors.iloc[:,0], format='%Y%m')
    fama_french_factors.set_index('date', inplace=True)

    # Eliminate data prior to start_date
    portfolio_SBM = portfolio_SBM[portfolio_SBM.index >= start_date]
    fama_french_factors = fama_french_factors[fama_french_factors.index >= start_date]

    # --- NEW CODE BLOCK ---
    # Eliminate data after end_date if one is provided
    if end_date:
        portfolio_SBM = portfolio_SBM[portfolio_SBM.index <= end_date]
        fama_french_factors = fama_french_factors[fama_french_factors.index <= end_date]
    # --- END NEW CODE BLOCK ---

    # Remove the first column which is the MONTH string column (now index), keep returns columns
    portfolio_returns = portfolio_SBM.iloc[:, 1:].copy()

    # Subtract RF from portfolio returns to get excess returns
    portfolio_returns = portfolio_returns.subtract(fama_french_factors['RF'], axis=0)

    # Prepare factor matrix based on num_factors
    if num_factors == 1:
        F_t = fama_french_factors[['Mkt-RF']].copy()
    elif num_factors == 3:
        F_t = fama_french_factors[['Mkt-RF', 'SMB', 'HML']].copy()
    else:
        raise ValueError("num_factors must be 1 or 3")

    return portfolio_returns, F_t

# -------------------------
# 2) Mean returns table
# -------------------------
def mean_returns_table(portfolio_returns):
    """Compute mean returns and reshape into 5x5 matrix"""
    mean_returns = portfolio_returns.mean()
    mean_matrix = mean_returns.values.reshape(5,5)
    return mean_returns, mean_matrix

# -------------------------
# 3) Time-series regressions
# -------------------------
def time_series_regressions(portfolio_returns, F_t):
    """
    Run OLS time-series regressions for each portfolio:
      r_it = alpha_i + beta_i' f_t + resid_it
    Returns:
      results_df: DataFrame with Portfolio, Alpha, Betas, SEs
      all_residuals: list of residual arrays (one per portfolio)
    """
    results = []
    alpha_se = []
    beta_se_list = []
    all_residuals = []
    r_squared_list = []

    # Prepare factor matrix for regression (T x K)
    F_vals = F_t.values  # shape (T, K)

    for portfolio in portfolio_returns.columns:
        y = portfolio_returns[portfolio].values
        X = sm.add_constant(F_vals)  # adds intercept column
        model = sm.OLS(y, X).fit()

        alpha = model.params[0]
        betas = model.params[1:]
        results.append({'Portfolio': portfolio, 'Alpha': alpha, 'Betas': betas})
        alpha_se.append(model.bse[0])
        beta_se_list.append(model.bse[1:])
        all_residuals.append(model.resid)
        r_squared_list.append(model.rsquared)

    results_df = pd.DataFrame(results)
    results_df['Alpha SE'] = alpha_se
    results_df['R_squared'] = r_squared_list
    
    # add beta SE columns
    for k in range(len(beta_se_list[0])):
        results_df[f'Beta{k+1} SE'] = [b[k] for b in beta_se_list]

    return results_df, all_residuals, F_vals

# -------------------------
# 4) Chi-squared / F test (Time-series)
# -------------------------
def chi2_test(results_df, all_residuals, F_vals):
    """
    Compute asymptotic chi-sq test and finite-sample F version for time-series alphas.
    Returns dictionary of results and diagnostics.
    """
    alpha_vec = results_df['Alpha'].values.reshape(-1,1)
    all_residuals_matrix = np.vstack(all_residuals)
    Sigma = np.cov(all_residuals_matrix)
    Sigma_inv = np.linalg.inv(Sigma)

    T = F_vals.shape[0]
    N = alpha_vec.shape[0]
    K = F_vals.shape[1]

    if K == 1:
        f_bar_scalar = np.mean(F_vals)
        sigma_f_scalar = np.var(F_vals, ddof=1)
        scaling_term = 1 + (f_bar_scalar**2 / sigma_f_scalar)
    else:
        f_bar = np.mean(F_vals, axis=0).reshape(-1,1)
        Sigma_f = np.cov(F_vals.T)
        Sigma_f_inv = np.linalg.inv(Sigma_f)
        # .item() extracts the single value from a 1x1 matrix
        scaling_term = (1 + (f_bar.T @ Sigma_f_inv @ f_bar)).item()

    # .item() is also used here to prevent the DeprecationWarning
    alpha_part = (alpha_vec.T @ Sigma_inv @ alpha_vec).item()

    # Asymptotic chi-sq statistic
    chi2_stat = T * (1.0 / scaling_term) * alpha_part

    # Finite sample F-statistic
    f_stat = ((T - N - K) / N) * (1.0 / scaling_term) * alpha_part

    # critical values and p-values
    chi2_crit = chi2.ppf(0.95, df=N)
    p_chi2 = 1 - chi2.cdf(chi2_stat, df=N)

    f_crit = f.ppf(0.95, dfn=N, dfd=T - N - K)
    p_f = 1 - f.cdf(f_stat, dfn=N, dfd=T - N - K)

    # Diagnostics
    alpha_flat = results_df['Alpha'].values
    rms_alpha = np.sqrt(np.mean(alpha_flat**2))
    mean_abs_alpha = np.mean(np.abs(alpha_flat))

    return {
        'chi2_stat': chi2_stat,
        'chi2_crit_5pct': chi2_crit,
        'chi2_p_value': p_chi2,
        'f_stat': f_stat,
        'f_crit_5pct': f_crit,
        'f_p_value': p_f,
        'rms_alpha': rms_alpha,
        'mean_abs_alpha': mean_abs_alpha,
        'T': T, 'N': N, 'K': K
    }
# -------------------------
# 5) Cross-sectional regression WITH intercept
# -------------------------
def cross_section_with_constant(mean_returns, betas):
    """
    E(R^i) = lambda_0 + beta_i' * lambda + alpha_i
    betas: N x K matrix
    mean_returns: length N vector
    Returns fitted model and diagnostics including multiple R^2 measures.
    """
    y = mean_returns.values
    X = sm.add_constant(betas)  # N x (K+1)
    cs_model = sm.OLS(y, X).fit()

    lambda_0 = cs_model.params[0]
    lambdas = cs_model.params[1:]
    predicted = cs_model.fittedvalues
    pricing_errors = y - predicted

    rms_alpha = np.sqrt(np.mean(pricing_errors**2))
    mean_abs_alpha = np.mean(np.abs(pricing_errors))
    
    # Multiple R^2 measures as mentioned in problem
    var_y = np.var(y, ddof=0)  # using N denominator as suggested
    var_predicted = np.var(predicted, ddof=0)
    var_alpha = np.var(pricing_errors, ddof=0)
    mean_alpha_sq = np.mean(pricing_errors**2)
    
    r2_1 = var_predicted / var_y  # var(βλ)/var(R)
    r2_2 = 1 - var_alpha / var_y  # 1 - var(α)/var(R) 
    r2_3 = 1 - mean_alpha_sq / var_y  # 1 - E(α²)/var(R)

    return {
        'model': cs_model,
        'lambda_0': lambda_0,
        'lambdas': lambdas,
        'predicted_returns': predicted,
        'pricing_errors': pricing_errors,
        'rms_alpha': rms_alpha,
        'mean_abs_alpha': mean_abs_alpha,
        'r2_standard': cs_model.rsquared,
        'r2_var_pred': r2_1,
        'r2_var_alpha': r2_2,
        'r2_mean_alpha_sq': r2_3
    }

# -------------------------
# 6) Cross-sectional regression WITHOUT intercept (include factors as assets)
# -------------------------
def cross_section_no_constant(mean_returns, betas, F_vals, all_residuals):
    """
    Run cross-sectional regression without constant:
      E(R^i) = beta_i' * lambda + alpha_i
    and include the factors as test assets (so we append factor means and identity KxK).
    Returns lambda estimates, std errors, alpha SEs (iid formula), diagnostics, and Chi-sq test.
    """
    y = mean_returns.values  # length N
    K = betas.shape[1]
    N = betas.shape[0]
    T = F_vals.shape[0]

    # Append factor mean returns to y
    factor_means = F_vals.mean(axis=0)  # length K
    y_with_factors = np.concatenate([y, factor_means])  # length N+K

    # Append identity matrix for the factors' betas (each factor loads 1 on itself)
    X_with_factors = np.vstack([betas, np.eye(K)])  # shape (N+K, K)

    cs_model = sm.OLS(y_with_factors, X_with_factors).fit()
    estimated_lambdas = cs_model.params  # length K
    lambda_se = cs_model.bse

    # Compute alpha covariance using iid formulas
    all_resid_matrix = np.vstack(all_residuals)  # N x T
    Sigma = np.cov(all_resid_matrix)  # NxN

    # Projection matrix: I - B (B'B)^{-1} B'
    B = betas  # N x K
    Btb_inv = np.linalg.inv(B.T @ B)  # K x K
    proj = np.eye(N) - B @ Btb_inv @ B.T  # NxN

    # --- FIX STARTS HERE ---
    # Handle the K=1 case for the lambda_term calculation, as np.cov returns a scalar.
    if K == 1:
        # For a single factor, use scalar variance
        sigma_f_scalar = np.var(F_vals, ddof=1)
        # estimated_lambdas is a 1-element array in this case
        lambda_scalar = estimated_lambdas[0]
        # Avoid division by zero
        if sigma_f_scalar > 1e-9:
            lambda_term = lambda_scalar**2 / sigma_f_scalar
        else:
            lambda_term = 0.0
    else:
        # Original logic for K > 1, which works correctly
        Sigma_f = np.cov(F_vals.T)
        Sigma_f_inv = np.linalg.inv(Sigma_f)
        lambda_term = float(estimated_lambdas.T @ Sigma_f_inv @ estimated_lambdas)
    # --- FIX ENDS HERE ---

    cov_alpha = (1.0 / T) * proj @ Sigma @ proj * (1.0 + lambda_term)
    alpha_se = np.sqrt(np.maximum(0, np.diag(cov_alpha)))  # length N
    
    # Extract pricing errors for the test portfolios (first N elements)
    predicted = cs_model.fittedvalues
    pricing_errors = y_with_factors - predicted
    portfolio_pricing_errors = pricing_errors[:N]  # Only for the 25 portfolios
    
    # Chi-squared test for cross-sectional pricing errors
    # Add a small value to the diagonal to prevent singularity if cov_alpha is ill-conditioned
    cov_alpha_inv = np.linalg.inv(cov_alpha + np.eye(N) * 1e-10)
    chi2_cs_stat = float(portfolio_pricing_errors.T @ cov_alpha_inv @ portfolio_pricing_errors)
    
    # Degrees of freedom: N portfolios - K factor risk premia
    df_cs = N - K
    chi2_cs_crit = chi2.ppf(0.95, df=df_cs)
    p_chi2_cs = 1 - chi2.cdf(chi2_cs_stat, df=df_cs)

    rms_alpha = np.sqrt(np.mean(portfolio_pricing_errors**2))
    mean_abs_alpha = np.mean(np.abs(portfolio_pricing_errors))
    
    # Multiple R^2 measures (using only portfolio data, not factors)
    var_y_port = np.var(y, ddof=0)
    predicted_port = predicted[:N]  # Only portfolio predictions
    var_predicted_port = np.var(predicted_port, ddof=0)
    var_alpha_port = np.var(portfolio_pricing_errors, ddof=0)
    mean_alpha_sq_port = np.mean(portfolio_pricing_errors**2)
    
    r2_1 = var_predicted_port / var_y_port
    r2_2 = 1 - var_alpha_port / var_y_port
    r2_3 = 1 - mean_alpha_sq_port / var_y_port

    return {
        'model': cs_model,
        'estimated_lambdas': estimated_lambdas,
        'lambda_se': lambda_se,
        'alpha_se': alpha_se,
        'predicted_returns': predicted,
        'pricing_errors': pricing_errors,
        'portfolio_pricing_errors': portfolio_pricing_errors,
        'rms_alpha': rms_alpha,
        'mean_abs_alpha': mean_abs_alpha,
        'r2_standard': cs_model.rsquared,
        'r2_var_pred': r2_1,
        'r2_var_alpha': r2_2, 
        'r2_mean_alpha_sq': r2_3,
        'chi2_cs_stat': chi2_cs_stat,
        'chi2_cs_crit': chi2_cs_crit,
        'chi2_cs_p_value': p_chi2_cs,
        'chi2_cs_df': df_cs
    }

# -------------------------
# 7) Enhanced pretty-printing utilities
# -------------------------
def print_header(title):
    """Print a nice header"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"{title:^80}")
    print(line)

def print_subheader(title):
    """Print a subheader"""
    line = "-" * 60
    print(f"\n{line}")
    print(f" {title}")
    print(line)

def print_matrix(matrix, title, fmt=".4f"):
    """Print a 5x5 matrix nicely formatted"""
    print(f"\n{title}:")
    for i, row in enumerate(matrix):
        row_str = " ".join([f"{x:{fmt}}" for x in row])
        print(f"  {row_str}")

def print_factor_estimates(F_vals, factor_names):
    """Print factor mean estimates and standard errors"""
    print("\nFactor Estimates (λ̂ = f̄):")
    factor_means = np.mean(F_vals, axis=0)
    factor_se = np.std(F_vals, axis=0) / np.sqrt(F_vals.shape[0])
    
    for i, name in enumerate(factor_names):
        t_stat = factor_means[i] / factor_se[i]
        print(f"  {name:>8}: {factor_means[i]:8.4f} ({factor_se[i]:6.4f}) t = {t_stat:6.2f}")

def print_test_results(results_dict, title):
    """Print test results in a nice format"""
    print(f"\n{title}:")
    print(f"  Test statistic:      {results_dict['chi2_stat']:8.4f}")
    print(f"  Critical value (5%): {results_dict['chi2_crit_5pct']:8.4f}")
    print(f"  p-value:            {results_dict['chi2_p_value']:8.4f}")
    if results_dict['chi2_p_value'] < 0.05:
        print("  Result: REJECT H₀ (model is rejected)")
    else:
        print("  Result: FAIL TO REJECT H₀ (model not rejected)")

def print_cross_sectional_results(cs_results, title, factor_names):
    """Print cross-sectional regression results"""
    print_subheader(title)
    
    if 'lambda_0' in cs_results:  # With constant
        print(f"λ₀ (constant):    {cs_results['lambda_0']:8.4f}")
    
    if 'lambda_se' in cs_results:  # Has standard errors
        print("λ̂ (factor risk premia):")
        for i, name in enumerate(factor_names):
            t_stat = cs_results['estimated_lambdas'][i] / cs_results['lambda_se'][i]
            print(f"  λ ({name:>6}):   {cs_results['estimated_lambdas'][i]:8.4f} ({cs_results['lambda_se'][i]:6.4f}) t = {t_stat:6.2f}")
    else:  # No standard errors computed
        print("λ̂ (factor risk premia):")
        for i, name in enumerate(factor_names):
            print(f"  λ ({name:>6}):   {cs_results['lambdas'][i]:8.4f}")
    
    # Chi-squared test (if available)
    if 'chi2_cs_stat' in cs_results:
        print("\nChi-squared test (pricing errors):")
        print(f"  Test statistic:      {cs_results['chi2_cs_stat']:8.4f}")
        print(f"  Critical value (5%): {cs_results['chi2_cs_crit']:8.4f}")
        print(f"  p-value:            {cs_results['chi2_cs_p_value']:8.4f}")
        if cs_results['chi2_cs_p_value'] < 0.05:
            print("  Result: REJECT H₀ (model is rejected)")
        else:
            print("  Result: FAIL TO REJECT H₀ (model not rejected)")
    
    print(f"RMS pricing error:   {cs_results['rms_alpha']:8.4f}")
    print(f"Mean |pricing error|: {cs_results['mean_abs_alpha']:8.4f}")
    print(f"R² (var(β̂λ̂)/var(R)):  {cs_results['r2_var_pred']:8.4f}")
    print(f"R² (1 - var(α̂)/var(R)): {cs_results['r2_var_alpha']:8.4f}")
    print(f"R² (1 - E(α̂²)/var(R)):  {cs_results['r2_mean_alpha_sq']:8.4f}")

# -------------------------
# 8) User input functions
# -------------------------
def get_user_inputs():
    """Get analysis parameters from user"""
    print("=" * 60)
    print("FAMA-FRENCH ANALYSIS CONFIGURATION")
    print("=" * 60)
    
    # Get date range
    print("\nDATE RANGE SELECTION:")
    print("Enter dates in YYYY-MM-DD format (e.g., 1960-01-01)")
    
    # Get start date
    while True:
        try:
            start_date = input("Start date (default: 1932-01-01): ").strip()
            if not start_date:
                start_date = '1932-01-01'
            pd.to_datetime(start_date)
            break
        except:
            print("Invalid date format. Please use YYYY-MM-DD.")
            
    # --- NEW CODE BLOCK ---
    # Get end date (optional)
    print("Leave End Date blank to use all data until the end of the file.")
    while True:
        try:
            end_date = input("End date (default: last available): ").strip()
            if not end_date:
                end_date = None  # Use None to signify no end date filter
                break
            pd.to_datetime(end_date)
            break
        except:
            print("Invalid date format. Please use YYYY-MM-DD.")
    # --- END NEW CODE BLOCK ---
            
    # Get number of factors
    print("\nFACTOR MODEL SELECTION:")
    print("1 - CAPM (Market factor only)")
    print("3 - Fama-French 3-factor (Market + SMB + HML)")
    
    while True:
        try:
            num_factors = int(input("Number of factors (1 or 3): ").strip())
            if num_factors in [1, 3]:
                break
            else:
                print("Please enter 1 or 3.")
        except ValueError:
            print("Please enter a valid number (1 or 3).")
    
    return start_date, end_date, num_factors

# -------------------------
# 9) Run the enhanced pipeline
# -------------------------
def run_analysis(start_date='1932-01-01', end_date=None, num_factors=3, interactive=True):
    """
    Run the complete Fama-French analysis
    
    Parameters:
    -----------
    start_date : str
        Start date for analysis (format: 'YYYY-MM-DD')
    end_date : str or None
        End date for analysis (format: 'YYYY-MM-DD'). If None, uses all data to the end.
    ...
    """
    
    if interactive:
        # Now returns 3 values
        start_date, end_date, num_factors = get_user_inputs()
    
    model_name = "CAPM" if num_factors == 1 else "Fama-French 3-Factor"
    print_header(f"{model_name} ASSET PRICING ANALYSIS")
    
    # 1) Load data
    print("\n📊 Loading and cleaning data...")
    try:
        portfolio_returns, F_t = load_and_clean(
            start_date=start_date,
            end_date=end_date,  # <-- PASS END_DATE HERE
            num_factors=num_factors
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please ensure the following files exist:")
        print("  - data/25_Portfolios_5x5.csv")
        print("  - data/F-F_Research_Data_Factors.csv")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Update print statements for clarity
    date_range_str = f"{portfolio_returns.index[0].strftime('%Y-%m')} to {portfolio_returns.index[-1].strftime('%Y-%m')}"
    
    print(f"    Start date requested: {start_date}")
    if end_date:
        print(f"    End date requested:   {end_date}")
    else:
        print(f"    End date requested:   None (using last available date)")
        
    print(f"    Data loaded: {len(portfolio_returns)} observations from {date_range_str}")
    print(f"    Portfolio returns: {portfolio_returns.shape[1]} assets")
    print(f"    Factors: {F_t.shape[1]} factors ({', '.join(F_t.columns)})")
    print(f"    Model: {model_name}")

    # (a) Mean returns table
    print_subheader("MEAN RETURNS")
    mean_returns, mean_matrix = mean_returns_table(portfolio_returns)
    print_matrix(mean_matrix, "Monthly Mean Excess Returns (5x5 Size-BM sorted portfolios)", ".4f")

    # (b,c) Time-series regressions
    print_subheader("TIME-SERIES REGRESSIONS")
    results_df, all_residuals, F_vals = time_series_regressions(portfolio_returns, F_t)
    
    # Print some key results matrices
    alpha_matrix = results_df['Alpha'].values.reshape(5,5)
    print_matrix(alpha_matrix, "Alphas (α̂ᵢ)", ".4f")
    
    alpha_se_matrix = results_df['Alpha SE'].values.reshape(5,5)
    print_matrix(alpha_se_matrix, "Alpha Standard Errors", ".4f")
    
    # Print betas for first factor (market)
    beta1_matrix = np.array([beta[0] for beta in results_df['Betas']]).reshape(5,5)
    print_matrix(beta1_matrix, f"Betas - {F_t.columns[0]}", ".4f")
    
    # Print additional beta matrices for 3-factor model
    if num_factors == 3:
        beta2_matrix = np.array([beta[1] for beta in results_df['Betas']]).reshape(5,5)
        print_matrix(beta2_matrix, f"Betas - {F_t.columns[1]}", ".4f")
        
        beta3_matrix = np.array([beta[2] for beta in results_df['Betas']]).reshape(5,5)
        print_matrix(beta3_matrix, f"Betas - {F_t.columns[2]}", ".4f")
    
    r2_matrix = results_df['R_squared'].values.reshape(5,5)
    print_matrix(r2_matrix, "R² values", ".4f")

    # Factor estimates
    factor_names = F_t.columns.tolist()
    print_factor_estimates(F_vals, factor_names)

    # (d) Chi-sq and F test
    print_subheader("TIME-SERIES JOINT TEST")
    chi2_results = chi2_test(results_df, all_residuals, F_vals)
    print_test_results(chi2_results, "Chi-squared test (H₀: all αᵢ = 0)")
    print(f"RMS α:              {chi2_results['rms_alpha']:8.4f}")
    print(f"Mean |α|:           {chi2_results['mean_abs_alpha']:8.4f}")
    
    print(f"\nFinite-sample F-test:")
    print(f"  F-statistic:        {chi2_results['f_stat']:8.4f}")
    print(f"  Critical value (5%): {chi2_results['f_crit_5pct']:8.4f}")
    print(f"  p-value:           {chi2_results['f_p_value']:8.4f}")

    print_header("CROSS-SECTIONAL REGRESSIONS")

    # (e) Cross-sectional regressions
    betas_matrix = np.array(results_df['Betas'].tolist())  # N x K

    # With constant
    cs_with_const = cross_section_with_constant(mean_returns, betas_matrix)
    print_cross_sectional_results(cs_with_const, "With constant: E(Rᵢ) = λ₀ + βᵢλ + αᵢ", factor_names)

    # Without constant (include factors as assets)
    cs_no_const = cross_section_no_constant(mean_returns, betas_matrix, F_vals, all_residuals)
    print_cross_sectional_results(cs_no_const, "Without constant (including factors): E(Rᵢ) = βᵢλ + αᵢ", factor_names)

    print_header("SUMMARY COMPARISON")
    print("\nFactor Risk Premia Estimates:")
    print("                    Time-Series    Cross-Sect(w/const)  Cross-Sect(no const)")
    factor_means = np.mean(F_vals, axis=0)
    for i, name in enumerate(factor_names):
        ts_est = factor_means[i]
        cs_const_est = cs_with_const['lambdas'][i] 
        cs_no_const_est = cs_no_const['estimated_lambdas'][i]
        print(f"  λ ({name:>6}):    {ts_est:8.4f}        {cs_const_est:8.4f}           {cs_no_const_est:8.4f}")

    print(f"\nModel Performance Summary:")
    print(f"                       Time-Series  Cross-Sect(const)  Cross-Sect(no const)")
    print(f"  RMS α:               {chi2_results['rms_alpha']:8.4f}      {cs_with_const['rms_alpha']:8.4f}         {cs_no_const['rms_alpha']:8.4f}")
    print(f"  Mean |α|:            {chi2_results['mean_abs_alpha']:8.4f}      {cs_with_const['mean_abs_alpha']:8.4f}         {cs_no_const['mean_abs_alpha']:8.4f}")
    print(f"  R² (main):           {'N/A':>8s}      {cs_with_const['r2_var_alpha']:8.4f}         {cs_no_const['r2_var_alpha']:8.4f}")
    
    print(f"\nHypothesis Tests:")
    print(f"  Time-series (χ²):    p = {chi2_results['chi2_p_value']:6.4f}")
    if 'chi2_cs_p_value' in cs_no_const:
        print(f"  Cross-sect (χ²):     p = {cs_no_const['chi2_cs_p_value']:6.4f}")

    print("\n" + "="*80)
    print(f"Analysis completed successfully using {model_name}!")
    print(f"Sample period: {date_range_str}")
    print("="*80)
    
    return {
        'portfolio_returns': portfolio_returns,
        'F_t': F_t,
        'results_df': results_df,
        'chi2_results': chi2_results,
        'cs_with_const': cs_with_const,
        'cs_no_const': cs_no_const,
        'analysis_params': {
            'start_date': start_date,
            'num_factors': num_factors,
            'model_name': model_name,
            'date_range': date_range_str
        }
    }

# -------------------------
# 10) Example usage and main execution
# -------------------------
if __name__ == "__main__":
    # Example 1: Interactive mode (default)
    # results = run_analysis(interactive=True)
    
    # Example 2: Non-interactive mode with custom parameters
    # CAPM analysis from 1960 onwards (end date determined by data)
    # results = run_analysis(start_date='1960-01-01', num_factors=1, interactive=False)
    
    # Example 3: 3-factor analysis from 1990 onwards
    # results = run_analysis(start_date='1990-01-01', num_factors=3, interactive=False)
    
    # Run in interactive mode by default
    results = run_analysis(interactive=True)