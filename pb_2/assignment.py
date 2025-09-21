import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import chi2, f

'''Eliminate the data prior to 193201 (i.e. 67:end) to avoid the missing data in the earliest years of the sample. (Make sure you don’t use any -99 data). Subtract the risk free rate rf from the 25 test assets to make them excess returns. The factors are already excess returns.
'''

# Load only the value-weighted monthly returns (first section)
portfolio_SBM = pd.read_csv('data/25_Portfolios_5x5.csv', skiprows=15, nrows=1188)
fama_french_factors = pd.read_csv('data/F-F_Research_Data_Factors.csv', skiprows=3, nrows=1188)

# Clean data
portfolio_SBM = portfolio_SBM.dropna(how='all')
portfolio_SBM['date'] = pd.to_datetime(portfolio_SBM.iloc[:,0], format = '%Y%m')
portfolio_SBM.set_index('date', inplace= True)
fama_french_factors['date'] = pd.to_datetime(fama_french_factors.iloc[:,0], format='%Y%m')
fama_french_factors.set_index('date', inplace=True)

# Eliminate data prior 193201
portfolio_SBM = portfolio_SBM[portfolio_SBM.index >= '1932-01-01']
fama_french_factors = fama_french_factors[fama_french_factors.index >= '1932-01-01']
# Calculate only excess returns for portfolios
portfolio_SBM = portfolio_SBM.subtract(fama_french_factors['RF'], axis=0)

## Question (a)
'''
Report mean returns like Fama and French (1993, Journal of Financial Economics) Table 1. Report all 25 long vectors in 5×5 matrices in small and value dimensions like Fama and French.
'''

print("Portfolio columns:", portfolio_SBM.columns.tolist())
print("Number of portfolio columns:", len(portfolio_SBM.columns))

portfolio_returns = portfolio_SBM.iloc[:, 1:] # Special note: remove first column as its now the index
mean_returns = portfolio_returns.mean()
mean_returns_matrix = mean_returns.values.reshape(5,5)
print("Mean returns matrix (5x5):")
print(mean_returns_matrix)

## Question (b)
'''Run OLS time-series regressions to find parameter estimates of αˆ,βˆ.'''
''' Given that we have 25 portfolios, we need to run 25 regressions, one for each portfolio.
Each regressio will use the excess returns of the portfolio as the dependent variable and then the three
Fama French factors (MKT, SMB, HML) as independent variables. See one note file for more explanations'''

## Question (c)
'''Find standard errors of αˆ,βˆ,λˆ using classic iid formulas.
Apparently for αˆ,βˆ are measured by OLS by using the residuals of the regression as a proxy of
the amount of noise in the regression and thus the standard errors.
That being said, those can be retrieved from the linear regression model'''

# Write the code here to retrieve the standard errors of alpha and betas from the regression model
# Note: sklearn's LinearRegression does not provide standard errors directly. You may need to use statsmodels for that.

F_t = fama_french_factors[['Mkt-RF', 'SMB', 'HML']].values # Prepare independent variables (Fama French factors)

results = [] # Initialize list to store results
alpha_se = []
beta_se = []
all_residuals = []

for portfolio in portfolio_returns.columns:
    y = portfolio_returns[portfolio].values
    X = sm.add_constant(F_t) # Add constant term for intercept
    model = sm.OLS(y, X).fit()
    alpha = model.params[0] # Intercept (alpha)
    betas = model.params[1:] # Coefficients (betas)
    results.append({'Portfolio': portfolio, 'Alpha': alpha, 'Betas': betas})
    alpha_se.append(model.bse[0]) # Standard error of alpha
    beta_se.append(model.bse[1:]) # Standard errors of betas
    all_residuals.append(model.resid) # Store residuals

results_df = pd.DataFrame(results) # Convert results to DataFrame
risk_premia_predicted = F_t.mean(axis=0) # Lets estimate the forecasted risk premia as the average of the factors
print(risk_premia_predicted)

results_df['Alpha SE'] = alpha_se # Add standard errors to results DataFrame
for i in range(3):
    results_df[f'Beta {i+1} SE'] = [beta[i] for beta in beta_se]

# Lets caculate the standard error of the risk premia
risk_premia_se = np.std(F_t, axis=0) / np.sqrt(F_t.shape[0])
print(results_df)
print(risk_premia_se)

## Question (d)
'''Compute the asymptotic Chi-squared test. Compute the test statistic, the 5% critical value,
and the probability value (chance of seeing a statistic as high or higher than observed in sample.
The cdf function will come in useful here). As a diagnostic, also present the root mean square and average absolute alphas.'''

alpha_vector = results_df['Alpha'].values # Vector of alphas 25 portfolios
alpha_matrix = alpha_vector.reshape(-1,1) # Column vector of alphas

all_residuals_matrix = np.array(all_residuals) 
cov_matrix = np.cov(all_residuals_matrix) # Covariance matrix of residuals
inv_cov_matrix = np.linalg.inv(cov_matrix)

T = F_t.shape[0]  # Number of time periods
N = alpha_matrix.shape[0]  # Number of test assets
K = F_t.shape[1]  # Number of factors

f_bar_matrix = np.mean(F_t, axis=0).reshape(-1, 1) # Mean of factors

cov_matrix_factors = np.cov(F_t.T) # Covariance matrix of factors
inv_cov_matrix_factors = np.linalg.inv(cov_matrix_factors)

# Scaling term: [1 + f_bar' * Sigma_f_inv * f_bar]
scaling_term = 1 + f_bar_matrix.T @ inv_cov_matrix_factors @ f_bar_matrix

# Alpha quadratic form: alpha' * Sigma_inv * alpha
alpha_part = alpha_matrix.T @ inv_cov_matrix @ alpha_matrix

# Asymptotic Chi-squared test statistic
chi2_statistic = T * (1 / scaling_term) * alpha_part
chi2_statistic = float(chi2_statistic)

# Finite-sample F-statistic
f_test_statistic = ((T - N - K) / N) * (1 / scaling_term) * alpha_part
f_test_statistic = float(f_test_statistic)

# Critical value (5% level)
chi2_crit = chi2.ppf(0.95, df=N)
f_crit = f.ppf(0.95, dfn=N, dfd=T - N - K)

# p-value
p_value_chi2 = 1 - chi2.cdf(chi2_statistic, df=N)
p_value_f = 1 - f.cdf(f_test_statistic, dfn=N, dfd=T - N - K) # F-test p-value

# Diagnostic 
rms_alpha = np.sqrt(np.mean(alpha_vector**2))
avg_abs_alpha = np.mean(np.abs(alpha_vector))

print("Chi-squared statistic:", chi2_statistic)
print("Chi-squared 5% critical value:", chi2_crit)
print("Chi-squared p-value:", p_value_chi2)
print("\nFinite-sample F-statistic:", f_test_statistic)
print("F-statistic 5% critical value:", f_crit)
print("F-statistic p-value:", p_value_f)
print("\nRMS alpha:", rms_alpha)
print("Average absolute alpha:", avg_abs_alpha)

## Question (e)
# Part 1.
''' Now, run OLS cross-sectional regressions.Run the cross sectional regression with a constant, 
E (Rei) = λ0 + βiλ + αi. There are standard error etc. formulas for this case, but I’ll spare you programming them up. 
Report i) Estimates λˆ0,λˆ, ii) Root mean square and mean absolute pricing errors αˆ, and the R2 of actual vs. predicted mean returns.'''

alphas = results_df['Alpha'].values # Vector of alphas 25 portfolios
X = np.array(results_df['Betas'].tolist()) # Matrix of betas 25x3
y = mean_returns.values # Vector of mean returns 25 portfolios
X = sm.add_constant(X) # Add constant term for intercept
cs_model = sm.OLS(y, X).fit()
lambda_0 = cs_model.params[0] # Intercept (alpha)
lambda_1 = cs_model.params[1:] # Coefficients (betas)

# Predicted mean returns
predicted_returns = cs_model.fittedvalues

# Pricing errors (alphas_hat)
pricing_errors = y - predicted_returns

# Diagnostics
rms_alpha = np.sqrt(np.mean(pricing_errors**2))
mean_abs_alpha = np.mean(np.abs(pricing_errors))
r2 = cs_model.rsquared

print("λ0 (intercept):", lambda_0)
print("λ (risk premia):", lambda_1)
print("RMS pricing error:", rms_alpha)
print("Mean absolute pricing error:", mean_abs_alpha)
print("R²:", r2)

## Part 2
y = mean_returns.values # Vector of mean returns 25 portfolios
factor_means = F_t.mean(axis=0)  # Means of Mkt-RF, SMB, HML
y_with_factors = np.append(y, factor_means) # Now y_with_factors is 28x1
identity_matrix = np.eye(3)
X_with_id = np.vstack([X, identity_matrix]) # Now X_with_id is 28x3

cs_model_v2 = sm.OLS(y_with_factors, X_with_id).fit()
lambda_0_v2 = cs_model_v2.params[0] # Intercept (alpha)
lambda_1_v2 = cs_model_v2.params[1:] # Coefficients (betas)

# Lets compute the standard errors of lambda and alpha with traditional iid formualas
lambda_se_v2 = cs_model_v2.bse # Standard errors of lambda