import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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

results = []  # Initialize list to store results
alpha_se = []
beta_se = []

for portfolio in portfolio_returns.columns:
    y = portfolio_returns[portfolio].values
    X = sm.add_constant(F_t)  # Add constant term for intercept
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]  # Intercept (alpha)
    betas = model.params[1:]  # Coefficients (betas)
    results.append({'Portfolio': portfolio, 'Alpha': alpha, 'Betas': betas.values})
    alpha_se.append(model.bse[0])  # Standard error of alpha
    beta_se.append(model.bse[1:].values)   # Standard errors of betas

results_df = pd.DataFrame(results) # Convert results to DataFrame
risk_premia_predicted = F_t.mean(axis=0) # Lets estimate the forecasted risk premia as the average of the factors
print(risk_premia_predicted)

results_df['Alpha SE'] = alpha_se # Add standard errors to results DataFrame
for i in range(3):
    results_df[f'Beta {i+1} SE'] = [beta[i] for beta in beta_se]

print(results_df)