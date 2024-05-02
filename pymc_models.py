'''
    Defines a set of pymc models for Bayesian regularisation
    implented in the Stan programming language at 
    https://github.com/sara-vanerp/bayesreg/tree/master/src/stan_files
'''
import arviz as az
import pymc as pm

# Model for Bayesian elastic net.
with pm.Model() as lm_noNA_elasticNet_FB:
    # Define priors
    # Implciit uniform prior imlmented in stan model in which this is based
    # So uniform with large bounds used here 
    mu = pm.Uniform("mu", lower=10e6, upper=10e6)

    beta_raw = pm.Normal("beta_raw", mu = 0, sigma = 1, shape = p)

    sigma2 = pm.Exponential("sigma2", lam=1)  # equivalent to a uniform on log(sigma^2)

    lambda1 = pm.HalfCauchy("lambda1",beta=1)
    lambda2 = pm.HalfCauchy("lambda2",beta=1)
    tau = pm.Gamma("tau", alpha = 0.5, beta = (8*lambda1*sigma2)/(lambda2**2)  )

    # Equivalent to transformed parameters section.
    beta = pm.Deterministic("beta", pm.math.sqrt((sigma2 * (tau - 1)) / (lambda2 * tau)) * beta_raw)
    sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma2))
    # Define linear predictor.
    linpred = pm.Deterministic("linpred", mu + pm.math.dot(X_train, beta) )
    
    # Define likelihood
    y_train = Normal("y_train", mu=linpred, sigma=sigma, observed=y_train)


with pm.Model as lm_noNA_ridge_FB:
    # Define priors
    # Implciit uniform prior implmented in stan model in which this is based
    # So uniform with large bounds used here 
    mu = pm.Uniform("mu", lower=10e6, upper=10e6)
    sigma2 = pm.Exponential("sigma2", lam=1)  # equivalent to a uniform on log(sigma^2)
    lambda1 = pm.HalfCauchy("lambda",  beta = 1)
    tau2 = pm.Deterministic("tau2", sigma2/lambda1)
    beta = pm.Normal("beta", mu = 0, sigma = pm.math.sqrt(tau2), shape = p)

    # Equivalent to transformed parameters section.
    sigma = pm.Deterministic("sigma", pm.math.sqrt(sigma2))
    # Define linear predictor.
    linpred = pm.Deterministic("linpred", mu + pm.math.dot(X_train, beta) )

    # Define likelihood
    y_train = Normal("y_train", mu=linpred, sigma=sigma, observed=y_train)