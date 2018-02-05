data {
  int<lower=0> N; // Number of different DFT calculations
  int<lower=0> L; // Number of features
  matrix[N, L] X; // Design matrix, N DFT calculations by L features (or correlations)
  vector[N] y; // The outputs of the DFT
}

parameters {
  vector[L] w; // ECIs
  real b; // Intercept of regression, is this is 0-pair ECI?
  real<lower = 0.0> sigma; // Standard deviation of fit
}

model {
  w ~ normal(0, 1); // Prior on ECIs -- could be a lot of different things
  y ~ normal(X * w + b, sigma);
  // This could have also been written element-wise:
  // for (i in 1:N) {
  //   y[i] ~ normal(X[i] * w + b, sigma);
  // }
}

// This is a posterior predictive. It's a standard way of figuring
// out if your fit is good by generating data under the computed posterior
// If the fit is good, it should look like your actual data, y
generated quantities {
  vector[N] yhat;

  for (i in 1:N)
    yhat[i] = normal_rng(X[i] * w + b, sigma);
}