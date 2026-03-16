data {
  int<lower=1> J;                // Number of persons
  int<lower=1> I;                // Number of items
  int<lower=1> D;                // Number of dimensions
  array[I] int<lower=1, upper=D> dim;  // Which dimension each item belongs to
  matrix<lower=0, upper=1> [J,I] Y;   // Response matrix
}
parameters {
  array[J] vector[D] theta;            // Latent abilities (multivariate)
  vector<lower=0>[I] alpha;      // Discriminations (constrained to positive)
  vector[I] beta;                // Difficulties
  cholesky_factor_corr[D] L_Omega; // Correlation between dimensions
}
model {
  array[J] real log_lik;

  // Priors
  L_Omega ~ lkj_corr_cholesky(2); 
  alpha ~ lognormal(0, 0.5);
  beta ~ normal(0, 2);
  // Latent traits drawn from Multivariate Normal(0, Omega)
  // We use Cholesky decomposition for efficiency
  for (j in 1:J) {
    theta[j] ~ multi_normal_cholesky(rep_vector(0, D), L_Omega);
  }

  // Likelihood
  for (j in 1:J) {
    for (i in 1:I) {
      real linear_pred = inv_logit(alpha[i] * theta[j, dim[i]] - beta[i]);
      log_lik[j] += Y[j, i] * log(linear_pred) + (1 - Y[j, i]) * log1m(linear_pred);
    }
  }
}
generated quantities {
  matrix[D, D] Omega;
  matrix<lower=0, upper=1> [J, I] y_rep; // Replicated data
  
  // Recover the correlation matrix
  Omega = multiply_lower_tri_self_transpose(L_Omega); 

  // Generate replicated data point-by-point
  for (j in 1:J) {
    for (i in 1:I) {
      real eta = alpha[i] * theta[j, dim[i]] - beta[i];
      y_rep[j, i] = bernoulli_logit_rng(eta);
    }
  }
}