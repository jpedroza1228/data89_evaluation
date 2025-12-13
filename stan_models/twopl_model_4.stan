// 2PL IRT Model 4: Two-Parameter Logistic Model with Item Groups
// This model allows for item grouping (e.g., by content area)

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
  int<lower=1> G;                    // number of item groups
  int<lower=1,upper=G> group[J];     // group assignment for each item
}

parameters {
  vector[N] theta;                   // student ability parameters
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
  vector[G] mu_b_group;              // mean difficulty per group
  real<lower=0> sigma_b_within;      // within-group sd for difficulty
}

model {
  // Priors
  theta ~ normal(0, 1);
  a ~ lognormal(0, 1);
  mu_b_group ~ normal(0, 1);
  sigma_b_within ~ cauchy(0, 1);
  
  for (j in 1:J) {
    b[j] ~ normal(mu_b_group[group[j]], sigma_b_within);
  }
  
  // Likelihood
  for (n in 1:N) {
    for (j in 1:J) {
      Y[n, j] ~ bernoulli_logit(a[j] * (theta[n] - b[j]));
    }
  }
}

generated quantities {
  matrix[N, J] prob_correct;
  vector[N] total_score;
  
  for (n in 1:N) {
    total_score[n] = 0;
    for (j in 1:J) {
      prob_correct[n, j] = inv_logit(a[j] * (theta[n] - b[j]));
      total_score[n] += prob_correct[n, j];
    }
  }
}
