// GDINA Model 2: General Diagnostic Classification Model with Hierarchical Priors
// This model includes hierarchical structure for attribute patterns

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=1> K;                    // number of attributes
  int<lower=0,upper=1> Y[N, J];      // response matrix
  int<lower=1> Q[J, K];              // Q-matrix (item-attribute relationships)
  int<lower=1> max_pattern;          // maximum attribute pattern (2^K)
}

parameters {
  simplex[max_pattern] alpha_prob;   // probability of each attribute pattern
  real<lower=0,upper=1> delta[J, max_pattern];  // item parameters for each pattern
  real<lower=0> sigma_delta;         // hierarchical variance parameter
}

transformed parameters {
  matrix[N, max_pattern] log_lik_pattern;
  
  for (n in 1:N) {
    for (p in 1:max_pattern) {
      log_lik_pattern[n, p] = 0;
      for (j in 1:J) {
        if (Y[n, j] == 1) {
          log_lik_pattern[n, p] += log(delta[j, p]);
        } else {
          log_lik_pattern[n, p] += log(1 - delta[j, p]);
        }
      }
    }
  }
}

model {
  // Priors
  alpha_prob ~ dirichlet(rep_vector(1.0, max_pattern));
  sigma_delta ~ cauchy(0, 2.5);
  
  for (j in 1:J) {
    for (p in 1:max_pattern) {
      delta[j, p] ~ beta(2, 2);
    }
  }
  
  // Likelihood
  for (n in 1:N) {
    vector[max_pattern] lp = log(alpha_prob);
    for (p in 1:max_pattern) {
      lp[p] += log_lik_pattern[n, p];
    }
    target += log_sum_exp(lp);
  }
}

generated quantities {
  matrix[N, max_pattern] pattern_prob;
  int<lower=1,upper=max_pattern> alpha_class[N];
  
  for (n in 1:N) {
    vector[max_pattern] lp = log(alpha_prob);
    for (p in 1:max_pattern) {
      lp[p] += log_lik_pattern[n, p];
    }
    pattern_prob[n, ] = to_row_vector(softmax(lp));
    alpha_class[n] = categorical_rng(softmax(lp));
  }
}
