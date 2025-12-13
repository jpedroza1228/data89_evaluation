// 2PL IRT Model 6: Two-Parameter Logistic Model with Multidimensional Abilities
// This model extends to multiple latent ability dimensions

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
  int<lower=1> K;                    // number of ability dimensions
  matrix[J, K] Q;                    // Q-matrix: items to dimensions mapping
}

parameters {
  matrix[N, K] theta;                // student ability parameters (multidimensional)
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
}

model {
  // Priors
  for (n in 1:N) {
    for (k in 1:K) {
      theta[n, k] ~ normal(0, 1);
    }
  }
  a ~ lognormal(0, 1);
  b ~ normal(0, 2);
  
  // Likelihood
  for (n in 1:N) {
    for (j in 1:J) {
      real ability_component = 0;
      for (k in 1:K) {
        ability_component += Q[j, k] * theta[n, k];
      }
      Y[n, j] ~ bernoulli_logit(a[j] * (ability_component - b[j]));
    }
  }
}

generated quantities {
  matrix[N, J] prob_correct;
  vector[N] total_score;
  
  for (n in 1:N) {
    total_score[n] = 0;
    for (j in 1:J) {
      real ability_component = 0;
      for (k in 1:K) {
        ability_component += Q[j, k] * theta[n, k];
      }
      prob_correct[n, j] = inv_logit(a[j] * (ability_component - b[j]));
      total_score[n] += prob_correct[n, j];
    }
  }
}
