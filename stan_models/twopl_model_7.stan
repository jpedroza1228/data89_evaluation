// 2PL IRT Model 7: Two-Parameter Logistic Model with Testlet Effects
// This model accounts for local item dependence within testlets

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
  int<lower=1> T;                    // number of testlets
  int<lower=1,upper=T> testlet[J];   // testlet assignment for each item
}

parameters {
  vector[N] theta;                   // student ability parameters
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
  matrix[N, T] gamma;                // testlet effects
  real<lower=0> sigma_testlet;       // sd of testlet effects
}

model {
  // Priors
  theta ~ normal(0, 1);
  a ~ lognormal(0, 1);
  b ~ normal(0, 2);
  sigma_testlet ~ cauchy(0, 1);
  
  for (n in 1:N) {
    for (t in 1:T) {
      gamma[n, t] ~ normal(0, sigma_testlet);
    }
  }
  
  // Likelihood
  for (n in 1:N) {
    for (j in 1:J) {
      Y[n, j] ~ bernoulli_logit(a[j] * (theta[n] + gamma[n, testlet[j]] - b[j]));
    }
  }
}

generated quantities {
  matrix[N, J] prob_correct;
  vector[N] total_score;
  
  for (n in 1:N) {
    total_score[n] = 0;
    for (j in 1:J) {
      prob_correct[n, j] = inv_logit(a[j] * (theta[n] + gamma[n, testlet[j]] - b[j]));
      total_score[n] += prob_correct[n, j];
    }
  }
}
