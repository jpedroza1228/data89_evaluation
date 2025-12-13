// 2PL IRT Model 2: Two-Parameter Logistic Model with Hierarchical Priors
// This model uses hierarchical priors for item parameters

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
}

parameters {
  vector[N] theta;                   // student ability parameters
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
  real mu_b;                         // mean of difficulty parameters
  real<lower=0> sigma_b;             // sd of difficulty parameters
  real<lower=0> sigma_a;             // sd of discrimination parameters
}

model {
  // Hyperpriors
  mu_b ~ normal(0, 1);
  sigma_b ~ cauchy(0, 2.5);
  sigma_a ~ cauchy(0, 1);
  
  // Priors
  theta ~ normal(0, 1);
  a ~ lognormal(0, sigma_a);
  b ~ normal(mu_b, sigma_b);
  
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
