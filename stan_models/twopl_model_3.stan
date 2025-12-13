// 2PL IRT Model 3: Two-Parameter Logistic Model with Student Covariates
// This model incorporates student-level covariates into ability estimation

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
  int<lower=0> P;                    // number of student covariates
  matrix[N, P] X;                    // student covariate matrix
}

parameters {
  vector[N] theta_raw;               // raw student ability parameters
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
  vector[P] beta;                    // covariate effects on ability
  real<lower=0> sigma_theta;         // residual sd of ability
}

transformed parameters {
  vector[N] theta;
  
  for (n in 1:N) {
    if (P > 0) {
      theta[n] = dot_product(X[n, ], beta) + sigma_theta * theta_raw[n];
    } else {
      theta[n] = sigma_theta * theta_raw[n];
    }
  }
}

model {
  // Priors
  theta_raw ~ normal(0, 1);
  sigma_theta ~ cauchy(0, 1);
  beta ~ normal(0, 1);
  a ~ lognormal(0, 1);
  b ~ normal(0, 2);
  
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
