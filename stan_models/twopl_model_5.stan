// 2PL IRT Model 5: Two-Parameter Logistic Model with Temporal/Longitudinal Effects
// This model accounts for growth/learning over time

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
  vector[N] time_point;              // time/assessment number for each student
}

parameters {
  vector[N] theta_baseline;          // baseline student ability
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
  real<lower=0> growth_rate;         // average growth rate over time
  real<lower=0> sigma_growth;        // variation in growth rates
}

transformed parameters {
  vector[N] theta;
  
  for (n in 1:N) {
    theta[n] = theta_baseline[n] + growth_rate * time_point[n];
  }
}

model {
  // Priors
  theta_baseline ~ normal(0, 1);
  growth_rate ~ normal(0, 0.5);
  sigma_growth ~ cauchy(0, 0.5);
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
