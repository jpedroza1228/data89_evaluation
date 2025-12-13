// 2PL IRT Model 1: Two-Parameter Logistic Item Response Theory Model
// This model estimates item discrimination (a) and difficulty (b) parameters

data {
  int<lower=1> N;                    // number of students
  int<lower=1> J;                    // number of items
  int<lower=0,upper=1> Y[N, J];      // response matrix
}

parameters {
  vector[N] theta;                   // student ability parameters
  vector<lower=0>[J] a;              // item discrimination parameters
  vector[J] b;                       // item difficulty parameters
}

model {
  // Priors
  theta ~ normal(0, 1);              // standard normal prior for abilities
  a ~ lognormal(0, 1);               // lognormal prior for discrimination
  b ~ normal(0, 2);                  // normal prior for difficulty
  
  // Likelihood
  for (n in 1:N) {
    for (j in 1:J) {
      Y[n, j] ~ bernoulli_logit(a[j] * (theta[n] - b[j]));
    }
  }
}

generated quantities {
  matrix[N, J] prob_correct;         // probability of correct response
  vector[N] total_score;             // expected total score for each student
  
  for (n in 1:N) {
    total_score[n] = 0;
    for (j in 1:J) {
      prob_correct[n, j] = inv_logit(a[j] * (theta[n] - b[j]));
      total_score[n] += prob_correct[n, j];
    }
  }
}
