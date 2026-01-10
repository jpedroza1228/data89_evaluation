data {
  int<lower=1> J;                    // number of students
  int<lower=1> I;                    // number of items
  matrix<lower=0,upper=1> [J,I] Y;   // response matrix
}
parameters {
  vector[J] theta;                   // student ability parameter
  vector[I] b;                       // item difficulty parameter
}
transformed parameters{
  matrix[J,I] eta;
  
  for (j in 1:J){
    for (i in 1:I){
      eta[j,i] = inv_logit(theta[j] - b[i]);
    }
  }
}
model {
  array[J] real log_lik;

  // Priors
  theta ~ normal(0, 1);              // standard normal prior for abilities
  b ~ normal(0, 2);                  // normal prior for difficulty

  // Likelihood
  for (j in 1:J) {
    log_lik[j] = 0;
    for (i in 1:I) {
      real p = fmin(fmax(eta[j,i], 1e-9), 1 - 1e-9);
      log_lik[j] += Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
    }
  }
}
generated quantities {
  matrix[J,I] prob_correct;
  matrix[J,I] Y_rep;
  for (j in 1:J) {
    for (i in 1:I) {
      prob_correct[j,i] = inv_logit(theta[j] - b[i]);
      Y_rep[j,i] = bernoulli_rng(prob_correct[j,i]);
    }
  }
}
