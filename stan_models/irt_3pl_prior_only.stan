data {
  int<lower=1> J;                    // number of students
  int<lower=1> I;                    // number of items
  matrix<lower=0,upper=1> [J,I] Y;   // response matrix
}
parameters {
  vector[J] theta;                   // student ability parameter
  vector<lower=0>[I] a;              // item discrimination parameter
  vector[I] b;                       // item difficulty parameter
  vector<lower=0,upper=1>[I] c;   // guessing parameter
}
transformed parameters{
  matrix[J,I] eta;
  for (j in 1:J){
    for (i in 1:I){
      eta[j,i] = c[i] + (1 - c[i]) * inv_logit(a[i] * (theta[j] - b[i]));
    }
  }
}
model {
  array[J] real log_lik;

  // Priors
  theta ~ normal(0, 1);              // standard normal prior for abilities
  for (i in 1:I){
    a[i] ~ lognormal(0.5, 1);             // lognormal prior for discrimination
    b[i] ~ normal(0, 2);                  // normal prior for difficulty
    c[i] ~ normal(0, 2);                    // prior for guessing (most mass near 0)
  }

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
  array[J] real log_lik;
  matrix[J,I] prob_correct;
  matrix[J,I] y_rep;

  // Likelihood
  for (j in 1:J) {
    log_lik[j] = 0;
    for (i in 1:I) {
      real p = fmin(fmax(eta[j,i], 1e-9), 1 - 1e-9);
      log_lik[j] += Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
    }
  }

  for (j in 1:J) {
    for (i in 1:I) {
      prob_correct[j,i] = c[i] + (1 - c[i]) * inv_logit(a[i] * (theta[j] - b[i]));
      y_rep[j,i] = bernoulli_rng(prob_correct[j,i]);
    }
  }
}
