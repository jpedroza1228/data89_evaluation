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
  // Priors
  theta ~ normal(0, 1);              // standard normal prior for abilities
  for (i in 1:I){
    b[i] ~ normal(0, 2);                  // normal prior for difficulty
  }
}
generated quantities {
  matrix[J,I] prob_correct;
  matrix[J,I] y_rep;

  for (j in 1:J) {
    for (i in 1:I) {
      prob_correct[j,i] = inv_logit(theta[j] - b[i]);
      y_rep[j,i] = bernoulli_rng(prob_correct[j,i]);
    }
  }
}
