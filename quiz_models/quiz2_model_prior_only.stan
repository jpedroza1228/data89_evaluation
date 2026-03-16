data {
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> C;
  int<lower=1> K;
  matrix<lower=0,upper=1> [J,I] Y;
  matrix<lower=0,upper=1> [I,K] Q;
  matrix<lower=0,upper=1> [C,K] alpha;
}
parameters {
  // simplex[C] nu;
  ordered[C] raw_nu_ordered;
  vector<lower=0, upper=1>[I] slip;
  vector<lower=0, upper=1>[I] guess;
  real<lower=0, upper=1> lambda1;
  real<lower=0, upper=1> lambda2;
  real<lower=0, upper=1> lambda3;
  real<lower=0, upper=1> lambda4;
}
transformed parameters{
  simplex[C] nu;
  vector[C] theta1;
  vector[C] theta2;
  vector[C] theta3;
  vector[C] theta4;
  matrix[I,C] delta;
  matrix[I,C] pi;

  for (c in 1 : C) {
    theta1[c] = (alpha[c, 1] > 0) ? lambda1 : (1 - lambda1);    
    theta2[c] = (alpha[c, 2] > 0) ? lambda2 : (1 - lambda2);
    theta3[c] = (alpha[c, 3] > 0) ? lambda3 : (1 - lambda3);
    theta4[c] = (alpha[c, 4] > 0) ? lambda4 : (1 - lambda4);
  }

  nu = softmax(raw_nu_ordered);
  vector[C] log_nu = log(nu);
  
  for(c in 1:C){
    for(i in 1:I){
      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) * pow(1 - theta2[c], Q[i, 2]) * pow(1 - theta3[c], Q[i, 3]) * pow(1 - theta4[c], Q[i, 4]));
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow((1 - slip[i]), delta[i,c]) * pow(guess[i], (1 - delta[i,c]));
    }
  }
}
model {
  // Priors
  // nu ~ dirichlet(rep_vector(1.0, C));
  raw_nu_ordered ~ normal(0, 2);
  lambda1 ~ beta(20, 5); 
  lambda2 ~ beta(20, 5);
  lambda3 ~ beta(20, 5);
  lambda4 ~ beta(20, 5);
  
  for (i in 1:I){
    slip[i] ~ beta(5, 20);
    guess[i] ~ beta(5, 20);
  }
}
generated quantities {
  matrix[J,I] y_rep;
  
  for (j in 1:J) {
    int z = categorical_rng(nu);  // sample class for person j
    for (i in 1:I) {
      y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
    }
  }
}