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
  // ordered[C] raw_nu; 
  simplex[C] nu; 
  vector[I] beta0;
  vector<lower=0>[I] beta1;
  vector<lower=0>[I] beta2;
  vector<lower=0>[I] beta3;
  vector[I] beta13;
  vector[I] beta23;
}
transformed parameters{
  matrix[I,C] pi;

  // nu = softmax(raw_nu);
  vector[C] log_nu = log(nu);

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = inv_logit(beta0[i] +
      beta1[i] * alpha[c,1] +
      beta2[i] * alpha[c,2] +
      beta3[i] * alpha[c,3] +
      beta13[i] * alpha[c,1] * alpha[c,3] +
      beta23[i] * alpha[c,2] * alpha[c,3]
    );
    }
  }
}
model {
  // Priors
  // raw_nu ~ normal(0, 1);
  nu  ~ dirichlet(rep_vector(1.0, C));
  for (i in 1:I){
    beta0[i] ~ normal(0, 1);
    beta1[i] ~ lognormal(0, 1);
    beta2[i] ~ lognormal(0, 1);
    beta3[i] ~ lognormal(0, 1);
    beta13[i] ~ normal(0, 1);
    beta23[i] ~ normal(0, 1);
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