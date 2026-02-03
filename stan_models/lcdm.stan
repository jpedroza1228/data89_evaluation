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
  array[C] real ps;
  array[I] real eta;

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

  for (j in 1:J) {
    for (c in 1:C){
      for (i in 1:I){
        // eta[i] = bernoulli_lpmf(Y[j,i] | pi[i,c]);
        real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
      }
      ps[c] = log_nu[c] + sum(eta); 
    }
    target += log_sum_exp(ps);
  }
}
generated quantities {
  matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent class c 
  matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
  array[I] real eta;
  row_vector[C] prob_joint;
  array[C] real prob_attr_class;
  matrix[J,I] y_rep;

  for (j in 1:J){
    for (c in 1:C){
      for (i in 1:I){
        // eta[i] = bernoulli_lpmf(Y[j,i] | pi[i,c]);
        real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
      }
      prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));
    }
    prob_resp_class[j] = prob_joint/sum(prob_joint);
  }
  for (j in 1:J){
    for (k in 1:K){
      for (c in 1:C){
        prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];
      }
      prob_resp_attr[j,k] = sum(prob_attr_class);
    }
  }
  
  for (j in 1:J) {
    int z = categorical_rng(nu);  // sample class for person j
    for (i in 1:I) {
      y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
    }
  }
}