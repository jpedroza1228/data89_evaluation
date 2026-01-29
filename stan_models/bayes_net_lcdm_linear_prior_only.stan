data{
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> C;
  int<lower=1> K;
  matrix<lower=0,upper=1> [J,I] Y;
  matrix<lower=0,upper=1> [I,K] Q;
  matrix<lower=0,upper=1> [C,K] alpha;
}
parameters{
  real gamma1;

  real gamma2; // baseline for attribute 2
  real<lower=0> gamma21; // linear effect of attribute 1 --> attribute 2

  vector[I] beta0;
  vector<lower=0>[I] beta1;
  vector<lower=0>[I] beta2;
  vector[I] beta12;
}
transformed parameters{
  simplex[C] nu;
  matrix[I,C] pi;

  nu[1] = (1 - inv_logit(gamma1)) * (1 - inv_logit(gamma2));
  nu[2] = inv_logit(gamma1) * (1 - inv_logit(gamma2 + gamma21));
  nu[3] = (1 - inv_logit(gamma1)) * inv_logit(gamma2);
  nu[4] = inv_logit(gamma1) * inv_logit(gamma2 + gamma21);

  vector[C] log_nu = log(nu);

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = inv_logit(beta0[i] +
      beta1[i] * alpha[c,1] +
      beta2[i] * alpha[c,2] +
      beta12[i] * alpha[c,1] * alpha[c,2]);
    }
  }
}
model{
  // array[C] real ps;
  // array[I] real eta;

  // Priors
  gamma1 ~ normal(0, 1);
  gamma2 ~ normal(0, 1);
  gamma21 ~ normal(0, 1);

  //priors for items to attributes
  for (i in 1:I){
    beta0[i] ~ normal(0, 1);
    beta1[i] ~ normal(0, 1);
    beta2[i] ~ normal(0, 1);
    beta12[i] ~ normal(0, 1);
  }

  // for (j in 1:J) {
  //   for (c in 1:C){
  //     for (i in 1:I){
  //       real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
  //       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
  //     }
  //     ps[c] = log_nu[c] + sum(eta); 
  //   }
  //   target += log_sum_exp(ps);
  //   }
}
generated quantities{
  //  matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent class c 
  // matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
  // array[I] real eta;
  // row_vector[C] prob_joint;
  // array[C] real prob_attr_class;
  matrix[J,I] y_rep;

  // for (j in 1:J){
  //   for (c in 1:C){
  //     for (k in 1:K){
  //     for (i in 1:I){
  //       // eta[i] = bernoulli_lpmf(Y[j,i] | pi[i,c]);
  //       real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
  //       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
  //     }
  //     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));
  //     }
  //   }
  //   prob_resp_class[j] = prob_joint/sum(prob_joint);
  // }
  // for (j in 1:J){
  //   for (k in 1:K){
  //     for (c in 1:C){
  //       prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];
  //     }
  //     prob_resp_attr[j,k] = sum(prob_attr_class);
  //   }
  // }
  
  for (j in 1:J) {
    int z = categorical_rng(nu);  // sample class for person j
    for (i in 1:I) {
      y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
    }
  }
}