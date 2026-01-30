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
  // ordered[C] raw_nu; // if label switching happens    
  vector<lower=0, upper=1>[I] tp; //slip (1 - tp)
  vector<lower=0, upper=1>[I] fp; //guess

  real<lower=0,upper=1> lambda1;
  real gamma20;
  real<lower=0> gamma21;
  real gamma30;
  real<lower=0> gamma31;
  real<lower=0,upper=1> lambda4;
  real<lower=0,upper=1> lambda5;
}
transformed parameters{
  simplex[C] nu;
  vector[C] theta1;
  vector[C] theta2;
  vector[C] theta3;
  vector[C] theta4;
  vector[C] theta5;
  matrix[I,C] delta;
  matrix[I,C] pi;
  vector[C] nu_unnormalized;

  // This model is A1 --> A2 --> A3; A4 & A5
  for (c in 1:C) {
    // variable = condition ? value_if_true : value_if_false;
    theta1[c] = (alpha[c, 1] > 0) ? lambda1 : (1 - lambda1);
    theta2[c] = inv_logit(gamma20 + gamma21 * theta1[c]);
    theta3[c] = inv_logit(gamma30 + gamma31 * theta2[c]);
    theta4[c] = (alpha[c, 4] > 0) ? lambda4 : (1 - lambda4);
    theta5[c] = (alpha[c, 5] > 0) ? lambda5 : (1 - lambda5);
    nu_unnormalized[c] = theta1[c] * theta2[c] * theta3[c] * theta4[c] * theta5[c];
  }

  nu = nu_unnormalized / sum(nu_unnormalized);

  vector[C] log_nu = log(nu);

  for(c in 1:C){
    for(i in 1:I){
      delta[i, c] = pow(theta1[c], Q[i, 2]) * pow(theta2[c], Q[i, 2]) * pow(theta3[c], Q[i, 3]) * pow(theta4[c], Q[i, 4]) * pow(theta5[c], Q[i, 5]);
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow((tp[i]), delta[i,c]) * pow(fp[i], (1 - delta[i,c]));
    }
  }
}
model{
  array[C] real ps;
  array[I] real eta;
   
  // Priors for attribute mastery probabilities
  lambda1 ~ beta(20, 5);
  // lambda2 ~ beta(20, 5);
  // lambda3 ~ beta(20, 5);
  gamma20 ~ normal(0, 2);
  gamma21 ~ normal(0, 2);
  gamma30 ~ normal(0, 2);
  gamma31 ~ normal(0, 2);
  lambda4 ~ beta(20, 5);
  lambda5 ~ beta(20, 5);

  for (i in 1:I){
    tp[i] ~ beta(20, 5);
    fp[i] ~ beta(5, 20);
  }

  // likelihood
  // for (j in 1:J) {
  //   for (c in 1:C){
  //     for (i in 1:I){
  //       real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
  //       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
  //     }
  //     ps[c] = log_nu[c] + sum(eta); 
  //   }
  //   target += log_sum_exp(ps);
  // }
}
generated quantities {
  matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent class c 
  matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
  array[I] real eta;
  row_vector[C] prob_joint;
  array[C] real prob_attr_class;
  matrix[J,I] y_rep;

  // likelihood
  // for (j in 1:J){
  //   for (c in 1:C){
  //     for (i in 1:I){
  //       real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
  //       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
  //     }
  //     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));
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