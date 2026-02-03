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
  vector<lower=0, upper=1>[I] tp; //slip (1 - tp)
  vector<lower=0, upper=1>[I] fp; //guess

  real<lower=0,upper=1> lambda1;
  real<lower=0,upper=1> lambda20;
  real<lower=0,upper=1> lambda21;
  real<lower=0,upper=1> lambda22;
  real<lower=0,upper=1> lambda30;
  real<lower=0,upper=1> lambda31;
  real<lower=0,upper=1> lambda32;
  real<lower=0,upper=1> lambda40;
  real<lower=0,upper=1> lambda41;
  real<lower=0,upper=1> lambda50;
  real<lower=0,upper=1> lambda51;
}
transformed parameters{
  vector[C] raw_nu;
  simplex[C] nu;
  // ordered[C] raw_nu; // if label switching happens    
  vector[C] theta1;
  vector[C] theta2;
  vector[C] theta3;
  vector[C] theta4;
  vector[C] theta5;
  matrix[I,C] delta;
  matrix[I,C] pi;

  // This model is A1 --> A2 --> A3; A4 & A5
  // All beta priors, even with linear edges, are predetermined with priors rather than logistic functions
  // Include option for having inv_logit(gamma10 + gamma11 * alpha[c,1]) with normal priors as another option

  for (c in 1:C) {
    // variable = condition ? value_if_true : value_if_false;
    theta1[c] = (alpha[c, 1] > 0) ? lambda1 : (1 - lambda1);
    theta2[c] = (alpha[c,1] > 0 && alpha[c,2] > 0) ? lambda22 : (alpha[c,1] > 0 || alpha[c,2] > 0) ? lambda21 : lambda20; // this show linear relationship with priors
    theta3[c] = (alpha[c,2] > 0 && alpha[c,3] > 0) ? lambda32 : (alpha[c,2] > 0 || alpha[c,3] > 0) ? lambda31 : lambda30;
    theta4[c] = (alpha[c, 4] > 0) ? lambda41 : lambda40;
    theta5[c] = (alpha[c, 5] > 0) ? lambda51 : lambda50;
    raw_nu[c] = theta1[c] * theta2[c] * theta3[c] * theta4[c] * theta5[c];
  }

  nu = raw_nu / sum(raw_nu);
  vector[C] log_nu = log(nu);

  for(c in 1:C){
    for(i in 1:I){
      delta[i, c] = pow(theta1[c], Q[i, 2]) * pow(theta2[c], Q[i, 2]);
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow(tp[i], delta[i,c]) * pow(fp[i], (1 - delta[i,c]));
    }
  }
}
model{
  array[C] real ps;
  array[I] real eta;
   
  // Priors for attribute mastery probabilities
  lambda1 ~ beta(20, 5);
  lambda20 ~ beta(5, 20);
  lambda21 ~ beta(12.5, 12.5);
  lambda22 ~ beta(20, 5);
  lambda30 ~ beta(5, 20);
  lambda31 ~ beta(12.5, 12.5);
  lambda32 ~ beta(20, 5);
  lambda40 ~ beta(5, 20);
  lambda41 ~ beta(20, 5);
  lambda50 ~ beta(5, 20);
  lambda51 ~ beta(20, 5);

  for (i in 1:I){
    tp[i] ~ beta(20, 5);
    fp[i] ~ beta(5, 20);
  }

  // likelihood
  for (j in 1:J) {
    for (c in 1:C){
      for (i in 1:I){
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

  // likelihood
  for (j in 1:J){
    for (c in 1:C){
      for (i in 1:I){
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