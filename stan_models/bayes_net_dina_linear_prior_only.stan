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
  real<lower=0,upper=1> lambda2;

  vector[I] beta0;
  vector<lower=0>[I] beta1;
  vector<lower=0>[I] beta2;
  vector[I] beta12;
}
transformed parameters{
  simplex[C] nu;
  vector[C] theta1;
  vector[C] theta2;
  matrix[I,C] delta;
  matrix[I,C] pi;

  theta1[1] = (1 - lambda1);
  theta1[2] = (1 - lambda1);
  theta1[3] = lambda1;
  theta1[4] = lambda1;

  theta2[1] = (1 - lambda2);
  theta2[2] = lambda2;
  theta2[3] = (1 - lambda2);
  theta2[4] = lambda2;

  nu[1] = theta1[1] * theta2[1];
  nu[2] = theta1[2] * theta2[2];
  nu[3] = theta1[3] * theta2[3];
  nu[4] = theta1[4] * theta2[4];

  vector[C] log_nu = log(nu);

  for(c in 1:C){
    for(i in 1:I){
      delta[i, c] = pow(theta1[c], Q[i, 2]) * pow(theta2[c], Q[i, 2]);
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow((tp[i]), delta[i,c]) * pow(fp[i], (1 - delta[i,c]));
    }
  }
}
model{
  // array[C] real ps;
  // array[I] real eta;
   
  // Priors for attribute mastery probabilities
  lambda1 ~ beta(20, 5);
  lambda2 ~ beta(20, 5);

  //priors for items to attributes
  beta0 ~ normal(0, 1);
  beta1 ~ normal(0, 1);
  beta2 ~ normal(0, 1);
  beta12 ~ normal(0, 1);

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
  // matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent class c 
  // matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
  // array[I] real eta;
  // row_vector[C] prob_joint;
  // array[C] real prob_attr_class;
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