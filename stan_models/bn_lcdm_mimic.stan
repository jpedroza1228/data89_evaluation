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
  vector<lower=0,upper=1>[K] theta;     // mastery prevalence
  vector[J] beta0;
  vector[J] beta1;
  vector[J] beta2;
  vector[J] beta12;
}
transformed parameters{
  matrix[I,C] pi;

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = inv_logit(beta0[i] +
      beta1[i] * alpha[c,1] * Q[i,1] +
      beta2[i] * alpha[c,2] * Q[i,2] +
      beta12[i] * alpha[c,1] * alpha[c,2] * Q[i,1] * Q[i,2]);
    }
  }
}
model{
  // Priors on mastery rates
  // if different prior expectancy of mastery for each attribute
  // theta[1] ~ beta(3, 1); 
  // theta[2] ~ beta(1, 1);

  //if same mastery then just loop through
  for (k in 1:K){
    theta[k] ~ beta(1, 1);
  }

  // Item priors
  beta0 ~ normal(0, 1);
  beta1 ~ normal(0, 1);
  beta2 ~ normal(0, 1);
  beta12 ~ normal(0, 1);

  vector[K] attr_lp;
  array[C] real ps;
  array[I] real eta;

  for (c in 1:C){
    for (k in 1:K){
      attr_lp[k] = alpha[c,k] * log(theta[k]) + (1 - alpha[c,k]) * log1m(theta[k]);
    }
  }

  for (j in 1:J) {
    for (c in 1:C){
      for (i in 1:I){
        real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
      }
      ps[c] = sum(attr_lp) + sum(eta); 
    }
    target += log_sum_exp(ps);
  }
}
generated quantities{
  matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent class c 
  matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
  vector[K] attr_lp;
  array[I] real eta;
  row_vector[C] prob_joint;
  array[C] real prob_attr_class;
  matrix[J,I] Y_rep;

  for (c in 1:C){
    for (k in 1:K){
      attr_lp[k] = alpha[c,k] * log(theta[k]) + (1 - alpha[c,k]) * log1m(theta[k]);
    }
  }

  for (j in 1:J){
    for (c in 1:C){
      for (i in 1:I){
        real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
      }
      prob_joint[c] = exp(sum(attr_lp)) * exp(sum(eta));
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
    int z = categorical_rng(attr_lp);  // sample class for person j
    for (i in 1:I) {
      Y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-attribute probability
    }
  }
}