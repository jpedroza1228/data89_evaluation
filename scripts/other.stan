data{
  int<lower=1> J; // Students
  int<lower=1> I; // Items
  int<lower=1> C; // Latent States/Classes
  int<lower=1> K; // Attributes
  matrix<lower=0,upper=1> [J,I] Y;
  matrix<lower=0,upper=1> [I,K] Q;
  matrix<lower=0,upper=1> [C,K] alpha;
}
parameters {
  ordered[C] raw_nu; // strictly increasing latent values to keep order of classes
  real<lower=0, upper=1> lambda1;
  // real<lower=0, upper=1> lambda2;
  real gamma0;
  real gamma1;
  vector[I] beta0;
  vector[I] beta1;
  vector[I] beta2;
  // vector[I] beta12;
}
transformed parameters{
  matrix[I,C] pi;
  vector[K] theta;
  simplex[C] nu; // final probability of class membership

  nu = softmax(raw_nu); // transforms ordered vector into simplex
  vector[C] log_nu = log(nu);

  // real no_mastery = inv_logit(gamma0);
  // real mastery = inv_logit(gamma0 + gamma1);

  // // lambda1 & lambda2 allow for different initial mastery priors
  // theta[1] = log1m(lambda1) + log1m(no_mastery);  // A1 = 0, A2 = 0
  // theta[2] = log(lambda1) + log1m(mastery); // A1 = 1, A2 = 0
  // theta[3] = log1m(lambda2) + log(mastery); // A1 = 0, A2 = 1
  // theta[4] = log(lambda2) + log(mastery); // A1 = 1, A2 = 1

  // Linear BN
  for (c in 1:C){
    theta[1] = alpha[c,1] * log(lambda1) + (1 - alpha[c,1]) * log1m(lambda1);
    real inv_attr_lp1 = inv_logit(gamma0 + gamma1 * alpha[c,2]);
    theta[2] = alpha[c,2] * log(inv_attr_lp1) + (1 - alpha[c,2]) * log1m(inv_attr_lp1);
  }

  // LCDM-style BN
  // for (c in 1:C){
  //   real inv_attr_lp1 = inv_logit(gamma0 + gamma1 * alpha[c,1]);
  //   theta[1] = alpha[c,1] * log(lambda1) + (1 - alpha[c,1]) * log1m(lambda1);
  //   real inv_attr_lp2 = inv_logit(gamma0 + gamma1 * alpha[c,2]);
  //   theta[2] = alpha[c,2] * log(lambda2) + (1 - alpha[c,2]) * log1m(lambda2);
  // }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = inv_logit(beta0[i] +
      beta1[i] * Q[i,1] +
      beta2[i] * Q[i,2]);
      // beta12[i] * Q[i,1] * Q[i,2]); # only if items are in both 
    }
  }
}
model {
  array[I] real eta;
  row_vector[C] ps;

  // Priors
  lambda1 ~ beta(20, 5);
  // lambda2 ~ beta(20, 5);

  gamma0 ~ normal(0, 1);
  gamma1 ~ normal(0, 1);
  beta0 ~ normal(0, 1);
  beta1 ~ normal(0, 1);
  // beta12 ~ normal(0, 1);

  raw_nu ~ normal(0, 1); // if label switching happens 

  // Likelihood
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
// generated quantities {
//   matrix[J,C] prob_resp_class;      // posterior probabilities of respondent j being in latent state/class c 
//   matrix[J,K] prob_resp_attr;       // posterior probabilities of respondent j being a master of attribute k 
//   array[I] real eta;
//   // row_vector[C] ps;
//   array[C] real prob_attr_class;
//   matrix[J,I] Y_rep;

//   for (j in 1:J){
//     row_vector[C] ps;
//     for (c in 1:C){
//       for (i in 1:I){
//         real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);
//         eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);
//       }
//       ps[c] = exp(theta[c]) + exp(sum(eta));
//     }
//     prob_resp_class[j] = ps/sum(ps);
//   }

//   for (j in 1:J){
//     for (k in 1:K){
//       for (c in 1:C){
//         prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];
//       }
//       prob_resp_attr[j,k] = sum(prob_attr_class);
//     }
//   }
  
//   for (j in 1:J) {
//     int z = categorical_rng(nu);  // sample class for person j
//     for (i in 1:I) {
//       Y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
//     }
//   }
// }