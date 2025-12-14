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
  real<lower=0,upper=1> theta1;    // P(A1 = 1)

  real gamma0; // baseline for attribute 2
  real gamma1; // linear effect of attribute 1 --> attribute 2

  vector[J] beta0;
  vector[J] beta1;
  vector[J] beta2;
  vector[J] beta12;
}
transformed parameters{
  vector[K] attr_lp;
  real inv_attr_lp1;
  matrix[I,C] pi;

  for (c in 1:C){
      attr_lp[1] = alpha[c,1] * log(theta1) + (1 - alpha[c,1]) * log1m(theta1);
      inv_attr_lp1 = inv_logit(gamma0 + gamma1 * alpha[c,2]);
      attr_lp[2] = alpha[c,2] * log(inv_attr_lp1) + (1 - alpha[c,2]) * log1m(inv_attr_lp1);
  }

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
  array[C] real ps;
  array[I] real eta;

  // Priors
  theta1 ~ beta(1, 1); //uniform prior only for attribute 1 only

  //priors for attribute edge att1 --> att2
  gamma0 ~ normal(0, 1);
  gamma1 ~ normal(0, 1);

  //priors for items to attributes
  beta0 ~ normal(0, 1);
  beta1 ~ normal(0, 1);
  beta2 ~ normal(0, 1);
  beta12 ~ normal(0, 1);

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

