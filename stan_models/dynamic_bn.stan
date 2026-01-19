// need to double check the attr_lp1 & attr_lp2 values to make sure they
// should include loops over attributes & classes
data{
  int<lower=1> J; // students
  int<lower=1> I; // items per time
  int<lower=1> T; // time points
  int<lower=1> K; // attributes
  int<lower=1> C; // latent classes
  matrix<lower=0, upper=1> [J, I] Y_t1;
  matrix<lower=0, upper=1> [J, I] Y_t2;
  matrix<lower=0,upper=1> [I,K] Q;
  matrix<lower=0,upper=1> [C,K] alpha;
}
parameters{
  // Time 1
  real<lower=0,upper=1> theta1_t1;
  real<lower=0,upper=1> theta2_t1;

  // Autoregressive transitions
  real gamma1_A1;  // A1_t1 -> A1_t2
  real gamma2_A2;  // A2_t1 -> A2_t2

  // Optional A1 -> A2 dependency
  // real gamma_A1_to_A2;

  // Item parameters (same as before)
  vector[I] beta0;
  vector[I] beta1;
  vector[I] beta2;
  vector[I] beta12;
}
transformed parameters{
  // compute conditional probabilities for all 4 patterns at t1 and t2
  matrix[C,C] p_t2_given_t1; // pattern c at t1 → pattern d at t2
  vector[K] attr_lp;
  matrix[I,C] pi_t1;
  matrix[I,C] pi_t2;

  for(c in 1:C) {
    for(d in 1:C){
      // independent BN: A1_t2 ~ logit(gamma1*A1_t1)
      real pA1 = inv_logit(logit(theta1_t1) + gamma1_A1 * alpha[c,1]);
      real pA2 = inv_logit(logit(theta2_t1) + gamma2_A2 * alpha[c,2]);

      // dependent BN: include gamma_A1_to_A2
      // pA2 = inv_logit(logit(theta2_t1) + gamma2_A2*alpha[c,2] + gamma_A1_to_A2*A1_t2);

      p_t2_given_t1[c,d] = (alpha[d,1] == 1 ? pA1 : 1 - pA1) * (alpha[d,2] == 1 ? pA2 : 1 - pA2);
    }
  }
  for (c in 1:C){
    attr_lp[1] = alpha[c,1] * log(theta1_t1) + (1 - alpha[c,1]) * log1m(theta1_t1);
    attr_lp[2] = alpha[c,2] * log(theta2_t1) + (1 - alpha[c,2]) * log1m(theta2_t1);
  }

  for (c in 1:C){
    for (i in 1:I){
      pi_t1[i,c] = inv_logit(beta0[i] +
      beta1[i] * alpha[c,1] * Q[i,1] +
      beta2[i] * alpha[c,2] * Q[i,2] +
      beta12[i] * alpha[c,1] * alpha[c,2] * Q[i,1] * Q[i,2]);
    }
  }

  for (d in 1:C){
    for (i in 1:I){
      pi_t2[i,d] = inv_logit(beta0[i] +
      beta1[i] * alpha[d,1] * Q[i,1] +
      beta2[i] * alpha[d,2] * Q[i,2] +
      beta12[i] * alpha[d,1] * alpha[d,2] * Q[i,1] * Q[i,2]);
    }
  }
}
model{
  vector[C] attr_lp2;
  array[C] real ps_t1;
  array[I] real eta_t1;
  array[C] real ps_t2;
  array[I] real eta_t2;

  // priors
  theta1_t1 ~ beta(5,5);
  theta2_t1 ~ beta(5,5);
  gamma1_A1 ~ normal(0,1);
  gamma2_A2 ~ normal(0,1);
  // gamma_A1_to_A2 ~ normal(0,1);
  beta0 ~ normal(0,1);
  beta1 ~ normal(0,1);
  beta2 ~ normal(0,1);
  beta12 ~ normal(0,1);

  // likelihood (marginalize over attributes)
  for (j in 1:J) {
    for (c in 1:C){
      for (i in 1:I){
        real p_t1 = fmin(fmax(pi_t1[i,c], 1e-9), 1 - 1e-9);
        eta_t1[i] = Y_t1[j,i] * log(p_t1) + (1 - Y_t1[j,i]) * log1m(p_t1);
      }
      ps_t1[c] = sum(attr_lp) + sum(eta_t1); 
    }

    for (d in 1:C) {
    for (c in 1:C){
       attr_lp2[c] = exp(ps_t1[c]) * p_t2_given_t1[c,d];
      }
      for (i in 1:I){
        real p_t2 = fmin(fmax(pi_t2[i,d], 1e-9), 1 - 1e-9);
        eta_t2[i] = Y_t2[j,i] * log(p_t2) + (1 - Y_t2[j,i]) * log1m(p_t2);
      }
      // find out why sum(attr_lp) is not working
      ps_t2[d] = sum(attr_lp2) + sum(eta_t2); 
    }
    target += log_sum_exp(ps_t2);
  }
}
generated quantities{
// this will need some work
// it will include t1 & t2 to create replicated values for too
}