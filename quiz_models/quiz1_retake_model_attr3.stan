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
parameters {
  simplex[C] nu_t1;

  vector<lower=0, upper=1>[I] tp_t1; //slip (1 - tp)
  vector<lower=0, upper=1>[I] fp_t1; //guess
  vector<lower=0, upper=1>[I] tp_t2; //slip (1 - tp)
  vector<lower=0, upper=1>[I] fp_t2; //guess

  // transitions from time point 1 to time point 2
  // dino parameters
  real<lower=0,upper=1> lambda1_t1;
  real<lower=0,upper=1> lambda2_t1;
  real<lower=0,upper=1> lambda3_t1;
  real<lower=0,upper=1> lambda1_t2;
  real<lower=0,upper=1> lambda2_t2;
  real<lower=0,upper=1> lambda3_t2;

  // lcdm parameters
  // time point 1 parameters
  // real gamma101; //baseline for attribute 1 
  // real gamma102; //baseline for attribute 2 
  // real gamma103; //baseline for attribute 3
  // real gamma111; //mastery coefficient for time point 1 (att 1)
  // real gamma121; //mastery coefficient for time point 1 (att 2)
  // real gamma131; //mastery coefficient for time point 1 (att 3)

  // time point 2 parameters
  // real gamma201; //baseline for attribute 1 
  // real gamma202; //baseline for attribute 2 
  // real gamma203; //baseline for attribute 3
  // real gamma211; //mastery coefficient for time point 2 (att 1)
  // real gamma221; //mastery coefficient for time point 2 (att 2)
  // real gamma231; //mastery coefficient for time point 2 (att 3)

  //item parameters
  // vector[I] beta0;
  // vector[I] beta1;
  // vector[I] beta2;
  // vector[I] beta12;
  // vector[I] beta13;
  // vector[I] beta23;
  // vector[I] beta123;
}
transformed parameters{
  matrix[C, C] trans_mat;
  vector[C] theta1_t1;
  vector[C] theta2_t1;
  vector[C] theta3_t1;
  vector[C] theta1_t2;
  vector[C] theta2_t2;
  vector[C] theta3_t2;
  matrix[I,C] delta_t1;
  matrix[I,C] delta_t2;
  matrix[I,C] pi_t1;
  matrix[I,C] pi_t2;

  for (c in 1:C){
    for (d in 1:C){
      // theta1_t1[c] = alpha[c,1] > 0 ? inv_logit(gamma101 + gamma111) : inv_logit(gamma101);
      // theta2_t1[c] = alpha[c,2] > 0 ? inv_logit(gamma102 + gamma121) : inv_logit(gamma102);
      // theta3_t1[c] = alpha[c,3] > 0 ? inv_logit(gamma103 + gamma131) : inv_logit(gamma103);

      theta1_t1[c] = alpha[c,1] > 0 ? lambda1_t1 : (1 - lambda1_t1);
      theta2_t1[c] = alpha[c,2] > 0 ? lambda2_t1 : (1 - lambda2_t1);
      theta3_t1[c] = alpha[c,3] > 0 ? lambda3_t1 : (1 - lambda3_t1);

      theta1_t2[d] = alpha[d,1] > 0 ? lambda1_t2 : (1 - lambda1_t2);
      theta2_t2[d] = alpha[d,2] > 0 ? lambda2_t2 : (1 - lambda2_t2);
      theta3_t2[d] = alpha[d,3] > 0 ? lambda3_t2 : (1 - lambda3_t2);

      trans_mat[c, d] = theta1_t2[d] * theta2_t2[d] * theta3_t2[d];
    }
  }

  // nu[1] = (1 - inv_logit(gamma1)) * (1 - inv_logit(gamma2));
  // nu[2] = inv_logit(gamma1) * (1 - inv_logit(gamma2 + gamma21));
  // nu[3] = (1 - inv_logit(gamma1)) * inv_logit(gamma2);
  // nu[4] = inv_logit(gamma1) * inv_logit(gamma2 + gamma21);

  vector[C] log_nu_t1 = log(nu_t1);

  // for (c in 1:C){
  //   for (i in 1:I){
  //     pi_t1[i,c] = inv_logit(beta0[i] +
  //     beta1[i] * alpha[c,1] +
  //     beta2[i] * alpha[c,2] +
  //     beta12[i] * alpha[c,1] * alpha[c,2] +
  //     beta13[i] * alpha[c,1] * alpha[c,3] +
  //     beta23[i] * alpha[c,2] * alpha[c,3] +
  //     beta123[i] * alpha[c,1] * alpha[c,2] * alpha[c,3]);

  //     pi_t2[i,c] = inv_logit(beta0[i] +
  //     beta1[i] * alpha[c,1] +
  //     beta2[i] * alpha[c,2] +
  //     beta12[i] * alpha[c,1] * alpha[c,2] +
  //     beta13[i] * alpha[c,1] * alpha[c,3] +
  //     beta23[i] * alpha[c,2] * alpha[c,3] +
  //     beta123[i] * alpha[c,1] * alpha[c,2] * alpha[c,3]);
  //   }
  // }

  for(c in 1:C){
    for(i in 1:I){
      delta_t1[i, c] = 1 - (pow(1 - theta1_t1[c], Q[i, 1]) * pow(1 - theta2_t1[c], Q[i, 2]) * pow(1 - theta3_t1[c], Q[i, 3]));

      delta_t2[i, c] = 1 - (pow(1 - theta1_t2[c], Q[i, 1]) * pow(1 - theta2_t2[c], Q[i, 2]) * pow(1 - theta3_t2[c], Q[i, 3]));
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi_t1[i,c] = pow((tp_t1[i]), delta_t1[i,c]) * pow(fp_t1[i], (1 - delta_t1[i,c]));

      pi_t2[i,c] = pow((tp_t2[i]), delta_t2[i,c]) * pow(fp_t2[i], (1 - delta_t2[i,c]));
    }
  }
}
model{
  array[C] real ps_t1;
  array[I] real eta_t1;
  array[C] real ps_t2;
  array[I] real eta_t2;

  // priors
  nu_t1 ~ dirichlet(rep_vector(1.0, C)); // uniform prior for the number of classes
  lambda1_t1 ~ beta(1, 1);
  lambda2_t1 ~ beta(1, 1);
  lambda3_t1 ~ beta(1, 1);
  lambda1_t2 ~ beta(20, 5);
  lambda2_t2 ~ beta(20, 5);
  lambda3_t2 ~ beta(20, 5);

  for (i in 1:I){
    tp_t1[i] ~ beta(20, 5);
    fp_t1[i] ~ beta(5, 20);
    tp_t2[i] ~ beta(20, 5);
    fp_t2[i] ~ beta(5, 20);
  }

  for (j in 1:J) {
    for (c in 1:C){
      for (i in 1:I){
        real p_t1 = fmin(fmax(pi_t1[i,c], 1e-9), 1 - 1e-9);
        eta_t1[i] = Y_t1[j,i] * log(p_t1) + (1 - Y_t1[j,i]) * log1m(p_t1);
      }

    for (d in 1:C){
      for (i in 1:I){
        real p_t2 = fmin(fmax(pi_t2[i,d], 1e-9), 1 - 1e-9);
        eta_t2[i] = Y_t2[j,i] * log(p_t2) + (1 - Y_t2[j,i]) * log1m(p_t2);
      }
      ps_t2[d] = log(trans_mat[c,d]) + sum(eta_t2); 
    }
    ps_t1[c] = log_nu_t1[c] + sum(eta_t1) + log_sum_exp(ps_t2);
    }
    target += log_sum_exp(ps_t1);
  }
}
generated quantities {
  matrix[J,C] prob_class_t1;
  matrix[J,C] prob_class_t2;
  matrix[J,K] prob_attr_t1;
  matrix[J,K] prob_attr_t2;
  array[I] real eta_t1;
  array[I] real eta_t2;
  matrix[C, C] log_joint;
  matrix[C, C] post_joint;
  vector[C] t1_to_d_paths;
  simplex[C] nu_t2; // Predicted population proportions at Time 2
  vector[K] growth_rate; // Net growth per attribute (T2 - T1)
  matrix[J, I] y_rep_t1;
  matrix[J, I] y_rep_t2;
  
  for (j in 1:J) {
    for (c in 1:C) {
      for (i in 1:I){
        real p_t1 = fmin(fmax(pi_t1[i,c], 1e-9), 1 - 1e-9);
        eta_t1[i] = Y_t1[j,i] * log(p_t1) + (1 - Y_t1[j,i]) * log1m(p_t1);
      
      for (d in 1:C) {
        real p_t2 = fmin(fmax(pi_t2[i,d], 1e-9), 1 - 1e-9);
        eta_t2[i] = Y_t2[j,i] * log(p_t2) + (1 - Y_t2[j,i]) * log1m(p_t2);
        
        log_joint[c, d] = log_nu_t1[c] + sum(eta_t1) + log(trans_mat[c, d]) + sum(eta_t2);
      }
    }
  }
    
    // Normalize to get posterior class probabilities
    real evidence = log_sum_exp(to_vector(log_joint));
    post_joint = exp(log_joint - evidence);

    // Marginalize for T1 and T2 class probs
    for (c in 1:C) {
      prob_class_t1[j, c] = sum(post_joint[c, ]); // Sum across rows (T2)
      prob_class_t2[j, c] = sum(post_joint[, c]); // Sum across columns (T1)
    }

    // Marginalize for Attribute mastery
    for (k in 1:K) {
      real sum_t1 = 0;
      real sum_t2 = 0;
      for (c in 1:C) {
        if (alpha[c, k] > 0) {
          sum_t1 = sum_t1 + prob_class_t1[j, c];
          sum_t2 = sum_t2 + prob_class_t2[j, c];
        }
      }
      prob_attr_t1[j, k] = sum_t1;
      prob_attr_t2[j, k] = sum_t2;
    }
  }

  // 1. Calculate T2 Population Proportions
  // nu_t2 = nu_t1 * trans_mat
  for (d in 1:C) {
    for (c in 1:C) {
      t1_to_d_paths[c] = nu_t1[c] * trans_mat[c, d];
    }
    nu_t2[d] = sum(t1_to_d_paths);
  }

  // 2. Calculate Growth per Attribute
  for (k in 1:K) {
    real prop_t1 = 0;
    real prop_t2 = 0;
    for (c in 1:C) {
      if (alpha[c, k] > 0) {
        prop_t1 = prop_t1 + nu_t1[c];
        prop_t2 = prop_t2 + nu_t2[c];
      }
    }
    // Net percentage point increase/decrease
    growth_rate[k] = prop_t2 - prop_t1;
  }
  
  for (j in 1:J) {
    // 1. Sample Time 1 Latent Class
    // nu_t1 is the population distribution at T1
    int z1 = categorical_rng(nu_t1);
    
    // 2. Sample Time 2 Latent Class based on T1 Class
    // We extract the row of the transition matrix corresponding to z1
    vector[C] t2_transition_probs = to_vector(trans_mat[z1, ]);
    int z2 = categorical_rng(t2_transition_probs);
    
    // 3. Generate Replicated Responses
    for (i in 1:I) {
      // Response at T1 based on Class z1
      y_rep_t1[j, i] = bernoulli_rng(pi_t1[i, z1]);
      
      // Response at T2 based on Class z2
      y_rep_t2[j, i] = bernoulli_rng(pi_t2[i, z2]);
    }
  }
}