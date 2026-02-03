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
  vector<lower=0, upper=1>[I] tp; //slip (1 - tp)
  vector<lower=0, upper=1>[I] fp; //guess
  real<lower=0, upper=1> lambda1;
  real<lower=0, upper=1> lambda20;
  // real<lower=0, upper=1> lambda21;
  real<lower=0, upper=1> lambda22;
  // real<lower=0, upper=1> lambda30;
  // real<lower=0, upper=1> lambda31;
  // real<lower=0, upper=1> lambda32;
  // real<lower=0, upper=1> lambda4;
  // real<lower=0, upper=1> lambda5;
}
transformed parameters{
  vector[C] raw_nu;
  simplex[C] nu;
  vector[C] theta1;
  vector[C] theta2;
  // vector[C] theta3;
  // vector[C] theta4;
  // vector[C] theta5;
  matrix[I, C] delta;
  matrix[I,C] pi;

  for (c in 1 : C) {
    theta1[c] = (alpha[c, 1] > 0) ? lambda1 : (1 - lambda1);
    // theta2[c] = (alpha[c, 2] > 0 && alpha[c, 1] > 0) ? lambda22 : (alpha[c, 2] > 0 || alpha[c, 1] > 0) ? lambda21 : lambda20;
    // theta3[c] = (alpha[c, 3] > 0 && alpha[c, 2] > 0) ? lambda32 : (alpha[c, 3] > 0 || alpha[c, 2] > 0) ? lambda31 : lambda30;
    
    theta2[c] = (alpha[c, 2] > 0) ? lambda22 : lambda20;
    // theta3[c] = (alpha[c, 3] > 0) ? lambda32 : lambda30;
    // theta4[c] = (alpha[c, 4] > 0) ? lambda4 : (1 - lambda4);
    // theta5[c] = (alpha[c, 5] > 0) ? lambda5 : (1 - lambda5);

    raw_nu[c] = theta1[c] * theta2[c];
    // raw_nu[c] = theta1[c] * theta2[c] * theta3[c] * theta4[c] * theta5[c];
  }

  nu = raw_nu/sum(raw_nu);
  vector[C] log_nu = log(nu);
  
  for(c in 1:C){
    for(i in 1:I){
      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) * pow(1 - theta2[c], Q[i, 2]));
      // * pow(1 - theta4[c], Q[i,4]) * pow(1 - theta5[c], Q[i,5])); 
    }
  }

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow((tp[i]), delta[i,c]) * pow(fp[i], (1 - delta[i,c]));
    }
  }
}
model {
  // Priors
  lambda1 ~ beta(20, 5);
  lambda20 ~ beta(5, 20);
  // lambda21 ~ beta(12.5, 12.5);
  lambda22 ~ beta(20, 5);
  // lambda30 ~ beta(5, 20);
  // lambda31 ~ beta(12.5, 12.5);
  // lambda32 ~ beta(20, 5); 
  // lambda3 ~ beta(20, 5);
  // lambda4 ~ beta(20, 5);
  // lambda5 ~ beta(20, 5);
  
  for (i in 1:I){
    tp[i] ~ beta(1, 1);
    fp[i] ~ beta(1, 1);
  }
}
generated quantities {
  matrix[J,I] y_rep;
  
  for (j in 1:J) {
    int z = categorical_rng(nu);  // sample class for person j
    for (i in 1:I) {
      y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
    }
  }
}