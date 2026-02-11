data{
  int<lower=1> J; // Number of students
  int<lower=1> I; // Number of items
  int<lower=1> C; // Number of classes
  int<lower=1> K; // Number of Attributes/Skills
  matrix<lower=0,upper=1> [J,I] Y; // Observed Data [student x item]
  matrix<lower=0,upper=1> [I,K] Q; // Q-matrix [item x attribute]
  matrix<lower=0,upper=1> [C,K] alpha; // Attribute mastery profile [class x attribute]
}
parameters{ //  parameters here are what get sampled/can have priors
  simplex[C] nu; // class proprtions (simplex makes it sum up to be 1)
  vector<lower=0, upper=1>[I] tp; // True positive/Did not slip (Know the answer but got it wrong [1 - tp])
  vector<lower=0, upper=1>[I] fp; // False positive/Guss and got answer right
}
transformed parameters{ //  used for calculations/are derived from calculatinos
  matrix[I,C] delta;  //  global attribute mastery indicator (product of alpha ** Q-matrix) 
  matrix[I,C] pi;     //  probabilities of getting items correct based on latent class

  vector[C] log_nu = log(nu);

  for(c in 1:C){{
    for(i in 1:I){
      delta[i, c] = 1 - (pow(1 - alpha[c,1], Q[i,1]) * pow(1 - alpha[c,2], Q[i,2]) * pow(1 - alpha[c,3], Q[i,3]));  
    }
  }}

  for (c in 1:C){
    for (i in 1:I){
      pi[i,c] = pow(tp[i], delta[i,c]) * pow(fp[i], (1 - delta[i,c]));
    }
  }
} 
model{
  // Prior for class membership
  nu ~ dirichlet(rep_vector(1.0, C)); // uniform prior for the number of classes

  for (i in 1:I){{
    tp[i] ~ beta(20, 5);
    fp[i] ~ beta(5, 20);
  }}
}
generated quantities {
  matrix[J,I] y_rep;
  
  for (j in 1:J) {{
    int z = categorical_rng(nu);  // sample class for person j
    for (i in 1:I) {{
      y_rep[j, i] = bernoulli_rng(pi[i, z]);  // generate response from item-by-class probability
    }}
  }}
}