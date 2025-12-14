import numpy as np

N = 300
J = 10
T = 2
K = 2

# Q-matrix
Q = np.random.randint(0,2,(J,K))

# Attribute patterns
alpha = np.array([[0,0],[1,0],[0,1],[1,1]])

# Independent BN
theta1_t1, theta2_t1 = 0.5, 0.5
gamma1_A1, gamma2_A2 = 1.0, 1.0

A = np.zeros((N,K,T),dtype=int)
Y = np.zeros((N,J,T),dtype=int)

# Time 1
A[:,0,0] = np.random.binomial(1,theta1_t1,N)
A[:,1,0] = np.random.binomial(1,theta2_t1,N)
for j in range(J):
    Y[:,j,0] = np.random.binomial(1, 1/(1+np.exp(-(A[:,0,0]*Q[j,0] + A[:,1,0]*Q[j,1])))))

# Time 2 (autoregressive)
pA1_t2 = 1/(1+np.exp(-(np.log(theta1_t1/(1-theta1_t1)) + gamma1_A1*A[:,0,0])))
pA2_t2 = 1/(1+np.exp(-(np.log(theta2_t1/(1-theta2_t1)) + gamma2_A2*A[:,1,0])))
A[:,0,1] = np.random.binomial(1,pA1_t2)
A[:,1,1] = np.random.binomial(1,pA2_t2)
for j in range(J):
    Y[:,j,1] = np.random.binomial(1, 1/(1+np.exp(-(A[:,0,1]*Q[j,0] + A[:,1,1]*Q[j,1])))))
