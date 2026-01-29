import numpy as np
import pandas as pd
import plotnine as pn
from scipy.stats import halfnorm

choice = np.random.binomial(1, .5, 200)
choice2 = np.random.binomial(1, .5, 200)
prior = np.random.beta(1, 1, 200)
prior2 = np.random.beta(1, 1, 200)
beta = halfnorm.rvs(size = 200)
beta02 = np.random.normal(0, 1, 200)
beta2 = halfnorm.rvs(size = 200)

data = pd.DataFrame({'choice1': choice,
                     'choice2': choice2,
                     'lambda1': prior,
                     'lambda2': prior2,
                     'beta1': beta,
                     'beta02': beta02,
                     'beta2': beta2})

# data['lambda1'] = np.where(data['choice1'] == 1, data['lambda1'], (1 - data['lambda1']))
# data['lambda2'] = np.where(data['choice2'] == 1, data['lambda2'], (1 - data['lambda2']))
data.head()

# exp(x)/(1 + exp(x))
data['inv_logit1'] = np.where(data['choice1'] == 1,
                              np.exp(data['lambda1'])/(1 + np.exp(data['lambda1'])),
                              np.exp(1 - data['lambda1'])/(1 + np.exp(1 - data['lambda1']))
)

# this is unnecessary for model
# data['theta1'] = data['choice1'] * np.log(data['inv_logit1']) + (1 - data['choice1']) * np.log(data['inv_logit1'])
# data['theta1_prob'] = np.exp(data['theta1'])

data['inv_logit2'] = np.where(data['choice2'] == 1,
                              np.exp(data['lambda2'] + data['beta2'])/(1 + np.exp(data['lambda2'] + data['beta2'])),
                              np.exp(1 - (data['lambda2'] + data['beta2']))/(1 + np.exp(1 - data['lambda2'] + data['beta2']))
                              )

data['inv_logit2b'] = np.where(data['choice2'] == 1,
                              np.exp(data['beta02'] + data['beta2'])/(1 + np.exp(data['beta02'] + data['beta2'])),
                              np.exp(1 - (data['beta02'] + data['beta2']))/(1 + np.exp(1 - data['beta02'] + data['beta2']))
                              )

from great_tables import GT as gt
gt.show(gt(data.round(2).head(20)))

data['theta2'] = data['choice1'] * np.log(data['inv_logit1']) + (1 - data['choice1']) * np.log(data['inv_logit1'])

