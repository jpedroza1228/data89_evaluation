import numpy as np
import pandas as pd
import plotnine as pn
from janitor import clean_names
from pyhere import here
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import joblib
from scipy import stats
import os
from cmdstanpy import CmdStanModel
from great_tables import GT as gt

os.environ['QT_API'] = 'PyQt6'

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

y = pd.read_csv(here('y.csv')).drop(columns = {'Unnamed: 0'})
q = pd.read_csv(here('q.csv')).drop(columns = {'Unnamed: 0'})

# survey_df = pd.DataFrame({'attend1': np.random.binomial(n = 1,
#                                                         p = .8,
#                                                         size = 200),
#                           'attend2': np.random.binomial(n = 1,
#                                                         p = .8,
#                                                         size = 200),
#                           'complete_hw': np.random.binomial(n = 1,
#                                                             p = .8,
#                                                             size = 200),
#                           'hw_party': np.random.binomial(n = 1,
#                                                             p = .3,
#                                                             size = 200),
#                           'tutor': np.random.binomial(n = 1,
#                                                       p = .1,
#                                                       size = 200),
#                           'attend_discuss': np.random.binomial(n = 1,
#                                                                p = .8,
#                                                                size = 200)})

# attribute mastery matrix
alpha = pd.DataFrame([(x, y) for x in np.arange(2) for y in np.arange(2)])
alpha = alpha.rename(columns = {0: 'hold1',
                                1: 'hold2'})

pn.ggplot.show(
  pn.ggplot(pd.DataFrame({'value': np.random.beta(1, 1, 200)}),
            pn.aes('value'))
  + pn.geom_histogram(color = 'black',
                      fill = 'gray',
                      alpha = .3,
                      bins = 30)
  + pn.theme_minimal()
)

pn.ggplot.show(
  pn.ggplot(pd.DataFrame({'value': np.random.beta(15, 15, 200)}),
            pn.aes('value'))
  + pn.geom_histogram(color = 'black',
                      fill = 'gray',
                      alpha = .3,
                      bins = 30)
  + pn.theme_minimal()
)

# stan dictionary data
stan_dict = {
  'J': y.shape[0],
  'I': y.shape[1],
  'K': q.shape[1],
  'C': alpha.shape[0],
  'Y': np.array(y),
  'Q': np.array(q), 
  'alpha': np.array(alpha)
}

stan_file = os.path.join(here('stan_models/bn_linear.stan'))
stan_model = CmdStanModel(stan_file = stan_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(11826)
stan_fit = stan_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

pd.Series(stan_fit.summary()['R_hat'].sort_values(ascending = False)).head()
# .to_csv(here('stan_models/stan_output/rhat_values/lcdm_rhat_values.csv'))

# (
#   joblib.dump([stan_model, stan_fit],
#               'stan_models/stan_output/bn_linear_quiz1_fit.joblib',
#               compress = 3)
# )

