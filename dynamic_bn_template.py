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

os.environ['QT_API'] = 'PyQt6'

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

quiz_df = pd.DataFrame({'stu_id': range(1, 301),
                        'item1': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item2': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item3': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item4': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item5': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item6': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item7': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item8': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item9': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300),
                        'item10': np.random.binomial(n = 1,
                                                    p = .5,
                                                    size = 300)})

survey_df = pd.DataFrame({'attend1': np.random.binomial(n = 1,
                                                        p = .8,
                                                        size = 300),
                          'attend2': np.random.binomial(n = 1,
                                                        p = .8,
                                                        size = 300),
                          'complete_hw': np.random.binomial(n = 1,
                                                            p = .8,
                                                            size = 300),
                          'hw_party': np.random.binomial(n = 1,
                                                            p = .3,
                                                            size = 300),
                          'tutor': np.random.binomial(n = 1,
                                                      p = .3,
                                                      size = 300),
                          'attend_discuss': np.random.binomial(n = 1,
                                                               p = .8,
                                                               size = 300)})

quiz_df.head()



retake_df = pd.DataFrame({'stu_id': pd.Series(np.arange(1, 301)).sample(n = 180),
                        'item1': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item2': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item3': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item4': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item5': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item6': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item7': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item8': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item9': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180),
                        'item10': np.random.binomial(n = 1,
                                                    p = .7,
                                                    size = 180)})
retake_df.head()

# attribute mastery matrix
alpha = pd.DataFrame([(x, y) for x in np.arange(2) for y in np.arange(2)])
alpha = alpha.rename(columns = {0: 'hold1',
                                1: 'hold2'})

q = pd.DataFrame({'hold1': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  'hold2': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})

combo_quiz = quiz_df.merge(retake_df, how = 'inner', on = 'stu_id')

quiz_sub = combo_quiz.filter(regex = '_x$')
quiz_retake = combo_quiz.filter(regex = '_y$')

# pn.ggplot.show(
#   pn.ggplot(pd.DataFrame({'value': np.random.beta(1, 1, 300)}),
#             pn.aes('value'))
#   + pn.geom_histogram(color = 'black',
#                       fill = 'gray',
#                       alpha = .3,
#                       bins = 30)
#   + pn.theme_minimal()
# )

# pn.ggplot.show(
#   pn.ggplot(pd.DataFrame({'value': np.random.beta(15, 15, 300)}),
#             pn.aes('value'))
#   + pn.geom_histogram(color = 'black',
#                       fill = 'gray',
#                       alpha = .3,
#                       bins = 30)
#   + pn.theme_minimal()
# )

# stan dictionary data
stan_dict = {
  'J': quiz_retake.shape[0],
  'I': quiz_retake.shape[1],
  'T': 2, # number of time points (first quiz --> retake)
  'K': q.shape[1],
  'C': alpha.shape[0],
  'Y_t1': np.array(quiz_sub),
  'Y_t2': np.array(quiz_retake),
  'Q': np.array(q), 
  'alpha': np.array(alpha)
}

stan_file = os.path.join(here('stan_models/dynamic_bn.stan'))
stan_model = CmdStanModel(stan_file = stan_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(11826)
stan_fit = stan_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        iter_warmup = 2000,
                        iter_sampling = 2000)
(
  joblib.dump([stan_model, stan_fit],
              'stan_models/stan_output/dynamic_quiz1_fit.joblib',
              compress = 3)
)

