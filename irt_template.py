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

# ---------- Assessing Item Difficulty & Discrimination ----------

stan_quiz = quiz_df.filter(regex = "^item")

# stan dictionary data
irt_dict = {
  'J': stan_quiz.shape[0],
  'I': stan_quiz.shape[1],
  'Y': np.array(stan_quiz)
}

irt_dict

# https://mc-stan.org/learn-stan/case-studies/tutorial_twopl.html
# 2pl IRT Model
irt_file = os.path.join(here('stan_models/quiz1_2pl_irt_model.stan'))
irt_model = CmdStanModel(stan_file = irt_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(11826)
irt_fit = irt_model.sample(data = irt_dict,
                        show_console = True,
                        chains = 4,
                        iter_warmup = 2000,
                        iter_sampling = 2000)
(
  joblib.dump([irt_model, irt_fit],
              'stan_models/stan_output/irt_quiz1_fit.joblib',
              compress = 3)
)

