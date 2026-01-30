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

link = 'https://raw.githubusercontent.com/jpedroza1228/projects_portfolio_and_practice/refs/heads/main/projects/dcm/lcdm_py/ecpe_data.csv'

import requests
from io import StringIO

response = requests.get(link, verify = False)
y = pd.read_csv(StringIO(response.text)).clean_names(case_type = 'snake')
# y = pd.read_csv(link)

np.random.seed(12345)
y = y.sample(n = 200)

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

y.head()

# ---------- Assessing Item Difficulty & Discrimination ----------

# stan dictionary data
irt_dict = {
  'J': y.shape[0],
  'I': y.shape[1],
  'Y': np.array(y)
}

# irt_dict

# https://mc-stan.org/learn-stan/case-studies/tutorial_twopl.html
# 2pl IRT Model
# irt_file = os.path.join(here('stan_models/irt_1pl.stan'))
# irt_file = os.path.join(here('stan_models/irt_2pl.stan'))
irt_file = os.path.join(here('stan_models/irt_3pl.stan'))
irt_model = CmdStanModel(stan_file = irt_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
irt_fit = irt_model.sample(data = irt_dict,
                        show_console = True,
                        chains = 4,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

pd.Series(irt_fit.summary()['R_hat'].sort_values(ascending = False)).to_csv(here('stan_models/stan_output/irt3pl_rhat_values.csv'))

(
  joblib.dump([irt_model, irt_fit],
              'stan_models/stan_output/irt_quiz1_fit.joblib',
              compress = 3)
)
