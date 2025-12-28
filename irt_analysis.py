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

model, fit = joblib.load(here('stan_models/stan_output/irt_quiz1_fit.joblib'))

idata = az.from_cmdstanpy(
    posterior = irt_fit,
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y})

name_mapping = {'y_rep': 'Y'}
idata = idata.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

df_fit = irt_fit.draws_pd()
df_fit.head()

# low difficulty = easy
# high difficulty = hard
diff = df_fit.filter(regex = '^b')
diff_avg = diff.mean().round(2).reset_index()
diff_avg = diff_avg.rename(columns = {0: 'difficulty'})

pn.ggplot.show(
  pn.ggplot(diff_avg,
            pn.aes('index',
                   'difficulty'))
  + pn.geom_col(color = 'black',
                fill = 'seagreen')
  + pn.coord_flip()
  + pn.theme_minimal()
)

# dis < .5 = poor items
# dis > 4 = extremely good at distinguishing between low and high ability
dis = df_fit.filter(regex = '^a\\[')
dis_avg = dis.mean().round(2).reset_index()
dis_avg = dis_avg.rename(columns = {0: 'discrimination'})

pn.ggplot.show(
  pn.ggplot(dis_avg,
            pn.aes('index',
                   'discrimination'))
  + pn.geom_col(color = 'black',
                fill = 'seagreen')
  + pn.coord_flip()
  + pn.theme_minimal()
)

az.plot_density(idata,
                var_names = 'a')
plt.show()
plt.clf()

az.plot_density(idata,
                var_names = 'b')
plt.show()
plt.clf()

# this prints out a plot for every stu/item combo
# az.plot_forest(idata,
#                var_names = 'eta')
# plt.show()
# plt.clf()

az.plot_ppc(idata,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            alpha = .05)
plt.show()
plt.clf()

az.plot_ppc(idata,
            data_pairs = {'Y': 'Y'},
            alpha = .05,
            num_pp_samples = 1000,
            kind = 'cumulative')
plt.show()
plt.clf()

az.plot_bpv(idata,
            kind = 't_stat', 
            t_stat = 'mean')
plt.show()
plt.clf()

az.plot_bpv(idata,
            kind = 't_stat', 
            t_stat = 'std')
plt.show()
plt.clf()

az.plot_bpv(idata,
            kind = 'p_value')
plt.show()
plt.clf()

az.plot_bpv(idata,
            kind = 'u_value')
plt.show()
plt.clf()