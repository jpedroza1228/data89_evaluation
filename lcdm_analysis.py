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

y = pd.read_csv(here('ecpe_200_data.csv')).drop(columns = 'Unnamed: 0')
y.head()

q = pd.read_csv(here('ecpe_qmatrix.csv')).drop(columns = 'Unnamed: 0')
q.head()

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


model, fit = joblib.load(here('stan_models/stan_output/lcdm_quiz1_fit.joblib'))

idata = az.from_cmdstanpy(
    posterior = fit,
    posterior_predictive = ['Y_rep'],
    observed_data = {'Y': y})

name_mapping = {'Y_rep': 'Y'}
idata = idata.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

az.plot_density(idata,
                var_names = 'nu')
plt.show()
plt.clf()

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

az.plot_forest(idata,
               var_names = 'prob_resp_class',
               colors = 'seagreen')
plt.show()
plt.clf()

df_fit = fit.draws_pd()
df_fit.head()

df_fit.columns.tolist()

prob_class_df = df_fit.filter(regex = 'prob_resp_class').mean().reset_index()
prob_class_df.head()
prob_class_df[['stu', 'class']] = prob_class_df['index'].str.split(',', expand = True)
prob_class_df[['drop', 'stu']] = prob_class_df['stu'].str.split('\\[', expand = True)
prob_class_df['class'] = prob_class_df['class'].str.replace(']', '')

prob_class_df = prob_class_df[['stu', 'class', 0]].rename(columns = {0: 'prob'})

prob_class_df[['stu', 'class']] = prob_class_df[['stu', 'class']].astype(int)

prob_class_max = (
  prob_class_df
  .groupby('stu')['prob']
  .max()
)

prob_final_class = prob_class_df.merge(prob_class_max).sort_values('stu')
prob_final_class = prob_final_class.round(2)

gt.show(
  gt(
    prob_final_class
  )
)

# alpha
prob_final_class['class'].value_counts(normalize = True)