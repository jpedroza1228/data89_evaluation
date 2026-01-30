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

model, fit = joblib.load(here('stan_models/stan_output/lcdm_quiz1_fit.joblib'))

idata = az.from_cmdstanpy(
    posterior = fit,
    posterior_predictive = ['Y_rep'],
    observed_data = {'Y': y})

name_mapping = {'Y_rep': 'Y'}
idata = idata.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

# diagnostics
az.rhat(idata) # estimate of rank normalized splitR-hat for set of traces
az.bfmi(idata) # estimated bayesian fraction of missing information
az.ess(idata) # estimate of effective sample size
az.mcse(idata) # markov chain standard error statistic
# az.psens(idata) # power-scaling sensitivity diagnostic

# plotting variables/ppc
az.plot_density(idata,
                var_names = 'nu')
plt.show()
plt.clf()

az.plot_density(idata,
                var_names = 'log_nu')
plt.show()
plt.clf()


az.plot_ppc(idata,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
plt.show()
plt.clf()

az.plot_ppc(idata,
            data_pairs = {'Y': 'Y'},
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


az.plot_forest(idata,
               var_names = 'prob_resp_attr',
               colors = 'seagreen')
plt.show()
plt.clf()

# put draws/samples into pandas dataframe
df_fit = fit.draws_pd()
df_fit.head()

df_fit.columns.tolist()

# PPP Value
# y replicated datasets
y_rep = df_fit.filter(regex = '^Y_rep')
y_describe = y.agg(['mean', 'std']).reset_index()

yrep_prob = pd.DataFrame({
  'mean': y_rep.mean(),
  'std': y_rep.std()
  # 'median': y_rep.median()
}).reset_index()

yrep_prob.iloc[0:5, :].head()
yrep_prob[['drop', 'other']] = yrep_prob['index'].str.split(pat = '\\[', expand = True)
yrep_prob[['stu', 'item']] = yrep_prob['other'].str.split(pat = ',', expand = True)
yrep_prob['item'] = yrep_prob['item'].str.replace(']', '')

yrep_prob = yrep_prob[['stu', 'item', 'mean', 'std']]

yrep_prob[['stu', 'item']] = yrep_prob[['stu', 'item']].astype(int)

yrep_prob.iloc[0:5, :].head()

yrep_prob['correct'] = np.where(yrep_prob['mean'] >= .5, 1, 0)

yrep_wide = (
  yrep_prob
  .pivot(
    index = 'stu',
    columns = 'item',
    values = 'correct')
  .reset_index(drop = True)
)

yrep_wide.columns = [f'item{i+1}' for i in np.arange(yrep_wide.shape[1])]
yrep_wide = yrep_wide.reset_index()
yrep_wide = yrep_wide.rename(columns = {'index': 'stu'})
yrep_wide['stu'] = yrep_wide['stu'] + 1

yrep_wide.head()

# posterior predictive p-value (PPP)
yrep_prob.head()
yrep_prob.info()
y_describe.head()

# necessary if leading zero on item names
y_describe.columns = y_describe.columns.str.replace('item0', 'item')   

yrep_prob.loc[yrep_prob['item'] == 1, 'mean']
y_describe.loc[y_describe['index'] == 'mean', 'item1']

def ppp_func(df, item_num, stat):
    thresh = np.array(y_describe.loc[y_describe['index'] == stat, f'item{item_num}'])[0]
    cond = df.loc[df['item'] == item_num, stat] > thresh
    ppp_val = np.where(cond, 1, 0).mean()
    return ppp_val


means = [ppp_func(df = yrep_prob, item_num = i, stat = 'mean') for i in np.arange(1, (y_describe.shape[1]))]
stds = [ppp_func(df = yrep_prob, item_num = i, stat = 'std') for i in np.arange(1, (y_describe.shape[1]))]
# medians = [ppp_func(df = yrep_prob, item_num = i, stat = 'median') for i in np.arange(1, (y_describe.shape[1]))]

ppp_df = pd.DataFrame({'means': pd.Series(means),
                       'stds': pd.Series(stds)})

ppp_df = ppp_df.reset_index()
ppp_df = ppp_df.rename(columns = {'index': 'item'})
ppp_df['item'] = ppp_df['item'] + 1

ppp_df.head()
y_describe


pn.ggplot.show(
  pn.ggplot(ppp_df, pn.aes('item', 'means'))
  + pn.geom_point(color = 'seagreen',
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_df['item'].max() + 1],
                          breaks = np.arange(1, ppp_df['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)

pn.ggplot.show(
  pn.ggplot(ppp_df, pn.aes('item', 'stds'))
  + pn.geom_point(color = 'seagreen',
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_df['item'].max() + 1],
                          breaks = np.arange(1, ppp_df['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)



# reliability (amount of students that mastered each latent attribute)

# # probability of student mastery of each attribute
az.plot_forest(idata, var_names = 'prob_resp_attr', colors = 'seagreen')
plt.show()
plt.clf()

stu_att_mastery = pd.DataFrame({
  'parameters': df_fit.filter(regex = '^prob_resp_attr').columns,
  'mean': df_fit.filter(regex = '^prob_resp_attr').mean().reset_index(drop = True)
})

stu_att_mastery[['drop', 'other']] = stu_att_mastery['parameters'].str.split(pat = '[', expand = True)
stu_att_mastery[['stu', 'att']] = stu_att_mastery['other'].str.split(pat = ',', expand = True)
stu_att_mastery['att'] = stu_att_mastery['att'].str.replace(']', '')

stu_att_mastery = stu_att_mastery.drop(columns = ['parameters', 'drop', 'other'])

stu_att_mastery['stu'] = stu_att_mastery['stu'].astype(int)

stu_att_mastery = (
  stu_att_mastery
  .pivot(index = 'stu', columns = 'att', values = 'mean')
  .reset_index()
  .sort_values(by = 'stu')
)

# can choose what you consider the threshold for mastery
stu_att_mastery['att1_master'] = pd.Series(np.where(stu_att_mastery['1'] >= .8, 1, 0))
stu_att_mastery['att2_master'] = pd.Series(np.where(stu_att_mastery['2'] >= .8, 1, 0))

stu_att_mastery['att1_prof'] = pd.Series(np.where(stu_att_mastery['1'] >= .5, 1, 0))
stu_att_mastery['att2_prof'] = pd.Series(np.where(stu_att_mastery['2'] >= .5, 1, 0))

stu_att_mastery['profile_master'] = (
  stu_att_mastery['att1_master'].astype(str)
  + stu_att_mastery['att2_master'].astype(str)
)

stu_att_mastery['profile_prof'] = (
  stu_att_mastery['att1_prof'].astype(str)
  + stu_att_mastery['att2_prof'].astype(str)
)

stu_att_mastery = stu_att_mastery.rename(columns = {'1': 'att1_prob', '2': 'att2_prob'})

# attribute level probabilities (att\\d) & classifications (bi)
stu_att_mastery.head()

for i in ['att1_master', 'att2_master']:
  print(stu_att_mastery[i].value_counts(normalize = True))

for i in ['att1_prof', 'att2_prof']:
  print(stu_att_mastery[i].value_counts(normalize = True))

stu_att_mastery['profile_master'].value_counts(normalize = True)
stu_att_mastery['profile_prof'].value_counts(normalize = True)

# attribute reliability per skill
att_rel = (
  stu_att_mastery[['att1_prob',
                   'att2_prob']]
  .mean()
  .reset_index()
)

att_rel = att_rel.rename(columns = {0: 'avg_prob'})

att_uncertain_rel = (
  stu_att_mastery[['att1_prob',
                   'att2_prob']]
  .quantile([.025, .975])
  .reset_index()
  .melt(id_vars = 'index', value_vars = ['att1_prob', 'att2_prob'])
  .pivot(index = 'att', columns = 'index', values = 'value')
  .rename(columns = {0.025: 'lower_ci95', 0.975: 'upper_ci95'})
  .reset_index()
)

att_uncertain_rel = att_uncertain_rel.merge(att_rel, how = 'left', on = 'att')

# classification reliability
att_uncertain_rel

pn.ggplot.show(
  pn.ggplot(att_uncertain_rel, pn.aes('att', 'avg_prob'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower_ci95', ymax = 'upper_ci95'), alpha = .3)
  + pn.geom_point(pn.aes(color = 'avg_prob'), size = 4)
  + pn.theme_light()
  + pn.theme(legend_position = None)
)







