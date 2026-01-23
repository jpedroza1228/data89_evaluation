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
import plotly.express as px
import plotly.io as pio

jpcolor = 'seagreen'

os.environ['QT_API'] = 'PyQt6'
pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})
pn.theme_set(pn.theme_light())
pio.templates.default = 'simple_white' # 'plotly_white'

contact = pd.read_csv(here('fake_names_emails.csv'))
contact.head()

# correct/incorrect responses to each quiz item
y = pd.read_csv(here('data/quiz/y.csv')).drop(columns = {'Unnamed: 0'})
y['name'] = contact['name']
# y = contact.join(y)

np.random.seed(12345)
y_sub = y.sample(n = 113)

# q-matrix
q = pd.read_csv(here('data/q_matrix/q.csv')).drop(columns = {'Unnamed: 0'})

# attribute mastery matrix
alpha = pd.DataFrame([(x, y) for x in np.arange(2) for y in np.arange(2)])
alpha = alpha.rename(columns = {0: 'hold1',
                                1: 'hold2'})

# stan dictionary data
irt_dict = {
  'J': y_sub.drop(columns = 'name').shape[0],
  'I': y_sub.drop(columns = 'name').shape[1],
  'Y': np.array(y_sub.drop(columns = 'name'))
}

# https://mc-stan.org/learn-stan/case-studies/tutorial_twopl.html
# irt_file = here('stan_models/irt_1pl.stan')
irt_file = os.path.join(here('stan_models/irt_1pl.stan'))
# irt_file = os.path.join(here('stan_models/irt_2pl.stan'))
# irt_file = os.path.join(here('stan_models/irt_3pl.stan'))
irt_model = CmdStanModel(stan_file = irt_file,
                         cpp_options = {'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
irt_fit = irt_model.sample(data = irt_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

irt_diagnose = pd.DataFrame(irt_fit.summary())
irt_diagnose.to_csv(here('diagnostics/1pl_irt_quiz1.csv'))
# irt_diagnose.to_csv(here('diagnostics/2pl_irt_quiz1.csv'))
# irt_diagnose.to_csv(here('diagnostics/3pl_irt_quiz1.csv'))

irt_diagnose['R_hat'].sort_values(ascending = False).head()

(
  joblib.dump([irt_model, irt_fit],
              here('joblib_models/1pl_irt_quiz1_modfit.joblib'),
              compress = 3)
)

iirt = az.from_cmdstanpy(
    posterior = irt_fit,
    posterior_predictive = ['Y_rep'],
    observed_data = {'Y': y.drop(columns = 'name')})

name_mapping = {'Y_rep': 'Y'}
iirt = iirt.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

# plotting variables/ppc
az.plot_density(iirt,
                var_names = 'beta')
plt.show()
plt.clf()

az.plot_ppc(iirt,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
plt.show()
plt.clf()

az.plot_ppc(iirt,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            kind = 'cumulative')
plt.show()
plt.clf()

az.plot_bpv(iirt,
            kind = 't_stat', 
            t_stat = 'mean')
plt.show()
plt.clf()

az.plot_bpv(iirt,
            kind = 't_stat', 
            t_stat = 'std')
plt.show()
plt.clf()

az.plot_forest(iirt,
               var_names = 'prob_correct',
               colors = jpcolor)
plt.show()
plt.clf()

irtdf = irt_fit.draws_pd()
irtdf.head()
# irtdf.columns.tolist()

ability = irtdf.filter(regex = 'theta').mean().reset_index()
ability = ability.rename(columns = {0: 'ability'})
ability['index'] = ability['index'].str.replace('theta[', '')
ability['index'] = ability['index'].str.replace(']', '')

pn.ggplot.show(
  pn.ggplot(ability,
            pn.aes('index',
                   'ability'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_hline(yintercept = 0,
                  color = 'black')
  + pn.labs(title = 'Ability Parameter for Each Student',
            x = 'Student',
            y = 'Ability')
  + pn.theme(axis_text_x = pn.element_blank())
)

diff = irtdf.filter(regex = '^b.*]$').mean().reset_index()
diff = diff.rename(columns = {0: 'difficulty'})
diff['index'] = diff['index'].str.replace('b[', '')
diff['index'] = diff['index'].str.replace(']', '')

pn.ggplot.show(
  pn.ggplot(diff,
            pn.aes('index',
                   'difficulty'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_hline(yintercept = 0,
                  color = 'black')
  + pn.labs(title = 'Difficulty Parameter for Each Item',
            x = 'Student',
            y = 'Difficulty')
  + pn.theme(axis_text_x = pn.element_blank())
)

yirt = irtdf.filter(regex = '^Y_rep')
yirt.head()
y_describe = y.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe.drop(columns = 'index').transpose()

yirt_prob = pd.DataFrame({
  'mean': yirt.mean(),
  'std': yirt.std()
}).reset_index()

yirt_prob.iloc[0:5, :].head()
yirt_prob[['drop', 'other']] = yirt_prob['index'].str.split(pat = '\\[', expand = True)
yirt_prob[['stu', 'item']] = yirt_prob['other'].str.split(pat = ',', expand = True)
yirt_prob['item'] = yirt_prob['item'].str.replace(']', '')
yirt_prob = yirt_prob[['stu', 'item', 'mean', 'std']]
yirt_prob[['stu', 'item']] = yirt_prob[['stu', 'item']].astype(int)
yirt_prob['correct'] = np.where(yirt_prob['mean'] >= .5, 1, 0)

yirt_prob.iloc[0:5, :].head()

yirt_wide = (
  yirt_prob
  .pivot(
    index = 'stu',
    columns = 'item',
    values = 'correct')
  .reset_index(drop = True)
)

yirt_wide.columns = [f'item{i+1}' for i in np.arange(yirt_wide.shape[1])]
yirt_wide = yirt_wide.reset_index()
yirt_wide = yirt_wide.rename(columns = {'index': 'stu'})
yirt_wide['stu'] = yirt_wide['stu'] + 1
yirt_wide.head()

# necessary if leading zero on item names
# y_describe.columns = y_describe.columns.str.replace('item0', 'item')

yirt_prob.loc[yirt_prob['item'] == 1, 'mean']
y_describe.loc[y_describe['index'] == 'mean', 'item1']

def ppp_func(df, item_num, stat):
    thresh = np.array(y_describe.loc[y_describe['index'] == stat, f'item{item_num}'])[0]
    cond = df.loc[df['item'] == item_num, stat] > thresh
    ppp_val = np.where(cond, 1, 0).mean()
    return ppp_val


irt_means = [ppp_func(df = yirt_prob, item_num = i, stat = 'mean') for i in np.arange(1, (y_describe.shape[1]))]
irt_stds = [ppp_func(df = yirt_prob, item_num = i, stat = 'std') for i in np.arange(1, (y_describe.shape[1]))]

ppp_irt = pd.DataFrame({'means': pd.Series(irt_means),
                       'stds': pd.Series(irt_stds)})

ppp_irt = ppp_irt.reset_index()
ppp_irt = ppp_irt.rename(columns = {'index': 'item'})
ppp_irt['item'] = ppp_irt['item'] + 1

ppp_irt.head()
y_describe


# diagnostic model
stan_dict = {
  'J': y_sub.drop(columns = 'name').shape[0],
  'I': y_sub.drop(columns = 'name').shape[1],
  'C': alpha.shape[0],
  'K': q.shape[1],
  'Y': np.array(y_sub.drop(columns = 'name')),
  'Q': np.array(q),
  'alpha': np.array(alpha)
}

dcm_file = os.path.join(here('stan_models/lcdm.stan'))
dcm_model = CmdStanModel(stan_file = dcm_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_fit = dcm_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

dcm_diagnose = pd.DataFrame(dcm_fit.summary())
# dcm_diagnose.to_csv(here('diagnostics/lcdm_quiz1.csv'))

irt_diagnose['R_hat'].sort_values(ascending = False).head()

(
  joblib.dump([dcm_model, dcm_fit],
              here('joblib_models/lcdm_quiz1_modfit.joblib'),
              compress = 3)
)

idcm = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['Y_rep'],
    observed_data = {'Y': y.filter(regex = 'item')})

idcm = idcm.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

# diagnostics
# az.rhat(idcm) # estimate of rank normalized splitR-hat for set of traces
# az.bfmi(idcm) # estimated bayesian fraction of missing information
# az.ess(idcm) # estimate of effective sample size
# az.mcse(idcm) # markov chain standard error statistic
# az.psens(idcm) # power-scaling sensitivity diagnostic

# plotting variables/ppc
az.plot_density(idcm,
                var_names = 'nu')
plt.show()
plt.clf()

az.plot_density(idcm,
                var_names = 'pi')
plt.show()
plt.clf()


az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
plt.show()
plt.clf()

az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            kind = 'cumulative')
plt.show()
plt.clf()

az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'mean')
plt.show()
plt.clf()

az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'std')
plt.show()
plt.clf()

az.plot_forest(idcm,
               var_names = 'prob_resp_attr',
               colors = jpcolor)
plt.show()
plt.clf()


# put draws/samples into pandas dataframe
dcm_fit = dcm_fit.draws_pd()
dcm_fit.head()


# pi matrix (item x latent attribute mastery (0 | 1))
pidf = dcm_fit.filter(regex = 'pi')
pidf.loc[:, ['pi[1,1]', 'pi[1,2]', 'pi[1,3]', 'pi[1,4]']].mean()

pidf_prob = pd.DataFrame({
  'mean': pidf.mean(),
  'std': pidf.std()
}).reset_index()

pidf = pidf_prob.melt(id_vars = 'index', value_vars = ['mean', 'std'])
pidf[['item', 'lat_class']] = pidf['index'].str.split(',', expand = True)
pidf['item'] = pidf['item'].str.replace('pi[', '').astype(int)
pidf['lat_class'] = pidf['lat_class'].str.replace(']', '').astype(int)
pidf = pidf[['item', 'lat_class', 'variable', 'value']]
pidf.head()

pidf_calc = pidf.groupby(['item', 'lat_class', 'variable'])['value'].mean().round(2).reset_index()

pidf_calc['lat_class'] = pd.Categorical(pidf_calc['lat_class'])
pidf_calc['item'] = pd.Categorical(pidf_calc['item'])

px.bar(pidf_calc.loc[pidf_calc['variable'] == 'mean'],
           x = 'item',
           y = 'value',
           color = 'lat_class',
           barmode = 'group',
           title = 'Pi matrix: [items, classes]')


pn.ggplot.show(
  pn.ggplot(pidf_calc,
            pn.aes('item',
                   'value'))
  + pn.geom_point(pn.aes(color = 'factor(lat_class)'),
                  position = pn.position_jitter())
  + pn.scale_x_continuous(limits = [1, 10],
                          breaks = np.arange(1, 11, 1))
  + pn.facet_wrap('variable')
)



attrdf = dcm_fit.filter(regex = '^prob_resp_attr')

attr_prob = pd.DataFrame({
  'mean': attrdf.mean(),
  'std': attrdf.std()
}).reset_index()

attrdf = attr_prob.melt(id_vars = 'index', value_vars = ['mean', 'std'])
attrdf[['stu', 'attr']] = attrdf['index'].str.split(',', expand = True)
attrdf['stu'] = attrdf['stu'].str.replace('prob_resp_attr[', '').astype(int)
attrdf['attr'] = attrdf['attr'].str.replace(']', '').astype(int)
attrdf = attrdf[['stu', 'attr', 'variable', 'value']]

mastery_prob = .8
attrdf_bi = attrdf.loc[attrdf['variable'] == 'mean', ['stu', 'attr', 'value']]
# attrdf_bi['prof'] = np.where(attrdf_bi['value'] >= .5, 1, 0)
attrdf_bi['master'] = np.where(attrdf_bi['value'] >= mastery_prob, 1, 0)
attrdf_bi.head()

# attrdf_bi.groupby('attr')['prof'].value_counts().reset_index()
attrdf_bi.groupby('attr')['master'].value_counts().reset_index()

attr_calc = attrdf.groupby(['stu', 'attr', 'variable'])['value'].mean().round(2).reset_index()
attr_calc['attr'] = pd.Categorical(attr_calc['attr'], ordered = True)

pn.ggplot.show(
  pn.ggplot(attr_calc.loc[attr_calc['variable'] == 'mean'],
            pn.aes('stu',
                   'value'))
  + pn.geom_point(color = jpcolor)
  + pn.geom_hline(yintercept = mastery_prob,
                  color = 'red',
                  linetype = 'dotted')
  + pn.scale_y_continuous(limits = [0, 1],
                          breaks = np.arange(0, 1.1, .1))
  + pn.facet_wrap('attr')
)


stu_att_mastery = pd.DataFrame({
  'parameters': dcm_fit.filter(regex = '^prob_resp_attr').columns,
  'mean': dcm_fit.filter(regex = '^prob_resp_attr').mean().reset_index(drop = True)
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

# # can choose what you consider the threshold for mastery
stu_att_mastery['att1_master'] = pd.Series(np.where(stu_att_mastery['1'] >= mastery_prob, 1, 0))
stu_att_mastery['att2_master'] = pd.Series(np.where(stu_att_mastery['2'] >= mastery_prob, 1, 0))

# stu_att_mastery['att1_prof'] = pd.Series(np.where(stu_att_mastery['1'] >= .5, 1, 0))
# stu_att_mastery['att2_prof'] = pd.Series(np.where(stu_att_mastery['2'] >= .5, 1, 0))

stu_att_mastery['profile_master'] = (
  stu_att_mastery['att1_master'].astype(str)
  + stu_att_mastery['att2_master'].astype(str)
)

# stu_att_mastery['profile_prof'] = (
#   stu_att_mastery['att1_prof'].astype(str)
#   + stu_att_mastery['att2_prof'].astype(str)
# )

stu_att_mastery = stu_att_mastery.rename(columns = {'1': 'att1_prob', '2': 'att2_prob'})

# # attribute level probabilities (att\\d) & classifications (bi)
stu_att_mastery.head()

# for i in ['att1_master', 'att2_master']:
#   print(stu_att_mastery[i].value_counts(normalize = True))

# for i in ['att1_prof', 'att2_prof']:
#   print(stu_att_mastery[i].value_counts(normalize = True))

stu_att_mastery['profile_master'].value_counts(normalize = True)
# stu_att_mastery['profile_prof'].value_counts(normalize = True)

# # attribute reliability per skill
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

# # classification reliability
att_uncertain_rel

pn.ggplot.show(
  pn.ggplot(att_uncertain_rel,
            pn.aes('att', 'avg_prob'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower_ci95',
                            ymax = 'upper_ci95'),
                     color = jpcolor,
                     alpha = .7,
                     size = 1)
  + pn.geom_point(color = jpcolor,
                  size = 3)
  + pn.scale_y_continuous(limits = [0, 1])
  + pn.theme_light()
  + pn.theme(legend_position = None)
)










# PPP Value
# y replicated datasets
ydcm = dcm_fit.filter(regex = '^Y_rep')

ydcm_prob = pd.DataFrame({
  'mean': ydcm.mean(),
  'std': ydcm.std()
}).reset_index()

ydcm_prob.iloc[0:5, :].head()
ydcm_prob[['drop', 'other']] = ydcm_prob['index'].str.split(pat = '\\[', expand = True)
ydcm_prob[['stu', 'item']] = ydcm_prob['other'].str.split(pat = ',', expand = True)
ydcm_prob['item'] = ydcm_prob['item'].str.replace(']', '')
ydcm_prob = ydcm_prob[['stu', 'item', 'mean', 'std']]
ydcm_prob[['stu', 'item']] = ydcm_prob[['stu', 'item']].astype(int)
ydcm_prob['correct'] = np.where(ydcm_prob['mean'] >= .5, 1, 0)
ydcm_prob.iloc[0:5, :].head()

ydcm_wide = (
  ydcm_prob
  .pivot(
    index = 'stu',
    columns = 'item',
    values = 'correct')
  .reset_index(drop = True)
)

ydcm_wide.columns = [f'item{i+1}' for i in np.arange(ydcm_wide.shape[1])]
ydcm_wide = ydcm_wide.reset_index()
ydcm_wide = ydcm_wide.rename(columns = {'index': 'stu'})
ydcm_wide['stu'] = ydcm_wide['stu'] + 1

ydcm_wide.head()

# posterior predictive p-value (PPP)
ydcm_prob.head()
y_describe.head()

ydcm_prob.loc[ydcm_prob['item'] == 1, 'mean']
y_describe.loc[y_describe['index'] == 'mean', 'item1']

dcm_means = [ppp_func(df = ydcm_prob, item_num = i, stat = 'mean') for i in np.arange(1, (y_describe.shape[1]))]
dcm_stds = [ppp_func(df = ydcm_prob, item_num = i, stat = 'std') for i in np.arange(1, (y_describe.shape[1]))]

ppp_dcm = pd.DataFrame({'means': pd.Series(dcm_means),
                       'stds': pd.Series(dcm_stds)})

ppp_dcm = ppp_dcm.reset_index()
ppp_dcm = ppp_dcm.rename(columns = {'index': 'item'})
ppp_dcm['item'] = ppp_dcm['item'] + 1

ppp_dcm.head() # proportion over the .5 threshold
y_describe

pn.ggplot.show(
  pn.ggplot(ppp_dcm, pn.aes('item', 'means'))
  + pn.geom_point(color = jpcolor,
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_dcm['item'].max() + 1],
                          breaks = np.arange(1, ppp_dcm['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)

pn.ggplot.show(
  pn.ggplot(ppp_dcm, pn.aes('item', 'stds'))
  + pn.geom_point(color = jpcolor,
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_dcm['item'].max() + 1],
                          breaks = np.arange(1, ppp_dcm['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)

