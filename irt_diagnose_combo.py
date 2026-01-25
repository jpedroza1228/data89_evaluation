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
# irt_file = os.path.join(here('stan_models/irt_1pl.stan'))
# irt_file = os.path.join(here('stan_models/irt_2pl.stan'))
irt_file = os.path.join(here('stan_models/irt_3pl.stan'))
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

# irt_diagnose.to_csv(here('diagnostics/1pl_irt_quiz1.csv'))
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
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y.drop(columns = 'name')})

name_mapping = {'y_rep': 'Y'}
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
irtdf.columns.tolist()

# item infit/outfit: > 1.3 --> might need removal, < .7 --> not add new info
# person infit/outfit: > 2 --> responses too unpredictable, < .8 --> too predictable

def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)


ability = irtdf.filter(regex = 'theta')
ability = pd.DataFrame({
  'mean': ability.mean(),
  'std': ability.std(),
  'q_lower': q_lower(ability),
  'q_upper': q_upper(ability)
}).reset_index()

ability['stu'] = ability['index'].str.replace('theta[', '')
ability['stu'] = ability['stu'].str.replace(']', '')
ability = ability.drop(columns = 'index')
ability.head()

pn.ggplot.show(
  pn.ggplot(ability,
            pn.aes('stu',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .3,
                     color = jpcolor)
  + pn.geom_hline(yintercept = 0,
                  color = 'black',
                  linetype = 'dashed')
  + pn.labs(title = 'Ability Parameter for Each Student',
            x = 'Student',
            y = 'Ability')
  + pn.theme(axis_text_x = pn.element_blank())
)

# a = discrimination (differentiate between individuals w/ different ability levels (theta))
# high value = strong discrimination, low value = weak discrimination
# negative = low theta individuals more likely to get responses correct

dis = irtdf.filter(regex = '^a.*]$')
dis = pd.DataFrame({
  'mean': dis.mean(),
  'std': dis.std(),
  'q_lower': q_lower(dis),
  'q_upper': q_upper(dis)
}).reset_index()

dis['item'] = dis['index'].str.replace('a[', '')
dis['item'] = dis['item'].str.replace(']', '')
dis = dis.drop(columns = 'index')
dis.head()

pn.ggplot.show(
  pn.ggplot(dis,
            pn.aes('item',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .3,
                     color = jpcolor)
  # + pn.geom_hline(yintercept = 0,
  #                 color = 'black',
  #                 linetype = 'dashed')
  + pn.labs(title = 'Discrimination Parameter for Each Item',
            x = 'Item',
            y = 'Discrimination')
  + pn.theme(axis_text_x = pn.element_blank())
)

# b = difficulty 
# 0 = average difficulty, 2+ = very hard, -2 = very easy

diff = irtdf.filter(regex = '^b.*]$')
diff = pd.DataFrame({
  'mean': diff.mean(),
  'std': diff.std(),
  'q_lower': q_lower(diff),
  'q_upper': q_upper(diff)
}).reset_index()

diff['item'] = diff['index'].str.replace('b[', '')
diff['item'] = diff['item'].str.replace(']', '')
diff = diff.drop(columns = 'index')
diff.head()

pn.ggplot.show(
  pn.ggplot(diff,
            pn.aes('item',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .3,
                     color = jpcolor)
  # + pn.geom_hline(yintercept = 0,
  #                 color = 'black',
  #                 linetype = 'dashed')
  + pn.labs(title = 'Difficulty Parameter for Each Item',
            x = 'Item',
            y = 'Difficulty')
  + pn.theme(axis_text_x = pn.element_blank())
)

# c = guessing
# higher value = easy to guess
guess = irtdf.filter(regex = '^c.*]$')
guess = pd.DataFrame({
  'mean': guess.mean(),
  'std': guess.std(),
  'q_lower': q_lower(guess),
  'q_upper': q_upper(guess)
}).reset_index()

guess['item'] = guess['index'].str.replace('c[', '')
guess['item'] = guess['item'].str.replace(']', '')
guess = guess.drop(columns = 'index')
guess.head()

pn.ggplot.show(
  pn.ggplot(guess,
            pn.aes('item',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .3,
                     color = jpcolor)
  # + pn.geom_hline(yintercept = 0,
  #                 color = 'black',
  #                 linetype = 'dashed')
  + pn.labs(title = 'Difficulty Parameter for Each Item',
            x = 'Item',
            y = 'Difficulty')
  + pn.theme(axis_text_x = pn.element_blank())
)

yirt = irtdf.filter(regex = '^y_rep')
yirt.head()
y_describe = y.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe.drop(columns = 'index').transpose()

yirt_prob = pd.DataFrame({
  'mean': yirt.mean(),
  'std': yirt.std(),
  'q_lower': q_lower(yirt),
  'q_upper': q_upper(yirt)
}).reset_index()

yirt_prob['index'] = yirt_prob['index'].str.replace('y_rep[', '')
yirt_prob['index'] = yirt_prob['index'].str.replace(']', '')
yirt_prob[['stu', 'item']] = yirt_prob['index'].str.split(pat = ',', expand = True)
yirt_prob = yirt_prob[['stu', 'item', 'mean', 'std', 'q_lower', 'q_upper']]
yirt_prob[['stu', 'item']] = yirt_prob[['stu', 'item']].astype(int)
yirt_prob['correct'] = np.where(yirt_prob['mean'] >= .5, 1, 0)

pn.ggplot.show(
  pn.ggplot(yirt_prob,
            pn.aes('stu',
                   'mean'))
  + pn.geom_point(pn.aes(color = 'factor(item)'),
                  alpha = .5)
  # + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
  #                           ymax = 'q_upper',
  #                           color = 'factor(item)'),
  #                    alpha = .1)
  + pn.geom_hline(yintercept = .5,
                  color = 'black',
                  linetype = 'dashed')
  + pn.labs(title = 'Probability Student Gets Items Correct',
            x = 'Student',
            y = 'Probability')
  + pn.theme(axis_text_x = pn.element_blank())
)

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

pn.ggplot.show(
  pn.ggplot(ppp_irt, pn.aes('item', 'means'))
  + pn.geom_point(color = jpcolor,
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_irt['item'].max() + 1],
                          breaks = np.arange(1, ppp_irt['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)

pn.ggplot.show(
  pn.ggplot(ppp_irt, pn.aes('item', 'stds'))
  + pn.geom_point(color = jpcolor,
                  size = 2)
  + pn.geom_hline(yintercept = .5, linetype = 'dashed')
  + pn.geom_hline(yintercept = .025, linetype = 'dotted')
  + pn.geom_hline(yintercept = .975, linetype = 'dotted')
  + pn.scale_x_continuous(limits = [1, ppp_irt['item'].max() + 1],
                          breaks = np.arange(1, ppp_irt['item'].max() + 1))
  + pn.scale_y_continuous(limits = [0, 1.01],
                          breaks = np.arange(0, 1.01, .1))
  + pn.theme_light()
)







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

dcm_diagnose['R_hat'].sort_values(ascending = False).head()

(
  joblib.dump([dcm_model, dcm_fit],
              here('joblib_models/lcdm_quiz1_modfit.joblib'),
              compress = 3)
)

idcm = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y.filter(regex = 'item')})

idcm = idcm.rename(name_dict = name_mapping, groups = ["posterior_predictive"])


# plotting variables/ppc
az.plot_trace(idcm,
                var_names = 'nu')
plt.show()
plt.clf()

az.plot_trace(idcm,
                var_names = 'pi')
plt.show()
plt.clf()

az.plot_trace(idcm,
                var_names = 'beta0')
plt.show()
plt.clf()

az.plot_trace(idcm,
                var_names = 'beta1')
plt.show()
plt.clf()

az.plot_trace(idcm,
                var_names = 'beta2')
plt.show()
plt.clf()

az.plot_trace(idcm,
                var_names = 'beta12')
plt.show()
plt.clf()

az.plot_forest(idcm.posterior["prob_resp_class"].isel(prob_resp_class_dim_0 = slice(0, 4),
                                                    prob_resp_class_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_class',
               colors = jpcolor)
plt.show()
plt.clf()

az.plot_forest(idcm.posterior["prob_resp_attr"].isel(prob_resp_attr_dim_0 = slice(0, 4),
                                                    prob_resp_attr_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_attr',
               colors = jpcolor)
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


# put draws/samples into pandas dataframe
dcmdf = dcm_fit.draws_pd()
dcmdf.head()


# pi matrix (item x latent attribute mastery (0 | 1))
pi_mat = dcmdf.filter(regex = 'pi')
pi_mat = pd.DataFrame({
  'mean': pi_mat.mean(),
  'std': pi_mat.std(),
  'q_lower': q_lower(pi_mat),
  'q_upper': q_upper(pi_mat)
}).reset_index()

pi_mat['index'] = pi_mat['index'].str.replace('pi[', '')
pi_mat['index'] = pi_mat['index'].str.replace(']', '')
pi_mat[['item', 'lat_class']] = pi_mat['index'].str.split(',', expand = True)
pi_mat[['item', 'lat_class']] = pi_mat[['item', 'lat_class']].astype(int)
pi_mat = pi_mat[['item', 'lat_class', 'mean', 'std', 'q_lower', 'q_upper']]
pi_mat.head()

pn.ggplot.show(
  pn.ggplot(pi_mat,
            pn.aes('factor(item)',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .1)
  + pn.geom_hline(yintercept = .5,
                  color = 'black',
                  linetype = 'dashed')
  + pn.facet_wrap('lat_class')
  + pn.labs(title = 'Probability That Latent Class Gets Items Correct',
            x = 'Item',
            y = 'Probability')
)

pn.ggplot.show(
  pn.ggplot(pi_mat,
            pn.aes('factor(item)',
                   'std'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.facet_wrap('lat_class')
  + pn.labs(title = 'Standard Deviation For Each Item By Latent Class',
            x = 'Item',
            y = 'Standard Deviation')
)

beta_df = dcmdf.filter(regex = 'beta')
beta_df = pd.DataFrame({
  'mean': beta_df.mean(),
  'std': beta_df.std(),
  'q_lower': q_lower(beta_df),
  'q_upper': q_upper(beta_df)
}).reset_index()

beta_df['index'] = beta_df['index'].str.replace(']', '')
beta_df[['var', 'item']] = beta_df['index'].str.split('[', expand = True)
beta_df['item'] = beta_df['item'].astype(int)
beta_df = beta_df[['item', 'var', 'mean', 'std', 'q_lower', 'q_upper']]
beta_df.head()

pn.ggplot.show(
  pn.ggplot(beta_df,
            pn.aes('factor(item)',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .1)
  + pn.facet_wrap('var')
  + pn.labs(title = 'Coefficient Value Per Variable',
            x = 'Item',
            y = 'Coefficient')
)

pn.ggplot.show(
  pn.ggplot(beta_df,
            pn.aes('factor(item)',
                   'std'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.facet_wrap('var')
  + pn.labs(title = 'Standard Deviation For Coefficient Value Per Variable',
            x = 'Item',
            y = 'Standard Deviation')
)


attr_df = dcmdf.filter(regex = '^prob_resp_attr')
attr_df = pd.DataFrame({
  'mean': attr_df.mean(),
  'std': attr_df.std(),
  'q_lower': q_lower(attr_df),
  'q_upper': q_upper(attr_df)
}).reset_index()

attr_df['index'] = attr_df['index'].str.replace('prob_resp_attr[', '')
attr_df['index'] = attr_df['index'].str.replace(']', '')
attr_df[['stu', 'attr']] = attr_df['index'].str.split(',', expand = True)
attr_df[['stu', 'attr']] = attr_df[['stu', 'attr']].astype(int)
attr_df = attr_df[['stu', 'attr', 'mean', 'std', 'q_lower', 'q_upper']]
attr_df.head()

mastery_prob = .8
prof_prob = .5

pn.ggplot.show(
  pn.ggplot(attr_df,
            pn.aes('stu',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .1)
  + pn.geom_hline(yintercept = mastery_prob,
                  linetype = 'dashed')
  + pn.facet_wrap('attr')
  + pn.labs(title = 'Attribute Mastery Per Student',
            x = 'Students',
            y = 'Probability')
)

pn.ggplot.show(
  pn.ggplot(attr_df,
            pn.aes('stu',
                   'std'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.facet_wrap('attr')
  + pn.labs(title = 'Standard Deviation Per Student For Each Attribute',
            x = 'Students',
            y = 'Standard Deviation')
)

attr_df['mastery'] = np.where(attr_df['mean'] > mastery_prob, 1, 0)
attr_df['proficiency'] = np.where(attr_df['mean'] > prof_prob, 1, 0)

attr_df.head()

attr_df.groupby('attr')['mastery'].value_counts().reset_index()
attr_df.groupby('attr')['proficiency'].value_counts().reset_index()

attr_mastery = attr_df.pivot(index = 'stu',
              columns = 'attr',
              values = 'mastery').reset_index()
attr_mastery = attr_mastery.rename(columns = {1: 'attr1',
                                              2: 'attr2'})
attr_mastery.head()

# INCLUDE THE NAMES FOR THE ATTRIBUTES HERE FOR MASTERY
# attr_mastery['attr1'] = np.where(attr_mastery['attr1'] == 1, '', '')
# attr_mastery['attr2'] = np.where(attr_mastery['attr2'] == 1, '', '')
# attr_mastery.to_csv('student_data/attribute_mastery_quiz1.csv')

attr_prof = attr_df.pivot(index = 'stu',
              columns = 'attr',
              values = 'mastery').reset_index()
attr_prof = attr_prof.rename(columns = {1: 'attr1',
                                        2: 'attr2'})
attr_prof.head()

# INCLUDE THE NAMES FOR THE ATTRIBUTES HERE FOR PROFICIENCY
# attr_prof['attr1'] = np.where(attr_prof['attr1'] == 1, '', '')
# attr_prof['attr2'] = np.where(attr_prof['attr2'] == 1, '', '')
# attr_prof.to_csv('student_data/attribute_proficiency_quiz1.csv')

attr_class_df = dcmdf.filter(regex = '^prob_resp_class')
attr_class_df = pd.DataFrame({
  'mean': attr_class_df.mean(),
  'std': attr_class_df.std(),
  'q_lower': q_lower(attr_class_df),
  'q_upper': q_upper(attr_class_df)
}).reset_index()

attr_class_df['index'] = attr_class_df['index'].str.replace('prob_resp_class[', '')
attr_class_df['index'] = attr_class_df['index'].str.replace(']', '')
attr_class_df[['stu', 'lat_class']] = attr_class_df['index'].str.split(',', expand = True)
attr_class_df[['stu', 'lat_class']] = attr_class_df[['stu', 'lat_class']].astype(int)
attr_class_df = attr_class_df[['stu', 'lat_class', 'mean', 'std', 'q_lower', 'q_upper']]
attr_class_df.head()

pn.ggplot.show(
  pn.ggplot(attr_class_df,
            pn.aes('stu',
                   'mean'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper'),
                     alpha = .1)
  + pn.geom_hline(yintercept = .5,
                  color = 'black',
                  linetype = 'dashed')
  + pn.facet_wrap('lat_class')
  + pn.labs(title = 'Probability Per Student In Each Latent Class',
            x = 'Students',
            y = 'Probability')
)

pn.ggplot.show(
  pn.ggplot(attr_class_df,
            pn.aes('stu',
                   'std'))
  + pn.geom_point(alpha = .5,
                  color = jpcolor)
  + pn.facet_wrap('lat_class')
  + pn.labs(title = 'Standard Deviation Per Student For Each Latent Class',
            x = 'Students',
            y = 'Standard Deviation')
)

# PPP Value
# y replicated datasets
ydcm = dcmdf.filter(regex = '^y_rep')

ydcm_prob = pd.DataFrame({
  'mean': ydcm.mean(),
  'std': ydcm.std(),
  'q_lower': q_lower(ydcm),
  'q_upper': q_upper(ydcm)
}).reset_index()

ydcm_prob['index'] = ydcm_prob['index'].str.replace('y_rep[', '')
ydcm_prob['index'] = ydcm_prob['index'].str.replace(']', '')
ydcm_prob[['stu', 'item']] = ydcm_prob['index'].str.split(pat = ',', expand = True)
ydcm_prob = ydcm_prob[['stu', 'item', 'mean', 'std', 'q_lower', 'q_upper']]
ydcm_prob[['stu', 'item']] = ydcm_prob[['stu', 'item']].astype(int)
ydcm_prob['correct'] = np.where(ydcm_prob['mean'] >= .5, 1, 0)


pn.ggplot.show(
  pn.ggplot(ydcm_prob,
            pn.aes('stu',
                   'mean'))
  + pn.geom_point(pn.aes(color = 'factor(item)'),
                  alpha = .5)
  # + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
  #                           ymax = 'q_upper',
  #                           color = 'factor(item)'),
  #                    alpha = .1)
  + pn.geom_hline(yintercept = .5,
                  color = 'black',
                  linetype = 'dashed')
  + pn.labs(title = 'Probability Student Gets Items Correct',
            x = 'Student',
            y = 'Probability')
  + pn.theme(axis_text_x = pn.element_blank())
)


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

# necessary if leading zero on item names
# y_describe.columns = y_describe.columns.str.replace('item0', 'item')

dcm_means = [ppp_func(df = ydcm_prob, item_num = i, stat = 'mean') for i in np.arange(1, (y_describe.shape[1]))]
dcm_stds = [ppp_func(df = ydcm_prob, item_num = i, stat = 'std') for i in np.arange(1, (y_describe.shape[1]))]

ppp_dcm = pd.DataFrame({'means': pd.Series(dcm_means),
                       'stds': pd.Series(dcm_stds)})

ppp_dcm = ppp_dcm.reset_index()
ppp_dcm = ppp_dcm.rename(columns = {'index': 'item'})
ppp_dcm['item'] = ppp_dcm['item'] + 1

ppp_dcm.head()

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







