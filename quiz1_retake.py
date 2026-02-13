import pandas as pd
import numpy as np
from scipy import stats
from janitor import clean_names
from pyhere import here
import os
import plotnine as pn
import matplotlib
import matplotlib.pyplot as plt
import arviz as az
import joblib
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
# pio.templates.default = 'simple_white' # 'plotly_white'

def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)

def acceptable_fit_stat(inference_data, func_name = ['waic', 'loo']):
  if func_name == 'waic':
    est = np.abs(az.waic(inference_data).iloc[0])
    se = az.waic(inference_data).iloc[1]
    
    if est > se * 2.5:
      print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
    else:
      print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')
  elif func_name == 'loo':
    est = np.abs(az.loo(inference_data).iloc[0])
    se = az.loo(inference_data).iloc[1]
    
    if est > se * 2.5:
      print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
    else:
      print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')
      

# attribute mastery matrix
alpha = pd.DataFrame([(a, b, c) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2)])
alpha = alpha.rename(columns = {0: 'Rules, Logic, Sets, & Bounds',
                                1: 'Computing',
                                2: 'Relating Joint, Conditional, Maginal'}).clean_names(case_type = 'snake')
alpha.head()

y = pd.read_csv(here('data/quiz_data/q1_scores_anonymized.csv'))
y.head()

y2 = pd.read_csv(here('data/quiz_data/q1_retake_scores_anonymized.csv'))
y2.head()

y2 = y2.rename(columns = {
  'quiz-1-retake-3x3-events (%)': 'item4', #joint distributions/relating
  'quiz-1-retake-balls-and-boxes (%)': 'item2', #balls in bins/computing
  'quiz-1-retake-daily-rain (%)': 'item5', # bounds/rules_logic_sets_bounds
  'quiz-1-retake-independence (%)': 'item6', # axioms/rules_logic_sets_bounds
  'quiz-1-retake-medical-test (%)': 'item7', #bayes rules/relating
  'quiz-1-retake-probability-spaces (%)': 'item1', # /rules_logic_sets_bounds
  'quiz-1-retake-true-statements (%)': 'item3' # /rules_logic_sets_bounds
})

y.columns = ['anon_id', 'item1', 'item2', 'item3a', 'item3b', 'item4', 'item5', 'item6', 'item7', 'score']
y['item3'] = y['item3a'].astype(str) + y['item3b'].astype(str)
y['item3'] = y['item3'].str.replace('nan', '')
y['item3'] = y['item3'].astype(float)
y = y[['anon_id', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']]

y.head()

y2 = y2[['anon_id', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']]

only_retake = y2[~y2['anon_id'].isin(y['anon_id'])]
# 8 students did not take Quiz 1 and only took Quiz 2
only_retake = pd.DataFrame({i: np.where(only_retake[i] == 100, 1, 0) for i in only_retake.columns})
only_retake.describe().transpose()[['mean', 'std']]

# students who took quiz and retake
y_sub = y[y['anon_id'].isin(y2['anon_id'])]
y2_sub = y2[y2['anon_id'].isin(y_sub['anon_id'])]

y_item = y_sub.drop(columns = 'anon_id')
y_item = pd.DataFrame({i: np.where(y_item[i] == 100, 1, 0) for i in y_item.columns})
y_item.head()

y2_item = y2_sub.drop(columns = 'anon_id')
y2_item = pd.DataFrame({i: np.where(y2_item[i] == 100, 1, 0) for i in y2_item.columns})
y2_item.head()

#q-matrix
q = pd.read_csv(here('data/q_matrix/q1_7item_3att_slack.csv')).clean_names(case_type = 'snake')
q.columns = ['row', 'attr1', 'attr2', 'attr3']
q = q.drop(columns = 'row')
q

name_mapping = {'y_rep': 'Y'}

# ex = np.random.normal(0, 2, 100)
# ex_prob = np.exp(ex)/(1 + np.exp(ex))

# ex_prob.mean()
# ex_prob.min()
# ex_prob.max()

# ex2 = np.random.beta(12.5, 12.5, 100)

# ex2.mean()
# ex2.min()
# ex2.max()


stan_dict = {
  'J': y_item.shape[0],
  'I': y_item.shape[1],
  'T': 2,
  'K': q.shape[1],
  'C': alpha.shape[0],
  'Y_t1': np.array(y_item),
  'Y_t2': np.array(y2_item),
  'Q': np.array(q),
  'alpha': np.array(alpha)
}

dcm_file = os.path.join(here(f'quiz_models/quiz1_retake_model_attr3.stan'))
dcm_model = CmdStanModel(stan_file = dcm_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_fit = dcm_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .90,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

dcm_diagnose = pd.DataFrame(dcm_fit.summary())

print(dcm_diagnose['R_hat'].sort_values(ascending = False).head())

dcm_diagnose.to_csv(here(f'diagnostics/quiz1_retake_attr3.csv'))
(
  joblib.dump([dcm_model, dcm_fit],
              here(f'joblib_models/quiz1_retake_attr3_modfit.joblib'),
              compress = 3)
)

# FIT BELOW FOR PRIOR ONLY MODEL
dcm_file = os.path.join(here(f'quiz_models/quiz1_retake_model_attr3.stan'))
dcm_model = CmdStanModel(stan_file = dcm_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_fit = dcm_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .90,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

dcm_diagnose = pd.DataFrame(dcm_fit.summary())




dcmdf = dcm_fit.draws_pd()

dcmdf.filter(regex = 'nu')


idcm = az.from_cmdstanpy(
    posterior = dino[1],
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y_item.filter(regex = 'item')},
    log_likelihood = {'Y': 'eta'}
    )

idcm = idcm.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

idcm_prior = az.from_cmdstanpy(prior = dino_prior[1],
prior_predictive = ['y_rep'])

idcm_prior = idcm_prior.rename(
    name_dict = name_mapping,
    groups = ['prior_predictive']
)

idcm.extend(idcm_prior)

# Plots
az.plot_forest(idcm,
               var_names = 'nu',
               colors = jpcolor)
plt.show()
plt.clf()

az.plot_forest(idcm,
               var_names = 'pi',
               colors = jpcolor)
plt.show()
plt.clf()

az.plot_forest(idcm.posterior["prob_resp_class"].isel(prob_resp_class_dim_0 = slice(0, 4),
                                                      prob_resp_class_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_class',
               colors = jpcolor)
plt.show()
plt.clf()

az.loo(idcm)
az.waic(idcm)

acceptable_fit_stat(inference_data = idcm, func_name = 'waic')
acceptable_fit_stat(inference_data = idcm, func_name = 'loo')


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

# Slipping/Guessing
slip_guess = dinodf.filter(regex = 'tp|fp')

slip_guess = pd.DataFrame({
  'mean': slip_guess.mean(),
  'std': slip_guess.std(),
  'q_lower': q_lower(slip_guess),
  'q_upper': q_upper(slip_guess)
}).reset_index()

slip_guess['index'] = slip_guess['index'].str.replace('[', '')
slip_guess['index'] = slip_guess['index'].str.replace(']', '')
slip_guess['type'] = slip_guess['index'].str.slice(start = 0, stop = 2)
slip_guess['item'] = slip_guess['index'].str.slice(start = 2) 

pn.ggplot.show(
  pn.ggplot(slip_guess,
    pn.aes('factor(item)', 'mean'))
  + pn.geom_point(pn.aes(color = 'type'),
                  alpha = .7)
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower',
                            ymax = 'q_upper',
                            color = 'factor(type)'),
                     alpha = .7)
  + pn.facet_wrap('type')
  + pn.scale_color_brewer('qual', 'Set1')
  + pn.labs(title = 'Probability Guessing/Slipping',
            x = 'Item',
            y = 'Probability',
            caption = 'tp = No slipping. Actually got answer correct.\nfp = Guessed and got answer correct')
)

# Pi Matrix 
pidf = dinodf.filter(regex = 'pi').reset_index()
pidf = pidf.rename(columns = {'index': 'draw'})
pilong = pidf.melt(id_vars = 'draw')
pilong['variable'] = pilong['variable'].str.replace('pi[', '')
pilong['variable'] = pilong['variable'].str.replace(']', '')
pilong[['item', 'latclass']] = pilong['variable'].str.split(',', expand = True)
pilong = pilong[['draw', 'item', 'latclass', 'value']]
pilong[['draw', 'item', 'latclass']] = pilong[['draw', 'item', 'latclass']].astype(int)

pn.ggplot.show(
  pn.ggplot(pilong,
            pn.aes('item',
                   'value'))
  + pn.geom_point(alpha = .3,
                  color = jpcolor)
  + pn.facet_wrap('latclass')
  + pn.scale_x_continuous(limits = [1, 7],
                          breaks = [1, 2, 3, 4, 5, 6, 7])
  + pn.theme(legend_position = 'none')
)

pilong_avg = pilong.groupby(['item', 'latclass'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

pn.ggplot.show(
  pn.ggplot(pilong_avg,
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.scale_x_continuous(limits = [1, 7],
                          breaks = [1, 2, 3, 4, 5, 6, 7])
  + pn.facet_wrap('latclass')
  + pn.theme(legend_position = 'none')
)

# Latent Class Averages
pilong.groupby('latclass')['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index().round(2)


# Students Probability Belonging to Latent Classes
attr_class = dinodf.filter(regex = '^prob_resp_class').reset_index()
attr_class = attr_class.rename(columns = {'index': 'draw'})
class_long = attr_class.melt(id_vars = 'draw')

class_long['variable'] = class_long['variable'].str.replace('prob_resp_class[', '')
class_long['variable'] = class_long['variable'].str.replace(']', '')
class_long[['stu', 'latclass']] = class_long['variable'].str.split(',', expand = True)
class_long[['draw', 'stu', 'latclass']] = class_long[['draw', 'stu', 'latclass']].astype(int)
class_long = class_long[['draw', 'stu', 'latclass', 'value']]

class_avg = class_long.groupby(['stu', 'latclass'])['value'].mean().reset_index()

class_stu_max = class_avg.groupby('stu')['value'].max().reset_index()

class_max = class_avg.merge(class_stu_max, 'inner')

pi


# 2. Merge with your raw response data (assuming 'raw_df' has columns: stu, item, score)
# Replace 'raw_df' with your actual dataframe name
obs_data = raw_df.merge(stu_class_map, on='stu')

# 3. Calculate the Observed Proportion (T_obs) per item per class
obs_stats = obs_data.groupby(['item', 'assigned_class'])['score'].mean().reset_index()
obs_stats.rename(columns={'score': 'obs_mean', 'assigned_class': 'latclass'}, inplace=True)

# 1. Merge predictions (pi) with observed stats
# This aligns every MCMC draw of pi_jc with the actual observed proportion for that class/item
ppp_df = pilong.merge(obs_stats, on=['item', 'latclass'])

# 2. For each draw, check if the model's estimate is greater than or equal to the observed mean
ppp_df['is_greater'] = ppp_df['value'] >= ppp_df['obs_mean']

# 3. The PPP is the mean of this boolean check per item/class
ppp_results = ppp_df.groupby(['item', 'latclass']).agg(
    model_mean=('value', 'mean'),
    obs_mean=('obs_mean', 'first'),
    ppp=('is_greater', 'mean')
).reset_index()

print(ppp_results)

(
    pn.ggplot(ppp_results, pn.aes(x='item', y='model_mean'))
    + pn.geom_point(color=jpcolor, size=3)
    # Add the observed X
    + pn.geom_point(pn.aes(y='obs_mean'), color='red', shape='x', size=4)
    # Add the PPP value as text
    + pn.geom_text(pn.aes(label='ppp.round(2)'), va='bottom', size=8, nudge_y=0.05)
    + pn.facet_wrap('latclass')
    + pn.labs(title="PPC: Model vs. Observed (X)", 
              subtitle="Numbers indicate PPP values (aim for ~0.50)",
              y="Probability of Correct Response")
    + pn.theme_minimal()
)






# Attribute Mastery
attr_df = dinodf.filter(regex = '^prob_resp_attr').reset_index()
attr_df = attr_df.rename(columns = {'index': 'draw'})
attr_long = attr_df.melt(id_vars = 'draw')

attr_long['variable'] = attr_long['variable'].str.replace('prob_resp_attr[', '')
attr_long['variable'] = attr_long['variable'].str.replace(']', '')
attr_long[['stu', 'attr']] = attr_long['variable'].str.split(',', expand = True)
attr_long[['draw', 'stu', 'attr']] = attr_long[['draw', 'stu', 'attr']].astype(int)
attr_long = attr_long[['draw', 'stu', 'attr', 'value']]

attr_avg = attr_long.groupby(['stu', 'attr'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

pn.ggplot.show(
  pn.ggplot(attr_avg,
            pn.aes('stu',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor,
                     alpha = .1)
  + pn.geom_point(alpha = .3,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .8,
                  color = 'black',
                  linetype = 'dashed')
  + pn.facet_wrap('attr')
  + pn.theme(legend_position = 'none',
             axis_text_x = pn.element_blank())
)


attr_avg['acc_comp'] = attr_avg['mean'].apply(lambda p: max(p, 1 - p))
attr_avg['cons_comp'] = attr_avg['mean'].apply(lambda p: p**2 + (1 - p)**2)

reliability_metrics = attr_avg.groupby('attr').agg(
    accuracy=('acc_comp', 'mean'),
    consistency=('cons_comp', 'mean')
).reset_index()

reliability_metrics


# Y-replicated Data
# PPP
ydcm = dinodf.filter(regex = '^y_rep')

# calculations for odds ratios/conditional probabilities
ydcm_long = ydcm.melt()

ydcm_long['variable'] = ydcm_long['variable'].str.replace('y_rep[', '')
ydcm_long['variable'] = ydcm_long['variable'].str.replace(']', '')
ydcm_long[['stu', 'item']] = ydcm_long['variable'].str.split(',', expand = True)
ydcm_long = ydcm_long[['stu', 'item', 'value']]
ydcm_long[['stu', 'item']] = ydcm_long[['stu', 'item']].astype(int)

# ydcm_long_count = ydcm_long.groupby('item')['value'].value_counts().reset_index()

ydcm_long['draw'] = ydcm_long.groupby(['stu', 'item']).cumcount()

ydcm_wide = ydcm_long.pivot(index = ['stu', 'draw'], columns = 'item', values = 'value')
ydcm_wide = ydcm_wide.reset_index()
ydcm_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']

ydcm_wide['total'] = ydcm_wide.filter(regex = 'item').sum(axis = 1)
ydcm_wide_count = ydcm_wide.groupby('draw')['total'].value_counts().reset_index()

pn.ggplot.show(
  pn.ggplot(ydcm_wide_count,
            pn.aes('total',
                   'count'))
  + pn.geom_point(alpha = .1,
                  color = jpcolor,
                  position = pn.position_jitter())
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
)

# Calculate mean, 2.5th percentile, and 97.5th percentile
ydcm_scores = ydcm_wide_count.groupby('total')['count'].agg(
    count = 'mean',
    lower = q_lower,
    upper = q_upper
).reset_index()

ydcm_wide_count['type'] = 'draw_counts'
ydcm_scores['type'] = 'avg_counts'

ydcm_wide_count['count'] = ydcm_wide_count['count'].astype(float)
ydcm_wide_count = ydcm_wide_count.merge(ydcm_scores, 'outer')

y_item['total'] = y_item.sum(axis = 1)
y_item_count = y_item['total'].value_counts().reset_index()
y_item_count['type'] = 'actual_counts'
y_item_count['count'] = y_item_count['count'].astype(float)

ydcm_wide_count = ydcm_wide_count.merge(y_item_count, 'outer')

pn.ggplot.show(
  pn.ggplot(ydcm_wide_count.loc[ydcm_wide_count['type'] != 'draw_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_point(pn.aes(color = 'type'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower',
                            ymax = 'upper'))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
)

pn.ggplot.show(
  pn.ggplot(ydcm_wide_count.loc[ydcm_wide_count['type'] != 'avg_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_point(pn.aes(color = 'type'))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
  + pn.facet_wrap('type')
)


# OVERALL PPP VALUES

y_item_count = y_item_count.sort_values('total')
ydcm_scores = ydcm_scores.sort_values('total')

chi2_obs = np.sum(((y_item_count['count'] - ydcm_scores['count'])**2) / (ydcm_scores['count'] + 1e-9))

draw_count = ydcm_wide_count.loc[ydcm_wide_count['type'] == 'draw_counts']

chi2_rep_list = []

for draw_id, group in draw_count.groupby('draw'):
    # Ensure all score points 0-7 are represented in this draw
    # Some draws might not have any students getting a specific score (e.g., a score of 0)
    draw_counts = group.set_index('total')['count'].reindex(range(8), fill_value=0).values
    
    # Calculate Chi-square for THIS draw
    chi2_rep = np.sum(((draw_counts - ydcm_scores['count'])**2) / (ydcm_scores['count'] + 1e-9))
    chi2_rep_list.append(chi2_rep)

np.mean(np.array(chi2_rep_list) >= chi2_obs)
chi_rep_df = pd.DataFrame({'chi_rep': chi2_rep_list})

pn.ggplot.show(
  pn.ggplot(chi_rep_df,
            pn.aes('chi_rep'))
  + pn.geom_histogram(color = 'black',
                      fill = jpcolor)
  + pn.geom_vline(xintercept = chi2_obs,
                  color = 'red',
                  linetype = 'dashed')
)










# ITEM SPECIFIC PPP VALUES
# need to recalculate these values

# p values over .975 or less than .025 
[chi2_contingency(pd.crosstab(ydcm_wide['item1'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item1: item2, item5, item7
[chi2_contingency(pd.crosstab(ydcm_wide['item2'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item2: item2, item4, item7
[chi2_contingency(pd.crosstab(ydcm_wide['item3'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item3: item1, item3, item4, item5, item6
[chi2_contingency(pd.crosstab(ydcm_wide['item4'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item4: item1, item2, item3, item4, item6, item7
[chi2_contingency(pd.crosstab(ydcm_wide['item5'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item5: item3, item5, item6
[chi2_contingency(pd.crosstab(ydcm_wide['item6'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item6: item1, item3, item4, item5, item6
[chi2_contingency(pd.crosstab(ydcm_wide['item7'], ydcm_wide[i])).pvalue for i in ydcm_wide.filter(regex = 'item').columns]
# X2 similar to item7: item2, item4, item7



# BAYESIAN NETWORK DINO MODEL
dino_bn = joblib.load(here('joblib_models/quiz1_model_bayesnet_modfit.joblib'))
dino_bn_prior = joblib.load(here('joblib_models/quiz1_model_bayesnet_modfit_prior_only.joblib'))

ibn = az.from_cmdstanpy(
    posterior = dino_bn[1],
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y_item.filter(regex = 'item')},
    log_likelihood = {'Y': 'eta'}
    )

ibn = ibn.rename(name_dict = name_mapping, groups = ["posterior_predictive"])

ibn_prior = az.from_cmdstanpy(prior = dino_bn_prior[1],
prior_predictive = ['y_rep'])

ibn_prior = ibn_prior.rename(
    name_dict = name_mapping,
    groups = ['prior_predictive']
)

ibn.extend(ibn_prior)

dinobn_df = dino_bn[1].draws_pd()

az.loo(ibn)
az.waic(ibn)

acceptable_fit_stat(inference_data = ibn, func_name = 'waic')
acceptable_fit_stat(inference_data = ibn, func_name = 'loo')

ybn = dinobn_df.filter(regex = '^y_rep')

ybn_long = ybn.melt()
ybn_long['variable'] = ybn_long['variable'].str.replace('y_rep[', '')
ybn_long['variable'] = ybn_long['variable'].str.replace(']', '')
ybn_long[['stu', 'item']] = ybn_long['variable'].str.split(',', expand = True)
ybn_long = ybn_long[['stu', 'item', 'value']]
ybn_long[['stu', 'item']] = ybn_long[['stu', 'item']].astype(int)

# ybn_long_count = ybn_long.groupby(['item', 'draw'])['value'].value_counts().reset_index()

ybn_long['draw'] = ybn_long.groupby(['stu', 'item']).cumcount()

ybn_wide = ybn_long.pivot(index = ['stu', 'draw'], columns = 'item', values = 'value')
ybn_wide = ybn_wide.reset_index()
ybn_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']

ybn_wide['total'] = ybn_wide.filter(regex = 'item').sum(axis = 1)
ybn_wide_count = ybn_wide.groupby('draw')['total'].value_counts().reset_index()

pn.ggplot.show(
  pn.ggplot(ybn_wide_count,
            pn.aes('total',
                   'count'))
  + pn.geom_point(alpha = .1,
                  color = jpcolor,
                  position = pn.position_jitter())
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
)

# Calculate mean, 2.5th percentile, and 97.5th percentile
ybn_scores = ybn_wide_count.groupby('total')['count'].agg(
    count = 'mean',
    lower = q_lower,
    upper = q_upper
).reset_index()

ybn_wide_count['type'] = 'draw_counts'
ybn_scores['type'] = 'avg_counts'

ybn_wide_count['count'] = ybn_wide_count['count'].astype(float)
ybn_wide_count = ybn_wide_count.merge(ybn_scores, 'outer')

ybn_wide_count = ybn_wide_count.merge(y_item_count, 'outer')

pn.ggplot.show(
  pn.ggplot(ybn_wide_count.loc[ybn_wide_count['type'] != 'draw_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_point(pn.aes(color = 'type'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower',
                            ymax = 'upper'))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
)

pn.ggplot.show(
  pn.ggplot(ybn_wide_count.loc[ybn_wide_count['type'] != 'avg_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_point(pn.aes(color = 'type'))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [0, 7],
                          breaks = np.arange(0, 8))
  + pn.facet_wrap('type')
)


# OVERALL PPP VALUES
ybn_scores = ybn_scores.sort_values('total')

chi2_obs_bn = np.sum(((y_item_count['count'] - ybn_scores['count'])**2) / (ybn_scores['count'] + 1e-9))

draw_count_bn = ybn_wide_count.loc[ybn_wide_count['type'] == 'draw_counts']

chi2_rep_list_bn = []

for draw_id, group in draw_count_bn.groupby('draw'):
    # Ensure all score points 0-7 are represented in this draw
    # Some draws might not have any students getting a specific score (e.g., a score of 0)
    draw_counts = group.set_index('total')['count'].reindex(range(8), fill_value=0).values
    
    # Calculate Chi-square for THIS draw
    chi2_rep = np.sum(((draw_counts - ybn_scores['count'])**2) / (ybn_scores['count'] + 1e-9))
    chi2_rep_list_bn.append(chi2_rep)

np.mean(np.array(chi2_rep_list_bn) >= chi2_obs_bn)
chi_rep_df_bn = pd.DataFrame({'chi_rep': chi2_rep_list_bn})

pn.ggplot.show(
  pn.ggplot(chi_rep_df_bn,
            pn.aes('chi_rep'))
  + pn.geom_histogram(color = 'black',
                      fill = jpcolor)
  + pn.geom_vline(xintercept = chi2_obs_bn,
                  color = 'red',
                  linetype = 'dashed')
)