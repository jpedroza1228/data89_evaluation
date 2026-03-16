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

q2_names = ['Random Variables & Distributions', 'Computing & Visualizing Distributions', 'Specific Models', 'Continuous Models']

# attribute mastery matrix
alpha = pd.DataFrame([(a, b, c, d) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2)])
alpha = alpha.rename(columns = {0: q2_names[0],
                                1: q2_names[1],
                                2: q2_names[2],
                                3: q2_names[3]}).clean_names(case_type = 'snake')
alpha.head()


y = pd.read_csv(here('data/quiz_data/q2_retake_scores_anonymized.csv')).clean_names(case_type = 'snake')
y.head()

# true answers
y.columns.tolist()
y.shape

# item1
y[['drop1',
   'def_true1',
   'def_true2',
   'def_true3']] = y['quiz_2_definitions_true_answer'].str.split('"key":"', expand = True)

y['def_true1'] = y['def_true1'].str[0]
y['def_true2'] = y['def_true2'].str[0]
y['def_true3'] = y['def_true3'].str[0]

# item2
y[['drop2',
   'sum2_true1',
   'sum2_true2']] = y['quiz_2_retake_sum_of_two_rolls_true_answer'].str.split('"key":"', expand = True)
y['sum2_true1'] = y['sum2_true1'].str[0]
y['sum2_true2'] = y['sum2_true2'].str[0]

# item3
y[['drop3',
   'disttable_true1']] = y['quiz_2_retake_distribution_tables_true_answer'].str.split('"key":"', expand = True)
y[['disttable_true1',
  'disttable_true2']] = y['disttable_true1'].str.split('"ans_interval":', expand = True)
y['disttable_true1'] = y['disttable_true1'].str[0]
y['disttable_true2'] = y['disttable_true2'].str.replace('}', '')

# item4
y[['drop4',
   'vispmf_true1',
   'vispmf_true2']] = y['quiz_2_retake_visualizing_pmfs_true_answer'].str.split('"key":"', expand = True)
y['vispmf_true1'] = y['vispmf_true1'].str[0]
y['vispmf_true2'] = y['vispmf_true2'].str[0]

# item5
y[['drop5',
   'dice_true1',
   'dice_true2',
   'dice_true3']] = y['quiz_2_retake_dice_roll_game_true_answer'].str.split('"key":"', expand = True)
y['dice_true1'] = y['dice_true1'].str[0]
y['dice_true2'] = y['dice_true2'].str[0]
y[['dice_true3', 'dice_true4']] = y['dice_true3'].str.split('"part_d_ans":"', expand = True)
y['dice_true3'] = y['dice_true3'].str[0]
y['dice_true4'] = y['dice_true4'].str.replace('"}', '')

# item6
y[['drop6',
   'marathon_true1',
   'marathon_true2']] = y['quiz_2_retake_marathon_true_answer'].str.split('"key":"', expand = True)
y['marathon_true1'] = y['marathon_true1'].str[0]
y['marathon_true2'] = y['marathon_true2'].str[0]

# item7
y[['drop7',
   'denscomp_true1',
   'denscomp_true2']] = y['quiz_2_retake_density_computations_true_answer'].str.split('":"', expand = True)
y['denscomp_true1'] = y['denscomp_true1'].str.replace('","part_b_ans', '')
y['denscomp_true2'] = y['denscomp_true2'].str.replace('"}', '')

y.filter(regex = r'true[1234]')

# submitted answers
y.columns.tolist()

# item1
y['quiz_2_definitions_submitted_answer'][0]
y[['drop8',
   'def_submit1',
   'def_submit2',
   'def_submit3']] = y['quiz_2_definitions_submitted_answer'].str.split('":"', expand = True)
y['def_submit1'] = y['def_submit1'].str[0]
y['def_submit2'] = y['def_submit2'].str[0]
y['def_submit3'] = y['def_submit3'].str[0]

# item2
y[['drop9',
   'sum2_submit1', 
   'sum2_submit2']] = y['quiz_2_retake_sum_of_two_rolls_submitted_answer'].str.split('":"', expand = True)
y['sum2_submit1'] = y['sum2_submit1'].str[0]
y['sum2_submit2'] = y['sum2_submit2'].str[0]

# item3
y[['drop10',
   'disttable_submit1',
   'disttable_submit2']] = y['quiz_2_retake_distribution_tables_submitted_answer'].str.split(':', expand = True)
y['disttable_submit1'] = y['disttable_submit1'].str.replace('"', '')
y['disttable_submit1'] = y['disttable_submit1'].str[0]
y['disttable_submit2'] = y['disttable_submit2'].str.replace('}', '')

# item4
y[['drop11',
   'vispmf_submit1',
   'vispmf_submit2']] = y['quiz_2_retake_visualizing_pmfs_submitted_answer'].str.split('":"', expand = True)
y['vispmf_submit1'] = y['vispmf_submit1'].str[0]
y['vispmf_submit2'] = y['vispmf_submit2'].str.replace('"}', '')

# item5
y[['drop12',
   'dice_submit1',
   'dice_submit2',
   'dice_submit3']] = y['quiz_2_retake_dice_roll_game_submitted_answer'].str.split('":"', expand = True)
y['dice_submit1'] = y['dice_submit1'].str[0]
y['dice_submit2'] = y['dice_submit2'].str[0]
y[['dice_submit3',
   'dice_submit4']] = y['dice_submit3'].str.split('"part_d_ans":', expand = True)
y['dice_submit3'] = y['dice_submit3'].str[0]
y['dice_submit4'] = y['dice_submit4'].str.replace('}', '')

# item6
y[['drop13',
   'marathon_submit1',
   'marathon_submit2']] = y['quiz_2_retake_marathon_submitted_answer'].str.split('":"', expand = True)
y['marathon_submit1'] = y['marathon_submit1'].str[0]
y['marathon_submit2'] = y['marathon_submit2'].str.replace('"}', '')

# item7
y[['drop14',
   'denscomp_submit1',
   'denscomp_submit2']] = y['quiz_2_retake_density_computations_submitted_answer'].str.split('":', expand = True)
y['denscomp_submit1'] = y['denscomp_submit1'].str.replace(',"part_b_ans', '')
y['denscomp_submit2'] = y['denscomp_submit2'].str.replace('}', '')

y.filter(regex = r'submit[1234]')

# true and submitted answers
true_col = y.filter(regex = r'true[1234]').columns.tolist()
submit_col = y.filter(regex = r'submit[1234]').columns.tolist()

y_sub = y[['anon_id'] + true_col + submit_col]

y_sub['disttable_true2'] = y_sub['disttable_true2'].astype(float)
y_sub['denscomp_true1'] = y_sub['denscomp_true1'].astype(float)
y_sub['denscomp_true2'] = y_sub['denscomp_true2'].astype(float)

y_sub['denscomp_submit1'] = np.where(y_sub['denscomp_submit1'] == 'null', -99, y_sub['denscomp_submit1'])
y_sub['denscomp_submit2'] = np.where(y_sub['denscomp_submit2'] == 'null', -99, y_sub['denscomp_submit2'])

y_sub['disttable_submit2'] = y_sub['disttable_submit2'].astype(float).round(2)
y_sub['denscomp_submit1'] = y_sub['denscomp_submit1'].astype(float).round(2)
y_sub['denscomp_submit2'] = y_sub['denscomp_submit2'].astype(float).round(2)

# y_sub.to_csv(here('data/quiz_data/quiz2_retake_ready_irt.csv'))

y_item = y_sub.drop(columns = 'anon_id')

y_item.columns.tolist()

# y_item.filter(regex = 'def')
y_item['item1'] = np.where(y_item['def_submit1'] == y_item['def_true1'], 1, 0)
y_item['item2'] = np.where(y_item['def_submit2'] == y_item['def_true2'], 1, 0)
y_item['item3'] = np.where(y_item['def_submit3'] == y_item['def_true3'], 1, 0)

# y_item.filter(regex = 'min2')
y_item['item4'] = np.where(y_item['sum2_submit1'] == y_item['sum2_true1'], 1, 0)
y_item['item5'] = np.where(y_item['sum2_submit2'] == y_item['sum2_true2'], 1, 0)

# y_item.filter(regex = 'disttable')
y_item['item6'] = np.where(y_item['disttable_submit1'] == y_item['disttable_true1'], 1, 0)
y_item['item7'] = np.where(y_item['disttable_submit2'] == y_item['disttable_true2'], 1, 0)

# y_item.filter(regex = 'vispmf')
y_item['item8'] = np.where(y_item['vispmf_submit1'] == y_item['vispmf_true1'], 1, 0)
y_item['item9'] = np.where(y_item['vispmf_submit2'] == y_item['vispmf_true2'], 1, 0)

# y_item.filter(regex = 'iddist')
y_item['item10'] = np.where(y_item['dice_submit1'] == y_item['dice_true1'], 1, 0)
y_item['item11'] = np.where(y_item['dice_submit2'] == y_item['dice_true2'], 1, 0)
y_item['item12'] = np.where(y_item['dice_submit3'] == y_item['dice_true3'], 1, 0)
y_item['item13'] = np.where(y_item['dice_submit4'] == y_item['dice_true4'], 1, 0)

# y_item.filter(regex = 'alice')
y_item['item14'] = np.where(y_item['marathon_submit1'] == y_item['marathon_true1'], 1, 0)
y_item['item15'] = np.where(y_item['marathon_submit2'] == y_item['marathon_true2'], 1, 0)

# y_item.filter(regex = 'denscomp')
y_item['item16'] = np.where(y_item['denscomp_submit1'] == y_item['denscomp_true1'], 1, 0)
y_item['item17'] = np.where(y_item['denscomp_submit2'] == y_item['denscomp_true2'], 1, 0)

y_item = y_item.filter(regex = 'item')

q_retake = pd.DataFrame({'rv_dist': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  'comp_vis_dist': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                  'specific_mod': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  'cont_mod': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]})
# q_retake.to_csv(here('data/q_matrix/q2_retake_granular.csv'))

# np.mean(np.random.beta(2, 1, 100))

# only using retake data for 
stan_dict = {
  'J': y_item.shape[0],
  'I': y_item.shape[1],
  'C': alpha.shape[0],
  'K': q_retake.shape[1],
  'Y': np.array(y_item),
  'Q': np.array(q_retake),
  'alpha': np.array(alpha)
}

dcm_file = os.path.join(here(f'quiz_models/quiz2_model.stan'))
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


dcm_prior_file = os.path.join(here(f'quiz_models/quiz2_model_prior_only.stan'))
dcm_prior_model = CmdStanModel(stan_file = dcm_prior_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_prior_fit = dcm_prior_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)
dcm_prior_diagnose = pd.DataFrame(dcm_prior_fit.summary())


print(dcm_diagnose['R_hat'].sort_values(ascending = False).head())
print(dcm_prior_diagnose['R_hat'].sort_values(ascending = False).head())


dcm_diagnose.to_csv(here(f'diagnostics/quiz2_retake_model.csv'))
(
  joblib.dump([dcm_model, dcm_fit],
              here(f'joblib_models/quiz2_retake_modfit.joblib'),
              compress = 3)
)

dcm_prior_diagnose.to_csv(here(f'diagnostics/quiz2_retake_model_prior_only.csv'))
(
  joblib.dump([dcm_prior_model, dcm_prior_fit],
              here(f'joblib_models/quiz2_retake_modfit_prior_only.joblib'),
              compress = 3)
)

idcm = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y_item.filter(regex = 'item')},
    log_likelihood = {'Y': 'eta'}
    )

idcm = idcm.rename(name_dict = {'y_rep': 'Y'}, groups = ["posterior_predictive"])

idcm_prior = az.from_cmdstanpy(prior = dcm_prior_fit,
prior_predictive = ['y_rep'])

idcm_prior = idcm_prior.rename(
    name_dict = {'y_rep': 'Y'},
    groups = ['prior_predictive']
)

idcm.extend(idcm_prior)


# plots
az.plot_dist_comparison(idcm, var_names = ['nu'])
az.plot_dist_comparison(idcm, var_names = ['tp'])
az.plot_dist_comparison(idcm, var_names = ['fp'])

az.plot_dist_comparison(idcm, var_names = ['lambda1'])
az.plot_dist_comparison(idcm, var_names = ['lambda2'])
az.plot_dist_comparison(idcm, var_names = ['lambda3'])
az.plot_dist_comparison(idcm, var_names = ['lambda4'])

az.plot_trace(idcm, var_names = 'nu')
az.plot_trace(idcm, var_names = ['tp'])
az.plot_trace(idcm, var_names = ['fp'])

az.plot_forest(idcm.posterior["prob_resp_class"].isel(prob_resp_class_dim_0 = slice(0, 2),
                                                    prob_resp_class_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_class',
               colors = jpcolor)

az.plot_forest(idcm.posterior["prob_resp_attr"].isel(prob_resp_attr_dim_0 = slice(0, 10),
                                                    prob_resp_attr_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_attr',
               colors = jpcolor)

az.loo(idcm)
acceptable_fit_stat(inference_data = idcm, func_name = 'waic')
acceptable_fit_stat(inference_data = idcm, func_name = 'loo')

az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            kind = 'cumulative')

az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'mean')
az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'std')

dcmdf = dcm_fit.draws_pd()


sg = dcmdf.filter(regex = 'tp|fp').reset_index()
sg = sg.rename(columns = {'index': 'draw'})

sglong = sg.melt(id_vars = 'draw')
sglong['variable'] = sglong['variable'].str.replace('[', '')
sglong['variable'] = sglong['variable'].str.replace(']', '')
sglong['type'] = sglong['variable'].str.slice(start = 0, stop = 2)
sglong['item'] = sglong['variable'].str.slice(start = 2) 
sglong = sglong[['draw', 'type', 'item', 'value']]
sglong[['draw', 'item']] = sglong[['draw', 'item']].astype(int)

sgavg = sglong.groupby(['item', 'type'])

sgavg = pd.DataFrame({
  'mean': sgavg['value'].mean(),
  'std': sgavg['value'].std(),
  'q_lower': q_lower(sgavg['value']),
  'q_upper': q_upper(sgavg['value'])
}).reset_index()

pn.ggplot.show(
  pn.ggplot(sgavg,
    pn.aes('item', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  alpha = .7)
  + pn.geom_point(pn.aes(color = 'type'),
                  alpha = .7)
  + pn.facet_wrap('type')
  + pn.scale_x_continuous(limits = [1, 17],
                          breaks = np.arange(1, 18, 1))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability Guessing/Slipping',
            x = 'Item',
            y = 'Probability',
            caption = 'fp = Guessed and got answer correct\ntp = No slipping. Actually got answer correct')
  + pn.theme(legend_position = 'none')
)


pidf = dcmdf.filter(regex = 'pi').reset_index()
pidf = pidf.rename(columns = {'index': 'draw'})
pilong = pidf.melt(id_vars = 'draw')
pilong['variable'] = pilong['variable'].str.replace('pi[', '')
pilong['variable'] = pilong['variable'].str.replace(']', '')
pilong[['item', 'latclass']] = pilong['variable'].str.split(',', expand = True)
pilong = pilong[['draw', 'item', 'latclass', 'value']]
pilong[['draw', 'item', 'latclass']] = pilong[['draw', 'item', 'latclass']].astype(int)

piavg = pilong.groupby(['item', 'latclass'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

pn.ggplot.show(
  pn.ggplot(piavg,
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 16],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  caption = '1 = 0000 | 2 = 0001 | 3 = 0010 | 4 = 0011\n5 = 0100 | 6 = 0101 | 7 = 0110 | 8 = 0111\n9 = 1000 | 10 = 1001 | 11 = 1010 | 12 = 1011\n13 = 1100 | 14 = 1101 | 15 = 1110 | 16 = 1111')
  # caption = '1 = 00\n2 = 01\n3 = 10\n4 = 11')
  + pn.theme(legend_position = 'none')
)
# breakdown of latent classes
pn.ggplot.show(
  pn.ggplot(piavg.loc[piavg['latclass'].isin([1, 2, 3, 4])],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 16],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  caption = '1 = 0000 | 2 = 0001 | 3 = 0010 | 4 = 0011')
  + pn.theme(legend_position = 'none')
)

pn.ggplot.show(
  pn.ggplot(piavg.loc[piavg['latclass'].isin([5, 6, 7, 8])],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 16],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  caption = '5 = 0100 | 6 = 0101 | 7 = 0110 | 8 = 0111')
  + pn.theme(legend_position = 'none')
)

pn.ggplot.show(
  pn.ggplot(piavg.loc[piavg['latclass'].isin([9, 10, 11, 12])],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 16],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  caption = '9 = 1000 | 10 = 1001 | 11 = 1010 | 12 = 1011')
  + pn.theme(legend_position = 'none')
)

pn.ggplot.show(
  pn.ggplot(piavg.loc[piavg['latclass'].isin([13, 14, 15, 16])],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 16],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  caption = '13 = 1100 | 14 = 1101 | 15 = 1110 | 16 = 1111')
  + pn.theme(legend_position = 'none')
)


attrdf = dcmdf.filter(regex = '^prob_resp_attr').reset_index()
attrdf = attrdf.rename(columns = {'index': 'draw'})
attrlong = attrdf.melt(id_vars = 'draw')

attrlong['variable'] = attrlong['variable'].str.replace('prob_resp_attr[', '')
attrlong['variable'] = attrlong['variable'].str.replace(']', '')
attrlong[['stu', 'attr']] = attrlong['variable'].str.split(',', expand = True)
attrlong[['draw', 'stu', 'attr']] = attrlong[['draw', 'stu', 'attr']].astype(int)
attrlong = attrlong[['draw', 'stu', 'attr', 'value']]

attravg = attrlong.groupby(['stu', 'attr'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

pn.ggplot.show(
  pn.ggplot(attravg,
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

attravg['mastery'] = np.where(attravg['mean'] > .8, 1, 0)

attravg_w = attravg.pivot(index = 'stu', columns = 'attr', values = ['mastery', 'mean'])
attravg_w.columns = ['attr1',
                     'attr2',
                     'attr3',
                     'attr4',
                     'attr1_avg',
                     'attr2_avg',
                     'attr3_avg',
                     'attr4_avg']
attr_mast = pd.concat([attravg_w, y_sub], axis = 1)

attr_mast['attr1_name'] = np.where(attr_mast['attr1'] == 1, f'Proficient in {q2_names[0]}', f'Did not meet proficiency of {q2_names[0]}')

attr_mast['attr2_name'] = np.where(attr_mast['attr2'] == 1, f'Proficient in {q2_names[1]}', f'Did not meet proficiency of {q2_names[1]}')

attr_mast['attr3_name'] = np.where(attr_mast['attr3'] == 1, f'Proficient in {q2_names[2]}', f'Did not meet proficiency of {q2_names[2]}')

attr_mast['attr4_name'] = np.where(attr_mast['attr4'] == 1, f'Proficient in {q2_names[3]}', f'Did not meet proficiency of {q2_names[3]}')

attr_col = attr_mast.filter(regex = 'attr').columns.tolist()
attr_mast = attr_mast[['anon_id'] + attr_col]

attr_mast.to_csv(here('student_data/attr_mastery_quiz2_retake.csv'))

# these are students who only took the retake
# they 
y_sub.loc[~y_sub['anon_id'].isin(attr_mast['anon_id'])]

gt.show(gt(attr_mast[['attr1', 'attr2', 'attr3', 'attr4']].value_counts().reset_index()))


attravg['acc_comp'] = attravg['mean'].apply(lambda p: max(p, 1 - p))
attravg['cons_comp'] = attravg['mean'].apply(lambda p: p**2 + (1 - p)**2)

reliability = attravg.groupby('attr').agg(
    accuracy=('acc_comp', 'mean'),
    consistency=('cons_comp', 'mean')
).reset_index()
gt.show(gt(reliability.round(3)).tab_header(title = 'Accuracy & Consistency'))


attr_class = dcmdf.filter(regex = '^prob_resp_class').reset_index()
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

class_max_df = class_max['latclass'].value_counts().reset_index()
class_max_df


ydcm = dcmdf.filter(regex = '^y_rep')

# calculations for odds ratios/conditional probabilities
ydcm_long = ydcm.melt()

ydcm_long['variable'] = ydcm_long['variable'].str.replace('y_rep[', '')
ydcm_long['variable'] = ydcm_long['variable'].str.replace(']', '')
ydcm_long[['stu', 'item']] = ydcm_long['variable'].str.split(',', expand = True)
ydcm_long = ydcm_long[['stu', 'item', 'value']]
ydcm_long[['stu', 'item']] = ydcm_long[['stu', 'item']].astype(int)
ydcm_long['draw'] = ydcm_long.groupby(['stu', 'item']).cumcount()

ydcm_wide = ydcm_long.pivot(index = ['stu', 'draw'], columns = 'item', values = 'value')
ydcm_wide = ydcm_wide.reset_index()
ydcm_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10', 'item11', 'item12', 'item13', 'item14', 'item15', 'item16', 'item17']

ydcm_wide['total'] = ydcm_wide.filter(regex = 'item').sum(axis = 1)
ydcm_wide_count = ydcm_wide.groupby('draw')['total'].value_counts().reset_index()

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

ydcm_wide_count.head()

pn.ggplot.show(
  pn.ggplot(ydcm_wide_count.loc[ydcm_wide_count['type'] != 'draw_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower',
                            ymax = 'upper'),
                     alpha = .5,
                     linetype = 'dashed')
  + pn.geom_point(pn.aes(color = 'type'))
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [0, 16],
                          breaks = np.arange(0, 17))
)

y_describe = y_item.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe.drop(columns = 'index').transpose().round(2)

stu_n = y_item.shape[0]

t_stats_dict = {}

# Loop from 1 to 7
for i in range(1, 17):
    item_name = f"item{i}"
    
    # Extract mean and std for the specific item
    # We use .values[0] to get the scalar number out of the filtered dataframe
    avg = y_describe.loc[y_describe['index'] == 'mean', item_name].values[0]
    std = y_describe.loc[y_describe['index'] == 'std', item_name].values[0]
    
    # Calculate the observed t-value
    # Formula: T = avg / (std / sqrt(n))
    t_val = avg / ((std + 1e-10) / np.sqrt(stu_n))
    
    # Store it in our dictionary
    t_stats_dict[item_name] = t_val

# Convert the dictionary into a final Pandas Series
obs_t_series = pd.Series(t_stats_dict)

obs_t_series = obs_t_series.reset_index()
obs_t_series = obs_t_series.rename(columns = {'index': 'item',
                                              0: 'observed_t'})
obs_t_series['item'] = obs_t_series['item'].str.replace('item', '')
obs_t_series['item'] = obs_t_series['item'].astype(float)

y_long_avg = ydcm_long.groupby(['item', 'draw'])['value'].agg(['mean', 'std']).reset_index()
y_long_avg['n'] = stu_n

y_long_avg['t_draw'] = y_long_avg['mean']/((y_long_avg['std'] + 1e-10)/np.sqrt(y_long_avg['n']))

y_long_avg = y_long_avg.merge(obs_t_series, 'inner', 'item')

y_long_avg['t_draw'].describe()

t_compare_list = [np.mean(y_long_avg.loc[(y_long_avg['item'] == i), 't_draw'] > y_long_avg.loc[(y_long_avg['item'] == i), 'observed_t']) for i in np.arange(17)]

pd.DataFrame({'item': np.arange(17),
              't_prop_over': t_compare_list})

# pd.DataFrame({'item': np.arange(17),
#               't_prop_over': t_compare_list}).to_csv(here('diagnostics/quiz2_retake_ppmc_item_level.csv'))


# Comparison
og_mast = pd.read_csv(here('student_data/attr_mastery_quiz2.csv')).drop(columns = 'Unnamed: 0')

og_mast = og_mast.loc[:, 'anon_id':'attr4']
og_mast.columns = ['anon_id',
                   'attr1_og',
                   'attr2_og',
                   'attr3_og',
                   'attr4_og']
attr_mast = attr_mast.loc[:, 'anon_id':'attr4']

# 104 took the retake
combo = og_mast.merge(attr_mast, 'inner', 'anon_id')
combo.shape

combo[['attr1_og', 'attr1']].value_counts().reset_index()
combo[['attr2_og', 'attr2']].value_counts().reset_index()
combo[['attr3_og', 'attr3']].value_counts().reset_index()
combo[['attr4_og', 'attr4']].value_counts().reset_index()