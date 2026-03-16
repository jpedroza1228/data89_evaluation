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
                                3: q2_names[3]
                                }).clean_names(case_type = 'snake')
alpha.head()

# y_item = pd.read_csv(here('zzz_presentations/quiz2_synthetic_data.csv')).clean_names(case_type = 'snake').drop(columns = 'unnamed_0')
y_item = pd.read_csv(here('zzz_presentations/quiz2_retake_synthetic_data.csv')).clean_names(case_type = 'snake').drop(columns = 'unnamed_0')
y_item.head()

# q = pd.DataFrame({'rv_dist': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#                   'comp_vis_dist': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
#                   'specific_mod': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#                   'cont_mod': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]})

q_retake = pd.DataFrame({'rv_dist': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  'comp_vis_dist': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                  'specific_mod': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  'cont_mod': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]})

# only using retake data for 
stan_dict = {
  'J': y_item.shape[0],
  'I': y_item.shape[1],
  'C': alpha.shape[0],
  # 'K': q.shape[1],
  'K': q_retake.shape[1],
  'Y': np.array(y_item),
  # 'Q': np.array(q),
  'Q': np.array(q_retake),
  'alpha': np.array(alpha)
}

# np.mean(np.random.beta(15, 10, 200))

dcm_file = os.path.join(here(f'quiz_models/quiz2_model.stan'))
dcm_model = CmdStanModel(stan_file = dcm_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_fit = dcm_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 2,
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
                        chains = 2,
                        # adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)
dcm_prior_diagnose = pd.DataFrame(dcm_prior_fit.summary())


print(dcm_diagnose['R_hat'].sort_values(ascending = False).head())
print(dcm_prior_diagnose['R_hat'].sort_values(ascending = False).head())


# dcm_diagnose.to_csv(here(f'diagnostics/quiz2_model.csv'))
# (
#   joblib.dump([dcm_model, dcm_fit],
#               here(f'joblib_models/quiz2_modfit.joblib'),
#               compress = 3)
# )

# dcm_prior_diagnose.to_csv(here(f'diagnostics/quiz2_model_prior_only.csv'))
# (
#   joblib.dump([dcm_prior_model, dcm_prior_fit],
#               here(f'joblib_models/quiz2_modfit_prior_only.joblib'),
#               compress = 3)
# )


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
az.plot_dist_comparison(idcm, var_names = ['slip'])
az.plot_dist_comparison(idcm, var_names = ['guess'])

az.plot_dist_comparison(idcm, var_names = ['lambda1'])
az.plot_dist_comparison(idcm, var_names = ['lambda2'])
az.plot_dist_comparison(idcm, var_names = ['lambda3'])
az.plot_dist_comparison(idcm, var_names = ['lambda4'])

az.plot_trace(idcm, var_names = ['nu'])
az.plot_trace(idcm, var_names = ['slip'])
az.plot_trace(idcm, var_names = ['guess'])

az.plot_forest(idcm.posterior["prob_resp_class"].isel(prob_resp_class_dim_0 = slice(0, 1),
                                                    prob_resp_class_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_resp_class',
               colors = jpcolor)

az.plot_forest(idcm.posterior["prob_resp_attr"].isel(prob_resp_attr_dim_0 = slice(0, 1),
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

sg = dcmdf.filter(regex = 'guess|slip').reset_index()
sg = sg.rename(columns = {'index': 'draw'})

sglong = sg.melt(id_vars = 'draw')
# sglong['variable'] = sglong['variable'].str.replace('[', '')
sglong['variable'] = sglong['variable'].str.replace(']', '')
sglong[['type', 'item']] = sglong['variable'].str.split('[', expand = True)
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
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability Guessing/Slipping',
            x = 'Item',
            y = 'Probability',
            caption = 'Guess = Guessed and got answer correct\nSlip = Knew the answer, but got it incorrect')
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

pn.ggplot.show(
  pn.ggplot(piavg.loc[piavg['latclass'].isin([16])],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, 17],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16, 17])
  # + pn.coord_flip()
  # + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'For Latent Class With Proficiency in All Skills',
  x = 'Item',
  y = "Posterior Mean")
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
  + pn.labs(title = 'Proficiency in Attributes/Skills',
       x = 'Student',
       y = 'Posterior Mean',
       caption = '95% Credible Intervals shown')
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
attr_mast = pd.concat([attravg_w, y_item], axis = 1)

attr_mast['attr1_name'] = np.where(attr_mast['attr1'] == 1, f'Proficient in {q2_names[0]}', f'Did not meet proficiency of {q2_names[0]}')

attr_mast['attr2_name'] = np.where(attr_mast['attr2'] == 1, f'Proficient in {q2_names[1]}', f'Did not meet proficiency of {q2_names[1]}')

attr_mast['attr3_name'] = np.where(attr_mast['attr3'] == 1, f'Proficient in {q2_names[2]}', f'Did not meet proficiency of {q2_names[2]}')

attr_mast['attr4_name'] = np.where(attr_mast['attr4'] == 1, f'Proficient in {q2_names[3]}', f'Did not meet proficiency of {q2_names[3]}')

attr_col = attr_mast.filter(regex = 'attr').columns.tolist()
attr_mast = attr_mast[attr_col]

# attr_mast
# attr_mast.to_csv(here('zzz_presentations/synthetic_attr_mastery_quiz2.csv'))


# y_sub.loc[~y_sub['anon_id'].isin(attr_mast['anon_id'])]

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
class_max_df = class_max_df.sort_values('latclass')
class_max_df['latclass'] = ['0000', #1
                            '0001', #2
                            '0011', #3
                            # '0010', #4
                            '0100', #5
                            '0101', #6
                            '0110', #7
                            '0111', #8
                            '1000', #9
                            '1001', #10
                            # '1010', #11
                            # '1011', #12
                            # '1100', #13
                            '1101', #14
                            # '1110', #15
                            '1111'] #16
gt.show(gt(class_max_df).tab_header(title = 'Latent Class Frequencies'))


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
# ydcm_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10', 'item11', 'item12', 'item13', 'item14', 'item15', 'item16']
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
  + pn.labs(title = 'Comparison Between Actual Counts and Posterior Average Counts',
            # subtitle = 'For Total Scores on Quiz',
            subtitle = 'For Total Scores on Quiz Retake',
            x = 'Total Score',
            y = 'Counts')
  + pn.scale_x_continuous(limits = [0, 16],
                          breaks = np.arange(0, 17))
  + pn.theme(legend_title = pn.element_blank())
)

y_describe = y_item.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe.drop(columns = 'index').transpose().round(2)

stu_n = y_item.shape[0]

t_stats_dict = {}

# Loop from 1 to 7
for i in range(1, 16 + 1):
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

pn.ggplot.show(
  pn.ggplot(y_long_avg.loc[y_long_avg['item'] == 11],
            pn.aes('t_draw'))
  + pn.geom_density(color = 'black',
                      fill = jpcolor,
                      alpha = .5)
  + pn.geom_vline(pn.aes(xintercept = 'observed_t'),
                  color = 'red',
                  linetype = 'dashed')
)

# pd.DataFrame({'item': np.arange(17),
#               't_prop_over': t_compare_list}).to_csv(here('diagnostics/quiz2_ppmc_item_level.csv'))


# Need comparison between 

post_mast = pd.read_csv(here('zzz_presentations/synthetic_attr_mastery_quiz2_retake.csv'))

attr_mast

# pre_actual = pd.read_csv(here('data/quiz_data/q2_scores_anonymized.csv')).clean_names(case_type = 'snake')
# post_actual = pd.read_csv(here('data/quiz_data/q2_retake_scores_anonymized.csv')).clean_names(case_type = 'snake')

# pre = pd.concat([pre_actual['anon_id'], attr_mast], axis = 1)
# post = pd.concat([post_actual['anon_id'], post_mast], axis = 1)

synth = pre.merge(post, 'inner', on = 'anon_id')

synth[['attr1_x', 'attr1_y']].value_counts().reset_index()
synth[['attr2_x', 'attr2_y']].value_counts().reset_index()
synth[['attr3_x', 'attr3_y']].value_counts().reset_index()
synth[['attr4_x', 'attr4_y']].value_counts().reset_index()