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
      

# attribute mastery matrix
alpha = pd.DataFrame([(a, b) for a in np.arange(2) for b in np.arange(2)])
alpha = alpha.rename(columns = {0: 'Rules, Logic, Sets, & Bounds',
                                1: 'Probabilities & Computing'}).clean_names(case_type = 'snake')
# alpha = pd.DataFrame([(a, b, c) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2)])
# alpha = alpha.rename(columns = {0: 'Rules, Logic, Sets, & Bounds',
#                                 1: 'Computing',
#                                 2: 'Relating Joint, Conditional, Maginal'}).clean_names(case_type = 'snake')
alpha.head()

# y = pd.read_csv(here('data/quiz_data/q1_scores_anonymized.csv'))
# y.head()

y2 = pd.read_csv(here('data/quiz_data/q1_retake_scores_anonymized_need_recode.csv'))
y2.head()

y2 = y2.clean_names(case_type = 'snake')

# True Answers
y2[['retake_3x3_events_true1',
    'retake_3x3_events_true2',
    'retake_3x3_events_true3']] = y2['quiz_1_retake_3x3_events_true_answer'].str.split(',"part', expand = True)
y2['retake_3x3_events_true1'] = y2['retake_3x3_events_true1'].str.split(':').str[1]
y2['retake_3x3_events_true2'] = y2['retake_3x3_events_true2'].str.extract(r'"key":"([abcd])"')
y2['retake_3x3_events_true3'] = y2['retake_3x3_events_true3'].str.extract(r'"key":"([abcd])"')

y2.filter(regex = 'events_true\\d$')

y2['retake_balls_boxes_true1'] = y2['quiz_1_retake_balls_and_boxes_true_answer'].str.extract(r'"key":"([abcd])"')

y2.filter(regex = 'boxes_true\\d$')

y2[['retake_daily_rain_drop1',
    'retake_daily_rain_true1',
    'retake_daily_rain_drop2',
    'retake_daily_rain_true2',
    'retake_daily_rain_drop3']] = y2['quiz_1_retake_daily_rain_true_answer'].str.split(r'"key":"([abcd])"',expand = True)

y2.filter(regex = 'rain_true\\d$')

y2[['retake_independence_drop1',
    'retake_independence_true1',
    'retake_independence_drop2',
    'retake_independence_true2',
    'retake_independence_drop3',
    'retake_independence_true3',
    'retake_independence_drop4']] = y2['quiz_1_retake_independence_true_answer'].str.split(r'"key":"([abcd])"', expand = True)

y2.filter(regex = 'independence_true\\d$')

y2[['retake_medical_test_drop1',
    'retake_medical_test_true1',
    'retake_medical_test_drop2']] = y2['quiz_1_retake_medical_test_true_answer'].str.split(r'"key":"([abcd])"', expand = True)
y2[['retake_medical_test_drop2', 'retake_medical_test_true2']] = y2['retake_medical_test_drop2'].str.split('precision":', expand = True)
y2['retake_medical_test_true2' ]= y2['retake_medical_test_true2'].str.replace('}', '')

y2.filter(regex = 'medical_test_true\\d$')

y2[['retake_prob_space_drop1',
    'retake_prob_space_true1',
    'retake_prob_space_drop2',
    'retake_prob_space_true2',
    'retake_prob_space_drop3',
    'retake_prob_space_true3',
    'retake_prob_space_drop4',
    'retake_prob_space_true4',
    'retake_prob_space_drop5']] = y2['quiz_1_retake_probability_spaces_true_answer'].str.split(r'"key":"([abcd])"', expand = True)

y2.filter(regex = 'space_true\\d$')

y2[['retake_true_statement_drop1',
    'retake_true_statement_true1',
    'retake_true_statement_drop2',
    'retake_true_statement_true2',
    'retake_true_statement_drop3',
    'retake_true_statement_true3',
    'retake_true_statement_drop4']] = y2['quiz_1_retake_true_statements_true_answer'].str.split(r'"key":"([abcd])"', expand = True)

y2.filter(regex = 'statement_true\\d$')


# Submitted Responses
y2[['retake_3x3_submit1',
    'retake_3x3_submit2',
    'retake_3x3_submit3']] = y2['quiz_1_retake_3x3_events_submitted_answer'].str.split(',', expand = True)
y2['retake_3x3_submit1'] = y2['retake_3x3_submit1'].str.replace('{"part_a":', '')
y2['retake_3x3_submit2'] = y2['retake_3x3_submit2'].str.replace('"part_b":"', '')
y2['retake_3x3_submit2'] = y2['retake_3x3_submit2'].str.replace('"', '')
y2['retake_3x3_submit3'] = y2['retake_3x3_submit3'].str.replace('"part_c":"', '')
y2['retake_3x3_submit3'] = y2['retake_3x3_submit3'].str.replace('"}', '')

y2.filter(regex = '3x3_submit\\d$')

y2['retake_balls_boxes_submit1'] = y2['quiz_1_retake_balls_and_boxes_submitted_answer'].str.extract('"([abcd])"}$')

y2.filter(regex = 'boxes_submit\\d$')

y2[['retake_daily_rain_submit1',
    'retake_daily_rain_submit2',
    'retake_daily_rain_submit3']] = y2['quiz_1_retake_daily_rain_submitted_answer'].str.split(',', expand = True)
y2['retake_daily_rain_submit1'] = y2['retake_daily_rain_submit1'].str.replace('{"statements":', '')
y2['retake_daily_rain_submit1'] = y2['retake_daily_rain_submit1'].str.replace('["', '')
y2['retake_daily_rain_submit1'] = y2['retake_daily_rain_submit1'].str.replace('"}', '')
y2['retake_daily_rain_submit1'] = y2['retake_daily_rain_submit1'].str.replace('"', '')
y2['retake_daily_rain_submit2'] = y2['retake_daily_rain_submit2'].str.replace('"]}', '')
y2['retake_daily_rain_submit2'] = y2['retake_daily_rain_submit2'].str.replace('"', '')
y2['retake_daily_rain_submit3'] = y2['retake_daily_rain_submit3'].str.replace('"]}', '')
y2['retake_daily_rain_submit3'] = y2['retake_daily_rain_submit3'].str.replace('"', '')

y2.filter(regex = 'rain_submit\\d$')

y2['retake_independence_submit1'] = y2['quiz_1_retake_independence_submitted_answer'].str.replace('{"statements":', '')
y2['retake_independence_submit1'] = y2['retake_independence_submit1'].str.replace(r'"', '', regex = True)
y2['retake_independence_submit1'] = y2['retake_independence_submit1'].str.replace('}', '')
y2['retake_independence_submit1'] = y2['retake_independence_submit1'].str.replace(r'[\[\]]', '', regex = True)
y2[['retake_independence_submit1',
    'retake_independence_submit2',
    'retake_independence_submit3']] = y2['retake_independence_submit1'].str.split(',', expand = True)

y2.filter(regex = 'independence_submit\\d$')

y2[['retake_medical_test_submit1',
    'retake_medical_test_submit2']] = y2['quiz_1_retake_medical_test_submitted_answer'].str.split('","', expand = True)
y2['retake_medical_test_submit1'] = y2['retake_medical_test_submit1'].str.replace('{"q1":"', '')
y2['retake_medical_test_submit2'] = y2['retake_medical_test_submit2'].str.replace('precision":', '')
y2['retake_medical_test_submit2'] = y2['retake_medical_test_submit2'].str.replace('}', '')

y2.filter(regex = 'test_submit\\d$')

y2[['retake_prob_space_submit1',
    'retake_prob_space_submit2',
    'retake_prob_space_submit3',
    'retake_prob_space_submit4']] = y2['quiz_1_retake_probability_spaces_submitted_answer'].str.split(',', expand = True)
y2['retake_prob_space_submit1'] = y2['retake_prob_space_submit1'].str.extract('"([abcd])"')
y2['retake_prob_space_submit2'] = y2['retake_prob_space_submit2'].str.extract('"([abcd])"')
y2['retake_prob_space_submit3'] = y2['retake_prob_space_submit3'].str.extract('"([abcd])"')
y2['retake_prob_space_submit4'] = y2['retake_prob_space_submit4'].str.extract('"([abcd])"')

y2.filter(regex = 'space_submit\\d$')

y2['retake_true_statement_submit1'] = y2['quiz_1_retake_true_statements_submitted_answer'].str.replace('{"statements":', '')
y2['retake_true_statement_submit1'] = y2['retake_true_statement_submit1'].str.replace(r'[\["\"\]}]', '', regex = True)
y2[['retake_true_statement_submit1',
    'retake_true_statement_submit2',
    'retake_true_statement_submit3',
    'retake_true_statement_submit4']] = y2['retake_true_statement_submit1'].str.split(',', expand = True)

y2.filter(regex = 'statement_submit\\d$')

prob_space_response = y2.filter(regex = 'space_(?:true\\d$|submit\\d$)').columns.tolist()
ball_box_response = y2.filter(regex = 'boxes_(?:true\\d$|submit\\d$)').columns.tolist()
true_state_response = y2.filter(regex = 'statement_(?:true\\d$|submit\\d$)').columns.tolist()
threexthree_response = y2.filter(regex = '3x3_.*(?:true|submit)\\d$').columns.tolist()
rain_response = y2.filter(regex = 'rain_(?:true\\d$|submit\\d$)').columns.tolist()
independence_response = y2.filter(regex = 'independence_(?:true\\d$|submit\\d$)').columns.tolist()
med_test_response = y2.filter(regex = 'test_(?:true\\d$|submit\\d$)').columns.tolist()
question_score = y2.filter(regex = '%_$').columns.tolist()

# y2 = y2.rename(columns = {
#   'quiz-1-retake-3x3-events (%)': 'item4', #joint distributions/relating
#   'quiz-1-retake-balls-and-boxes (%)': 'item2', #balls in bins/computing
#   'quiz-1-retake-daily-rain (%)': 'item5', # bounds/rules_logic_sets_bounds
#   'quiz-1-retake-independence (%)': 'item6', # axioms/rules_logic_sets_bounds
#   'quiz-1-retake-medical-test (%)': 'item7', #bayes rules/relating
#   'quiz-1-retake-probability-spaces (%)': 'item1', # /rules_logic_sets_bounds
#   'quiz-1-retake-true-statements (%)': 'item3' # /rules_logic_sets_bounds
# })

# number of TOTAL ITEMS for all 7 quiz questions
# q1 = 4 items
# q2 = 1 items
# q3 = 4 items (answer all that apply)
# q4 = 3 items
# q5 = 4 items (answer all that apply)
# q6 = 4 items (answer all that apply)
# q7 = 2 items

y2 = y2[['anon_id'] + prob_space_response + ball_box_response + true_state_response + threexthree_response + rain_response + independence_response + med_test_response + question_score]

# NEED TO CREATE 2 COLUMNS FOR DAILY RAIN (ITEM 5)
# FIRST SCORE AS 0 FOR NO ANSWER = CORRECT
# THEN REVERSE SCORE THESE ITEMS
y2['retake_daily_rain_true3'] = np.nan
y2['retake_daily_rain_true4'] = np.nan
# creating submit because they got it correct for not answering
y2['retake_daily_rain_submit4'] = np.nan

# NEED TO CREATE 1 COLUMN FOR TRUE STATEMENTS (ITEM 6)
# FIRST SCORE AS 0 FOR NO ANSWER = CORRECT
# THEN REVERSE SCORE THE ITEM
y2['retake_true_statement_true4'] = np.nan

# NEED TO CREATE 1 COLUMN FOR INDEPENDENCE STATEMENTS (ITEM 3)
# FIRST SCORE AS 0 FOR NO ANSWER = CORRECT
# THEN REVERSE SCORE THE ITEM
y2['retake_independence_true4'] = np.nan
# also creating a submit because they got it correct for not answering
y2['retake_independence_submit4'] = np.nan

y2['item1'] = np.where(y2['retake_prob_space_submit1'] == y2['retake_prob_space_true1'], 1, 0)
y2['item2'] = np.where(y2['retake_prob_space_submit2'] == y2['retake_prob_space_true2'], 1, 0)
y2['item3'] = np.where(y2['retake_prob_space_submit3'] == y2['retake_prob_space_true3'], 1, 0)
y2['item4'] = np.where(y2['retake_prob_space_submit4'] == y2['retake_prob_space_true4'], 1, 0)

y2['item5'] = np.where(y2['retake_balls_boxes_submit1'] == y2['retake_balls_boxes_true1'], 1, 0)

y2['item6'] = np.select([(y2['retake_true_statement_submit1'] == y2['retake_true_statement_true1']),
           (y2['retake_true_statement_submit1'] == y2['retake_true_statement_true2']),
           (y2['retake_true_statement_submit1'] == y2['retake_true_statement_true3'])],
          [1, 1, 1],
          default = 0)
y2['item7'] = np.select([(y2['retake_true_statement_submit2'] == y2['retake_true_statement_true1']),
           (y2['retake_true_statement_submit2'] == y2['retake_true_statement_true2']),
           (y2['retake_true_statement_submit2'] == y2['retake_true_statement_true3'])],
           [1, 1, 1],
           default = 0)
y2['item8'] = np.select([(y2['retake_true_statement_submit3'].isnull() & y2['retake_true_statement_true3'].isnull()),
           (y2['retake_true_statement_submit3'] == y2['retake_true_statement_true1']),
           (y2['retake_true_statement_submit3'] == y2['retake_true_statement_true2']),
           (y2['retake_true_statement_submit3'] == y2['retake_true_statement_true3'])],
          [1, 1, 1, 1],
          default = 0)

y2['item9'] = np.select([(y2['retake_true_statement_submit4'].isnull() & y2['retake_true_statement_true4'].isnull()),
           (y2['retake_true_statement_submit4'] == y2['retake_true_statement_true1']),
           (y2['retake_true_statement_submit4'] == y2['retake_true_statement_true2']),
           (y2['retake_true_statement_submit4'] == y2['retake_true_statement_true3'])],
          [1, 1, 1, 1],
          default = 0)

y2['item10'] = np.where(y2['retake_3x3_submit1'] == y2['retake_3x3_events_true1'], 1, 0)
y2['item11'] = np.where(y2['retake_3x3_submit2'] == y2['retake_3x3_events_true2'], 1, 0)
y2['item12'] = np.where(y2['retake_3x3_submit3'] == y2['retake_3x3_events_true3'], 1, 0)

y2['item13'] = np.select([(y2['retake_daily_rain_submit1'] == y2['retake_daily_rain_true1']),
           (y2['retake_daily_rain_submit1'] == y2['retake_daily_rain_true2']),
           (y2['retake_daily_rain_submit1'] == y2['retake_daily_rain_true3'])],
          [1, 1, 1],
          default = 0)
y2['item14'] = np.select([(y2['retake_daily_rain_submit2'].isnull() & y2['retake_daily_rain_true2'].isnull()),
           (y2['retake_daily_rain_submit2'] == y2['retake_daily_rain_true1']),
           (y2['retake_daily_rain_submit2'] == y2['retake_daily_rain_true2']),
           (y2['retake_daily_rain_submit2'] == y2['retake_daily_rain_true3'])],
          [1, 1, 1, 1],
          default = 0)
y2['item15'] = np.select([(y2['retake_daily_rain_submit3'].isnull() & y2['retake_daily_rain_true3'].isnull()),
           (y2['retake_daily_rain_submit3'] == y2['retake_daily_rain_true1']),
           (y2['retake_daily_rain_submit3'] == y2['retake_daily_rain_true2']),
           (y2['retake_daily_rain_submit3'] == y2['retake_daily_rain_true3'])],
          [1, 1, 1, 1],
          default = 0)
y2['item16'] = np.select([(y2['retake_daily_rain_submit4'].isnull() & y2['retake_daily_rain_true4'].isnull()),
           (y2['retake_daily_rain_submit4'] == y2['retake_daily_rain_true1']),
           (y2['retake_daily_rain_submit4'] == y2['retake_daily_rain_true2']),
           (y2['retake_daily_rain_submit4'] == y2['retake_daily_rain_true3'])],
          [1, 1, 1, 1],
          default = 0)

y2['item17'] = np.select([(y2['retake_independence_submit1'] == y2['retake_independence_true1']),
           (y2['retake_independence_submit1'] == y2['retake_independence_true2']),
           (y2['retake_independence_submit1'] == y2['retake_independence_true3'])],
          [1, 1, 1],
          default = 0)
y2['item18'] = np.select([(y2['retake_independence_submit2'].isnull() == y2['retake_independence_true2'].isnull()),
           (y2['retake_independence_submit2'] == y2['retake_independence_true1']),
           (y2['retake_independence_submit2'] == y2['retake_independence_true2']),
           (y2['retake_independence_submit2'] == y2['retake_independence_true3'])],
          [1, 1, 1, 1],
          default = 0)
y2['item19'] = np.select([(y2['retake_independence_submit3'].isnull() == y2['retake_independence_true3'].isnull()),
           (y2['retake_independence_submit3'] == y2['retake_independence_true1']),
           (y2['retake_independence_submit3'] == y2['retake_independence_true2']),
           (y2['retake_independence_submit3'] == y2['retake_independence_true3'])],
          [1, 1, 1, 1],
          default = 0)
y2['item20'] = np.select([(y2['retake_independence_submit4'].isnull() & y2['retake_independence_true4'].isnull()),
           (y2['retake_independence_submit4'] == y2['retake_independence_true1']),
           (y2['retake_independence_submit4'] == y2['retake_independence_true2']),
           (y2['retake_independence_submit4'] == y2['retake_independence_true3'])],
          [1, 1, 1, 1],
          default = 0)


y2['item21'] = np.where(y2['retake_medical_test_submit1'] == y2['retake_medical_test_true1'], 1, 0)
y2['item22'] = np.where(y2['retake_medical_test_submit2'] == y2['retake_medical_test_true2'], 1, 0)


data_items = y2.filter(regex = 'item').columns.tolist()

# y.columns = ['anon_id', 'item1', 'item2', 'item3a', 'item3b', 'item4', 'item5', 'item6', 'item7', 'score']
# y['item3'] = y['item3a'].astype(str) + y['item3b'].astype(str)
# y['item3'] = y['item3'].str.replace('nan', '')
# y['item3'] = y['item3'].astype(float)
# y = y[['anon_id', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']]
# y.head()


# only_retake = y2[~y2['anon_id'].isin(y['anon_id'])]
# 8 students did not take Quiz 1 and only took the retake
# only_retake

# students who took quiz and retake
# y_sub = y.loc[y['anon_id'].isin(y2['anon_id']), :]
# y2_sub = y2.loc[~y2['anon_id'].isin(only_retake['anon_id']), :]

# y = y.sort_values('anon_id')
y2 = y2.sort_values('anon_id')

y2 = y2[['anon_id'] + data_items]
# y2.to_csv(here('data/quiz_data/q1_retake_clean.csv'))

y2_item = y2.filter(regex = 'item')

emaildf = pd.read_csv(here('student_data/quiz1_email_list.csv')).drop(columns = ['Unnamed: 0', 'email_draft'])
emaildf['attr1'] = np.where(emaildf['attr1'].str.contains('not'), 0, 1)
emaildf['attr2'] = np.where(emaildf['attr2'].str.contains('not'), 0, 1)
emaildf = emaildf[['anon_id', 'attr1', 'attr2']]
emaildf = emaildf.sort_values('anon_id')

# 32 students that got an email about lack of proficiency
# took the retake quiz out of 77 total retakes (that took quiz 1)
emaildf[emaildf['anon_id'].isin(y2['anon_id'])]

mastdf = pd.read_csv(here('student_data/attr_mastery_quiz1.csv'))
mastdf['attr1'] = np.where(mastdf['attr1'].str.contains('not'), 0, 1)
mastdf['attr2'] = np.where(mastdf['attr2'].str.contains('not'), 0, 1)
mastdf = mastdf[['anon_id', 'attr1', 'attr2']]
mastdf = mastdf.sort_values('anon_id')
mastdf.loc[mastdf['anon_id'].isna()]
mastdf.head()

quiz_prof = mastdf[mastdf['anon_id'].isin(y2['anon_id'])]
gt.show(gt(quiz_prof).tab_header(title = 'Proficiency Status After Quiz 1'))

quiz_prof['attr1'].value_counts(normalize = True)
np.mean(np.random.beta(16, 9, 80))
quiz_prof['attr2'].value_counts(normalize = True)
np.mean(np.random.beta(19, 6, 80))

# y2_item = pd.DataFrame({i: np.where(y2_item[i] == 100, 1, 0) for i in y2_item.columns})
y2_item.head()
y2_item.shape

#q-matrix
q = pd.read_csv(here('data/q_matrix/q1_retake_granular_2att.csv')).clean_names(case_type = 'snake')
# q = pd.read_csv(here('data/q_matrix/q1_7item_3att_slack.csv')).clean_names(case_type = 'snake')
q.columns = ['row', 'attr1', 'attr2']
# q.columns = ['row', 'attr1', 'attr2', 'attr3']
q = q.drop(columns = 'row')
q


# only using retake data for 
stan_dict = {
  'J': y2_item.shape[0],
  'I': y2_item.shape[1],
  'C': alpha.shape[0],
  'K': q.shape[1],
  'Y': np.array(y2_item),
  'Q': np.array(q),
  'alpha': np.array(alpha)
}

dcm_file = os.path.join(here(f'quiz_models/quiz1_retake_model.stan'))
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

dcm_prior_file = os.path.join(here(f'quiz_models/quiz1_retake_model_prior_only.stan'))
dcm_prior_model = CmdStanModel(stan_file = dcm_prior_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_prior_fit = dcm_prior_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .90,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

dcm_prior_diagnose = pd.DataFrame(dcm_prior_fit.summary())

print(dcm_diagnose['R_hat'].sort_values(ascending = False).head())
print('\n\n')
print(dcm_prior_diagnose['R_hat'].sort_values(ascending = False).head())

idcm = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y2_item.filter(regex = 'item')},
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
# az.plot_dist_comparison(idcm, var_names = ['lambda3'])

az.plot_trace(idcm, var_names = 'nu')
az.plot_trace(idcm, var_names = ['tp'])
az.plot_trace(idcm, var_names = ['fp'])

az.plot_forest(idcm.posterior["prob_resp_class"].isel(prob_resp_class_dim_0 = slice(0, 4),
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
  # + pn.scale_x_continuous(limits = [1, 7],
  #                         breaks = [1, 2, 3, 4, 5, 6, 7])
  + pn.scale_x_continuous(limits = [1, 22],
                          breaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                    12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22])
  + pn.coord_flip()
  + pn.facet_wrap('latclass')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class',
  # caption = '1 = 000 | 2 = 001 | 3 = 010 | 4 = 011\n5 = 100 | 6 = 101 | 7 = 110 | 8 = 111')
  caption = '1 = 00\n2 = 01\n3 = 10\n4 = 11')
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

attravg_w = attravg.pivot(index = 'stu', columns = 'attr', values = 'mastery')
attravg_w = attravg_w.rename(columns = {1: 'attr1',
                                        2: 'attr2'})
                                        # 3: 'attr3'})

attr_mast = pd.concat([attravg_w, y2], axis = 1)

# these are students who only took the retake
# they 
attr_mast.loc[attr_mast['anon_id'].isna()]

attr_mast[['attr1', 'attr2']].value_counts().reset_index()

attr_mast = attr_mast.rename(columns = {'attr1': 'attr1_retake',
                         'attr2': 'attr2_retake'})
attr_mast.head()

compare = attr_mast.merge(mastdf)
compare = compare[['anon_id', 'attr1', 'attr1_retake', 'attr2', 'attr2_retake']]
gt.show(gt(compare).tab_header(title = 'Comparison: Quiz & Retake'))

compare[['attr1', 'attr1_retake']].value_counts().reset_index()
compare[['attr2', 'attr2_retake']].value_counts().reset_index()

compare.loc[compare['anon_id'].isin(emaildf['anon_id']), ['attr1', 'attr1_retake']].value_counts().reset_index()
compare.loc[compare['anon_id'].isin(emaildf['anon_id']), ['attr2', 'attr2_retake']].value_counts().reset_index()

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

class_max['latclass'].value_counts()



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
ydcm_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10', 'item11', 'item12', 'item13', 'item14', 'item15', 'item16', 'item17', 'item18', 'item19', 'item20', 'item21', 'item22']

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

y2_item['total'] = y2_item.sum(axis = 1)
y2_item_count = y2_item['total'].value_counts().reset_index()
y2_item_count['type'] = 'actual_counts'
y2_item_count['count'] = y2_item_count['count'].astype(float)

ydcm_wide_count = ydcm_wide_count.merge(y2_item_count, 'outer')

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
  + pn.scale_x_continuous(limits = [0, 22],
                          breaks = np.arange(0, 23))
)

y_describe2 = y2_item.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe2.drop(columns = 'index').transpose()

stu_n = y2_item.shape[0]

t_stats_dict = {}

# Loop from 1 to 7
for i in range(1, 23):
    item_name = f"item{i}"
    
    # Extract mean and std for the specific item
    # We use .values[0] to get the scalar number out of the filtered dataframe
    avg = y_describe2.loc[y_describe2['index'] == 'mean', item_name].values[0]
    std = y_describe2.loc[y_describe2['index'] == 'std', item_name].values[0]
    
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

y_long_avg.loc[(y_long_avg['item'] == 1)]

# can check all 22 items
pn.ggplot.show(
  pn.ggplot(y_long_avg.loc[(y_long_avg['item'] == 2)],
  pn.aes('t_draw'))
  + pn.geom_histogram(color = jpcolor, fill = jpcolor)
  + pn.geom_vline(pn.aes(xintercept = 'observed_t'),
  color = 'black',
  linetype = 'dashed',
  size = 1.5)
)

dcm_diagnose.to_csv(here(f'diagnostics/quiz1_retake_model.csv'))
(
  joblib.dump([dcm_model, dcm_fit],
              here(f'joblib_models/quiz1_retake_modfit.joblib'),
              compress = 3)
)

dcm_prior_diagnose.to_csv(here(f'diagnostics/quiz1_retake_model_prior_only.csv'))
(
  joblib.dump([dcm_prior_model, dcm_prior_fit],
              here(f'joblib_models/quiz1_retake_modfit_prior_only.joblib'),
              compress = 3)
)













# Dynamic Bayes Net
# stan_dict_dy = {
#   'J': y_item.shape[0],
#   'I': y_item.shape[1],
#   'T': 2,
#   'K': q.shape[1],
#   'C': alpha.shape[0],
#   'Y_t1': np.array(y_item),
#   'Y_t2': np.array(y2_item),
#   'Q': np.array(q),
#   'alpha': np.array(alpha)
# }

# dcm_file_dy = os.path.join(here(f'quiz_models/quiz1_retake_model_attr3.stan'))
# dcm_model_dy = CmdStanModel(stan_file = dcm_file_dy,
#                          cpp_options={'STAN_THREADS': 'TRUE'})

# np.random.seed(12345)
# dcm_fit_dy = dcm_model_dy.sample(data = stan_dict_dy,
#                         show_console = True,
#                         chains = 4,
#                         # adapt_delta = .90,
#                         iter_warmup = 2000,
#                         iter_sampling = 2000)

# dcm_diagnose = pd.DataFrame(dcm_fit_dy.summary())

# dcm_prior_file_dy = os.path.join(here(f'quiz_models/quiz1_retake_model_attr3_prior_only.stan'))
# dcm_prior_model_dy = CmdStanModel(stan_file = dcm_prior_file_dy,
#                          cpp_options={'STAN_THREADS': 'TRUE'})

# np.random.seed(12345)
# dcm_prior_fit_dy = dcm_prior_model_dy.sample(data = stan_dict_dy,
#                         show_console = True,
#                         chains = 4,
#                         # adapt_delta = .90,
#                         iter_warmup = 2000,
#                         iter_sampling = 2000)

# dcm_prior_diagnose = pd.DataFrame(dcm_prior_fit_dy.summary())

# print(dcm_diagnose['R_hat'].sort_values(ascending = False).head())
# print(dcm_prior_diagnose['R_hat'].sort_values(ascending = False).head())


dcm_diagnose.to_csv(here(f'diagnostics/quiz1_retake_attr3.csv'))
(
  joblib.dump([dcm_model, dcm_fit],
              here(f'joblib_models/quiz1_retake_attr3_modfit.joblib'),
              compress = 3)
)

# prior only model
dcm_prior_diagnose.to_csv(here(f'diagnostics/quiz1_retake_attr3_prior_only.csv'))
(
  joblib.dump([dcm_prior_model, dcm_prior_fit],
              here(f'joblib_models/quiz1_retake_attr3_modfit_prior_only.joblib'),
              compress = 3)
)

idcm = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['y_rep_t1'],
    observed_data = {'Y': y_item.filter(regex = 'item')},
    log_likelihood = {'Y': 'eta_t1'}
    )

idcm = idcm.rename(name_dict = {'y_rep_t1': 'Y'}, groups = ["posterior_predictive"])

idcm_prior = az.from_cmdstanpy(prior = dcm_prior_fit,
prior_predictive = ['y_rep_t1'])

idcm_prior = idcm_prior.rename(
    name_dict = {'y_rep_t1': 'Y'},
    groups = ['prior_predictive']
)

idcm.extend(idcm_prior)

# time point 2
idcm2 = az.from_cmdstanpy(
    posterior = dcm_fit,
    posterior_predictive = ['y_rep_t2'],
    observed_data = {'Y': y2_item.filter(regex = 'item')},
    log_likelihood = {'Y': 'eta_t2'}
    )

idcm2 = idcm2.rename(name_dict = {'y_rep_t2': 'Y'}, groups = ["posterior_predictive"])

idcm2_prior = az.from_cmdstanpy(prior = dcm_prior_fit,
prior_predictive = ['y_rep_t2'])

idcm2_prior = idcm2_prior.rename(
    name_dict = {'y_rep_t2': 'Y'},
    groups = ['prior_predictive']
)

idcm2.extend(idcm2_prior)


# priors for first time point appear all wrong
az.plot_dist_comparison(idcm, var_names = ['theta1_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['theta1_t2'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['theta2_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['theta2_t2'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['theta3_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['theta3_t2'])
# plt.show()




az.plot_dist_comparison(idcm, var_names = ['nu_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['trans_mat'])
# plt.show()


az.plot_dist_comparison(idcm, var_names = ['tp_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['tp_t2'])
# plt.show()



az.plot_dist_comparison(idcm, var_names = ['fp_t1'])
# plt.show()

az.plot_dist_comparison(idcm, var_names = ['fp_t2'])
# plt.show()



az.plot_trace(idcm, var_names = 'nu_t1')
# plt.show()

az.plot_trace(idcm, var_names = 'trans_mat')
# plt.show()

az.plot_trace(idcm, var_names = 'nu_t2')



# not sure how useful this is
az.plot_forest(idcm.posterior['prob_class_t1'].isel(prob_class_t1_dim_0 = slice(0, 4),
                                                          prob_class_t1_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_class_t1',
               colors = jpcolor)
# plt.show()

# this latent class looks to be much more separated
az.plot_forest(idcm.posterior['prob_class_t2'].isel(prob_class_t2_dim_0 = slice(0, 4),
                                                          prob_class_t2_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_class_t2',
               colors = jpcolor)
# plt.show()



az.plot_forest(idcm.posterior['prob_attr_t1'].isel(prob_attr_t1_dim_0 = slice(0, 5),
                                                          prob_attr_t1_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_attr_t1',
               colors = jpcolor)
# plt.show()

az.plot_forest(idcm.posterior['prob_attr_t2'].isel(prob_attr_t2_dim_0 = slice(0, 5),
                                                          prob_attr_t2_dim_1 = slice(None)
                                                    ),
               var_names = 'prob_attr_t2',
               colors = jpcolor)
# plt.show()

az.plot_forest(idcm,
               var_names = 'growth_rate',
               colors = jpcolor)


az.loo(idcm)
az.waic(idcm)

az.loo(idcm2)
az.waic(idcm2)

acceptable_fit_stat(inference_data = idcm, func_name = 'waic')
acceptable_fit_stat(inference_data = idcm, func_name = 'loo')

acceptable_fit_stat(inference_data = idcm2, func_name = 'waic')
acceptable_fit_stat(inference_data = idcm2, func_name = 'loo')


az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
# plt.show()

az.plot_ppc(idcm2,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000)
# plt.show()

az.plot_ppc(idcm,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            kind = 'cumulative')

az.plot_ppc(idcm2,
            data_pairs = {'Y': 'Y'},
            num_pp_samples = 1000,
            kind = 'cumulative')



az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'mean')

az.plot_bpv(idcm2,
            kind = 't_stat', 
            t_stat = 'mean')


az.plot_bpv(idcm,
            kind = 't_stat', 
            t_stat = 'std')

az.plot_bpv(idcm2,
            kind = 't_stat', 
            t_stat = 'std')


dcmdf = dcm_fit.draws_pd()

nudf = dcmdf.filter(regex = '^nu.*t')
nudf = nudf.melt().reset_index()
nudf = nudf.rename(columns = {'index': 'draw'})
nudf[['variable', 'latclass']] = nudf['variable'].str.split('[', expand = True)
nudf['latclass'] = nudf['latclass'].str.replace(']', '')
nudf[['draw', 'latclass']] = nudf[['draw', 'latclass']].astype(int)
nudf = nudf[['variable', 'draw', 'latclass', 'value']]

prop_plot = nudf.groupby(['variable', 'latclass'])['value'].agg(['mean', q_lower, q_upper]).reset_index().round(2)

gt.show(gt(prop_plot))

pn.ggplot.show(
  pn.ggplot(prop_plot, pn.aes('latclass', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = jpcolor)
  + pn.geom_point(color = jpcolor)
  + pn.facet_wrap('variable')
  + pn.scale_x_continuous(limits = [1, 7],
                          breaks = [1, 2, 3, 4, 5, 6, 7])
)

# time point 1 slipping/guessing
slip_guess = dcmdf.filter(regex = 'tp.*t|fp.*t').reset_index()
slip_guess = slip_guess.rename(columns = {'index': 'draw'})

sg_long = slip_guess.melt(id_vars = 'draw')
sg_long[['variable', 'item']] = sg_long['variable'].str.split('[', expand = True)
sg_long['item'] = sg_long['item'].str.replace(']', '')
sg_long = sg_long[['draw', 'variable', 'item', 'value']]
sg_long[['draw', 'item']] = sg_long[['draw', 'item']].astype(int)
sg_long[['variable', 'time']] = sg_long['variable'].str.split('_', expand = True)

sg_avg = sg_long.groupby(['item', 'variable', 'time'])

sg_avg = pd.DataFrame({
  'mean': sg_avg['value'].mean(),
  'std': sg_avg['value'].std(),
  'q_lower': q_lower(sg_avg['value']),
  'q_upper': q_upper(sg_avg['value'])
}).reset_index()

pn.ggplot.show(
  pn.ggplot(sg_avg,
    pn.aes('item', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  color = jpcolor)
  + pn.geom_point(color = jpcolor)
  + pn.facet_grid('variable', 'time')
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [1, 7],
                          breaks = [1, 2, 3, 4, 5, 6, 7])
  + pn.labs(title = 'Probability Guessing/Slipping',
            x = 'Item',
            y = 'Probability',
            caption = 'tp = No slipping. Actually got answer correct.\nfp = Guessed and got answer correct')
  + pn.theme(legend_position = 'none')
)

pidf = dcmdf.filter(regex = 'pi').reset_index()
pidf = pidf.rename(columns = {'index': 'draw'})
pilong = pidf.melt(id_vars = 'draw')
pilong[['variable', 'time']] = pilong['variable'].str.split('_', expand = True)
pilong[['time', 'rest']] = pilong['time'].str.split('[', expand = True)
pilong[['item', 'latclass']] = pilong['rest'].str.split(',', expand = True)
pilong['latclass'] = pilong['latclass'].str.replace(']', '')
pilong = pilong[['draw', 'variable', 'latclass', 'item', 'time', 'value']]
pilong[['draw', 'item', 'latclass']] = pilong[['draw', 'item', 'latclass']].astype(int)

piavg = pilong.groupby(['item', 'latclass', 'time'])

piavg = pd.DataFrame({
  'mean': piavg['value'].mean(),
  'std': piavg['value'].std(),
  'q_lower': q_lower(piavg['value']),
  'q_upper': q_upper(piavg['value'])
}).reset_index()

pn.ggplot.show(
  pn.ggplot(piavg,
    pn.aes('item', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  color = jpcolor)
  + pn.geom_point(color = jpcolor)
  + pn.geom_hline(yintercept = .5,
                  color = 'black',
                  linetype = 'dashed')
  + pn.facet_grid('time', 'latclass')
  + pn.scale_x_continuous(limits = [1, 7],
                          breaks = [1, 2, 3, 4, 5, 6, 7])
  + pn.theme(legend_position = 'none')
)




attrdf = dcmdf.filter(regex = '^prob_attr').reset_index()
attrdf = attrdf.rename(columns = {'index': 'draw'})
attrlong = attrdf.melt(id_vars = 'draw')

attrlong['variable'] = attrlong['variable'].str.replace('prob_resp_attr[', '')
attrlong['variable'] = attrlong['variable'].str.replace(']', '')
attrlong[['stu', 'attr']] = attrlong['variable'].str.split(',', expand = True)
attrlong[['draw', 'stu', 'attr']] = attrlong[['draw', 'stu', 'attr']].astype(int)
attrlong = attrlong[['draw', 'stu', 'attr', 'value']]

attravg = attrlong.groupby(['stu', 'attr'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()












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