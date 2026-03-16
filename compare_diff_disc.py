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

def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)

jpcolor = 'seagreen'

os.environ['QT_API'] = 'PyQt6'
pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})
pn.theme_set(pn.theme_light())
# pio.templates.default = 'simple_white' # 'plotly_white'

# y = pd.read_csv(here('data/quiz_data/q1_scores_anonymized.csv'))

# y.columns = ['anon_id', 'item1', 'item2', 'item3a', 'item3b', 'item4', 'item5', 'item6', 'item7', 'score']
# y['item3'] = y['item3a'].astype(str) + y['item3b'].astype(str)
# y['item3'] = y['item3'].str.replace('nan', '')
# y['item3'] = y['item3'].astype(float)
# y = y[['anon_id', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7']]
# y_item = y.drop(columns = 'anon_id')
# y_item = pd.DataFrame({i: np.where(y_item[i] == 100, 1, 0) for i in y_item.columns})

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
y2_item = y2.filter(regex = 'item')


stan_dict = {
  'J': y2_item.shape[0],
  'I': y2_item.shape[1],
  'D': 2,
  # 'dim': [1, 2, 1, 2, 1, 1, 2],
  'dim': [1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
  'Y': np.array(y2_item)
}

mirt_file = os.path.join(here(f'stan_models/multidim_2pl.stan'))
mirt_model = CmdStanModel(stan_file = mirt_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
mirt_fit = mirt_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        # adapt_delta = .90,
                        iter_warmup = 2000,
                        iter_sampling = 2000)

mirt_diagnose = pd.DataFrame(mirt_fit.summary())
print(mirt_diagnose['R_hat'].sort_values(ascending = False).head())

imirt = az.from_cmdstanpy(
    posterior = mirt_fit,
    posterior_predictive = ['y_rep'],
    observed_data = {'Y': y2_item.filter(regex = 'item')}
    )

imirt = imirt.rename(name_dict = {'y_rep': 'Y'}, groups = ["posterior_predictive"])

az.plot_forest(imirt,
               var_names = 'alpha',
               colors = jpcolor)

az.plot_forest(imirt,
               var_names = 'beta',
               colors = jpcolor)

# az.plot_forest(imirt,
#                var_names = 'theta',
#                colors = jpcolor)

az.plot_forest(imirt,
               var_names = 'L_Omega',
               colors = jpcolor)

az.plot_forest(imirt,
               var_names = 'Omega',
               colors = jpcolor)

mirtdf = mirt_fit.draws_pd()

dis = mirtdf.filter(regex = '^alpha').reset_index()
dis = dis.rename(columns = {'index': 'draw'})

dislong = dis.melt(id_vars = 'draw')
dislong['variable'] = dislong['variable'].str.replace('[', '')
dislong['variable'] = dislong['variable'].str.replace(']', '')
dislong['type'] = dislong['variable'].str.slice(start = 0, stop = 5)
dislong['item'] = dislong['variable'].str.slice(start = 5) 
dislong = dislong[['draw', 'type', 'item', 'value']]
dislong[['draw', 'item']] = dislong[['draw', 'item']].astype(int)

disavg = dislong.groupby('item')

disavg = pd.DataFrame({
  'mean': disavg['value'].mean(),
  'std': disavg['value'].std(),
  'q_lower': q_lower(disavg['value']),
  'q_upper': q_upper(disavg['value'])
}).reset_index()

pn.ggplot.show(
  pn.ggplot(disavg,
    pn.aes('item', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  alpha = .7)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.theme(legend_position = 'none')
)

diff = mirtdf.filter(regex = '^beta').reset_index()
diff = diff.rename(columns = {'index': 'draw'})

difflong = diff.melt(id_vars = 'draw')
difflong['variable'] = difflong['variable'].str.replace('[', '')
difflong['variable'] = difflong['variable'].str.replace(']', '')
difflong['type'] = difflong['variable'].str.slice(start = 0, stop = 4)
difflong['item'] = difflong['variable'].str.slice(start = 4) 
difflong = difflong[['draw', 'type', 'item', 'value']]
difflong[['draw', 'item']] = difflong[['draw', 'item']].astype(int)

diffavg = difflong.groupby('item')

diffavg = pd.DataFrame({
  'mean': diffavg['value'].mean(),
  'std': diffavg['value'].std(),
  'q_lower': q_lower(diffavg['value']),
  'q_upper': q_upper(diffavg['value'])
}).reset_index()

pn.ggplot.show(
  pn.ggplot(diffavg,
    pn.aes('item', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  alpha = .7)
  + pn.geom_point(alpha = .7,
                  color = jpcolor)
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.theme(legend_position = 'none')
)