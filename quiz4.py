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

q4_names = ['Standard Deviation & Variance', 'Convergence Rates', 'Series' , 'Runction Approximation']   

# attribute mastery matrix
alpha = pd.DataFrame([(a, b, c, d) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2)])
alpha = alpha.rename(columns = {0: q4_names[0],
                                1: q4_names[1],
                                2: q4_names[2],
                                3: q4_names[3]
                                }).clean_names(case_type = 'snake')
alpha.head()

y = pd.read_csv(here('data/quiz_data/q4_scores_anonymized.csv')).clean_names(case_type = 'snake')
y.head()

# true answers
y.columns.tolist()


y[['drop1',
   'var_true1',
   'var_true2']] = y['quiz_4_variance_true_answer'].str.split('variance_part_', expand = True)

y['var_true1'] = y['var_true1'].str.replace('a":"', '')
y['var_true1'] = y['var_true1'].str.replace('","', '')
y['var_true2'] = y['var_true2'].str.replace('b":"', '')
y['var_true2'] = y['var_true2'].str.replace('"}', '')

y[['var_true1', 'var_true2']] = y[['var_true1', 'var_true2']].astype(float)

y.filter(regex = r'true[1234]')


# fastest, slowest, second_fastest, second_slowest
y[['drop2',
   'rate_true1_fast',
   'rate_true2_slow',
   'rate_true3_2fast',
   'rate_true4_2slow']] = y['quiz_4_convergence_rates_true_answer'].str.split('"key":"', expand = True)

y['rate_true1_fast'] = y['rate_true1_fast'].str[0]
y['rate_true2_slow'] = y['rate_true2_slow'].str[0]
y['rate_true3_2fast'] = y['rate_true3_2fast'].str[0]
y['rate_true4_2slow'] = y['rate_true4_2slow'].str[0]

y.filter(regex = r'true[1234]')


y[['drop3',
   'hosp_true1f',
   'hosp_true2g']] = y['quiz_4_l_hopitals_rule_true_answer'].str.split('"key":"', expand = True)
y['hosp_true1f'] = y['hosp_true1f'].str[0]
y['hosp_true2g'] = y['hosp_true2g'].str[0]

y.filter(regex = r'true[1234]')


# 3 options for question:
# limit & integral statements
# limit & ratio
# integral & ratio

y[['other1',
   'test_true1_ratio1',
   'test_true1_ratio2',
   'test_true1_ratio3']] = y['quiz_4_convergence_tests_true_answer'].str.split(r'[abcd]_ans', expand = True)


both = y.loc[y['other1'].str.contains('^{"limit'), :]

both[['test_true2_limit',
   'test_true2_int']] = both['other1'].str.split('\\],"', expand = True)
both[['drop_both1',
      'test_true2_limit1',
      'test_true2_limit2',
      'test_true2_limit3']] = both['test_true2_limit'].str.split('"key":"', expand = True)
both['test_true2_limit1'] = both['test_true2_limit1'].str[0]
both['test_true2_limit2'] = both['test_true2_limit2'].str[0]
both['test_true2_limit3'] = both['test_true2_limit3'].str[0]

both[['drop_both2',
      'test_true2_int1',
      'test_true2_int2']] = both['test_true2_int'].str.split('"key":"', expand = True)
both['test_true2_int1'] = both['test_true2_int1'].str[0]
both['test_true2_int2'] = both['test_true2_int2'].str[0]


both[['both_drop3',
   'tail_true1']] = both['quiz_4_tail_plots_true_answer'].str.split('"key":"', expand = True)
both[['both_drop4',
   'tail_true2']] = both['tail_true1'].str.split('gamma', expand = True)
both['tail_true1'] = both['tail_true1'].str[0]
both['tail_true2'] = both['tail_true2'].str.replace('":', '')
both['tail_true2'] = both['tail_true2'].str.replace('}', '')

both.filter(regex = r'true[1234]')

both[['both_drop5',
   'geo_true1']] = both['quiz_4_geometric_series_true_answer'].str.split('"key":"', expand = True)
both[['geo_true1',
   'geo_true2']] = both['geo_true1'].str.split('"geometric_series":', expand = True)
both['geo_true1'] = both['geo_true1'].str[0]
both['geo_true2'] = both['geo_true2'].str.replace('}', '')

both.filter(regex = r'true[1234]')

both[['both_drop6',
   'taylor_true1',
   'taylor_true2',
   'taylor_true3']] = both['quiz_4_taylor_series_true_answer'].str.split('"[abc]":', expand = True)
both['taylor_true1'] = both['taylor_true1'].str.replace(',', '')
both['taylor_true2'] = both['taylor_true2'].str.replace(',', '')
both['taylor_true3'] = both['taylor_true3'].str.replace('}', '')

both.filter(regex = r'true[1234]')


y[['drop4',
   'test_true1_ratio1']] = y['test_true1_ratio1'].str.split('"key":"', expand = True)
y['test_true1_ratio1'] = y['test_true1_ratio1'].str[0]

y[['drop5',
   'test_true1_ratio2']] = y['test_true1_ratio2'].str.split('"key":"', expand = True)
y['test_true1_ratio2'] = y['test_true1_ratio2'].str[0]

y[['test_true1_ratio3',
   'test_true_state']] = y['test_true1_ratio3'].str.split('},"', expand = True)

y[['drop6',
   'test_true1_ratio3']] = y['test_true1_ratio3'].str.split('"key":"', expand = True)
y['test_true1_ratio3'] = y['test_true1_ratio3'].str[0]

y[['test_true_type',
   'test_true1_ratiostate1',
   'test_true1_ratiostate2',
   'test_true1_ratiostate3']] = y['test_true_state'].str.split('"key":"', expand = True)

y['test_true_type'] = np.select([(y['test_true_type'].str.contains('limit', na = False)),
                                 (y['test_true_type'].str.contains('integral', na = False))],
                                ['limit', 'integral'],
                                default = 'neither')
y['test_true_type'].value_counts()

limit_only = y.loc[y['test_true_type'] == 'limit']
int_only = y.loc[y['test_true_type'] == 'integral']

limit_only['test_true2_limit1'] = limit_only['test_true1_ratiostate1'].str[0]
limit_only['test_true2_limit2'] = limit_only['test_true1_ratiostate2'].str[0]
limit_only['test_true2_limit3'] = limit_only['test_true1_ratiostate3'].str[0]

limit_only.head()


limit_only[['drop7',
   'tail_true1']] = limit_only['quiz_4_tail_plots_true_answer'].str.split('"key":"', expand = True)
limit_only[['drop8',
   'tail_true2']] = limit_only['tail_true1'].str.split('gamma', expand = True)
limit_only['tail_true1'] = limit_only['tail_true1'].str[0]
limit_only['tail_true2'] = limit_only['tail_true2'].str.replace('":', '')
limit_only['tail_true2'] = limit_only['tail_true2'].str.replace('}', '')

limit_only.filter(regex = r'true[1234]')


limit_only[['drop9',
   'geo_true1']] = limit_only['quiz_4_geometric_series_true_answer'].str.split('"key":"', expand = True)
limit_only[['geo_true1',
   'geo_true2']] = limit_only['geo_true1'].str.split('"geometric_series":', expand = True)
limit_only['geo_true1'] = limit_only['geo_true1'].str[0]
limit_only['geo_true2'] = limit_only['geo_true2'].str.replace('}', '')

limit_only.filter(regex = r'true[1234]')

limit_only[['drop10',
   'taylor_true1',
   'taylor_true2',
   'taylor_true3']] = limit_only['quiz_4_taylor_series_true_answer'].str.split('"[abc]":', expand = True)
limit_only['taylor_true1'] = limit_only['taylor_true1'].str.replace(',', '')
limit_only['taylor_true2'] = limit_only['taylor_true2'].str.replace(',', '')
limit_only['taylor_true3'] = limit_only['taylor_true3'].str.replace('}', '')

limit_only.filter(regex = r'true[1234]')


int_only.head()

int_only['test_true2_int1'] = int_only['test_true1_ratiostate1'].str[0]
int_only['test_true2_int2'] = int_only['test_true1_ratiostate2'].str[0]

int_only.head()


int_only[['drop7',
   'tail_true1']] = int_only['quiz_4_tail_plots_true_answer'].str.split('"key":"', expand = True)
int_only[['drop8',
   'tail_true2']] = int_only['tail_true1'].str.split('gamma', expand = True)
int_only['tail_true1'] = int_only['tail_true1'].str[0]
int_only['tail_true2'] = int_only['tail_true2'].str.replace('":', '')
int_only['tail_true2'] = int_only['tail_true2'].str.replace('}', '')

int_only.filter(regex = r'true[1234]')


int_only[['drop9',
   'geo_true1']] = int_only['quiz_4_geometric_series_true_answer'].str.split('"key":"', expand = True)
int_only[['geo_true1',
   'geo_true2']] = int_only['geo_true1'].str.split('"geometric_series":', expand = True)
int_only['geo_true1'] = int_only['geo_true1'].str[0]
int_only['geo_true2'] = int_only['geo_true2'].str.replace('}', '')

int_only.filter(regex = r'true[1234]')

int_only[['drop10',
   'taylor_true1',
   'taylor_true2',
   'taylor_true3']] = int_only['quiz_4_taylor_series_true_answer'].str.split('"[abc]":', expand = True)
int_only['taylor_true1'] = int_only['taylor_true1'].str.replace(',', '')
int_only['taylor_true2'] = int_only['taylor_true2'].str.replace(',', '')
int_only['taylor_true3'] = int_only['taylor_true3'].str.replace('}', '')



both.filter(regex = r'true[1234]')
limit_only.filter(regex = r'true[1234]')
int_only.filter(regex = r'true[1234]')

# --------------------- submitted answers -------------------
y.columns.tolist()

both['quiz_4_variance_submitted_answer']

both[['drop11',
   'var_submit1',
   'var_submit2']] = both['quiz_4_variance_submitted_answer'].str.split('variance_part_', expand = True)
both['var_submit1'] = both['var_submit1'].str.replace('a":', '')
both['var_submit1'] = both['var_submit1'].str.replace(',"', '')
both['var_submit2'] = both['var_submit2'].str.replace('b":', '')
both['var_submit2'] = both['var_submit2'].str.replace('}', '')

both[['drop12',
   'rate_submit1_fast',
   'rate_submit2_slow',
   'rate_submit3_2fast',
   'rate_submit4_2slow']] = both['quiz_4_convergence_rates_submitted_answer'].str.split('":"', expand = True)
both['rate_submit1_fast'] = both['rate_submit1_fast'].str[0]
both['rate_submit2_slow'] = both['rate_submit2_slow'].str[0]
both['rate_submit3_2fast'] = both['rate_submit3_2fast'].str[0]
both['rate_submit4_2slow'] = both['rate_submit4_2slow'].str[0]

both[['hosp_submit1f',
   'hosp_submit2g']] = both['quiz_4_l_hopitals_rule_submitted_answer'].str.split('","', expand = True)
both['hosp_submit1f'] = both['hosp_submit1f'].str.replace('{"f":"', '')
both['hosp_submit2g'] = both['hosp_submit2g'].str.replace('g":"', '')
both['hosp_submit2g'] = both['hosp_submit2g'].str.replace('"}', '')

both['quiz_4_convergence_tests_submitted_answer']

both[['test_submit2_limit',
      'test_submit2_int']] = both['quiz_4_convergence_tests_submitted_answer'].str.split(r'integral_statements', expand = True)
both['test_submit2_limit'] = both['test_submit2_limit'].str.replace('{"limit_statements":', '')
both[['test_submit2_limit1',
      'test_submit2_limit2',
      'test_submit2_limit3',
      'drop12']] = both['test_submit2_limit'].str.split(',', expand = True)
both['test_submit2_limit1'] = both['test_submit2_limit1'].str.replace('"', '')
both['test_submit2_limit1'] = both['test_submit2_limit1'].str.replace('[', '')
both['test_submit2_limit2'] = both['test_submit2_limit2'].str.replace('"', '')
both['test_submit2_limit2'] = both['test_submit2_limit2'].str.replace(']', '')
both['test_submit2_limit3'] = both['test_submit2_limit3'].str.replace('"', '')
both['test_submit2_limit3'] = both['test_submit2_limit3'].str.replace(']', '')

both[['test_submit2_int1',
      'test_submit2_int2',
      'test_submit2_int3']] = both['test_submit2_int'].str.split(',', expand = True)
both['test_submit2_int1'] = both['test_submit2_int1'].str.replace('":', '')
both['test_submit2_int1'] = both['test_submit2_int1'].str.replace('"', '')
both['test_submit2_int1'] = both['test_submit2_int1'].str.replace('}', '')
both['test_submit2_int1'] = both['test_submit2_int1'].str.replace('[', '')
both['test_submit2_int2'] = both['test_submit2_int2'].str.replace('"', '')
both['test_submit2_int2'] = both['test_submit2_int2'].str.replace(']}', '')
both['test_submit2_int3'] = both['test_submit2_int3'].str.replace('"', '')
both['test_submit2_int3'] = both['test_submit2_int3'].str.replace(']}', '')

both[['tail_submit1',
   'tail_submit2']] = both['quiz_4_tail_plots_submitted_answer'].str.split('","', expand = True)
both['tail_submit1'] = both['tail_submit1'].str.replace('{"ans":"', '')
both['tail_submit2'] = both['tail_submit2'].str.replace('gamma":', '')
both['tail_submit2'] = both['tail_submit2'].str.replace('}', '')

both[['geo_submit1',
   'geo_submit2']] = both['quiz_4_geometric_series_submitted_answer'].str.split('","', expand = True)
both['geo_submit1'] = both['geo_submit1'].str.replace('{"converges":"', '')
both['geo_submit2'] = both['geo_submit2'].str.replace('geometric_series":', '')
both['geo_submit2'] = both['geo_submit2'].str.replace('}', '')

both[['drop14',
   'taylor_submit1',
   'taylor_submit2',
   'taylor_submit3']] = both['quiz_4_taylor_series_submitted_answer'].str.split('":', expand = True)
both['taylor_submit1'] = both['taylor_submit1'].str.replace(r',"[a-e]', '', regex = True)
both['taylor_submit2'] = both['taylor_submit2'].str.replace(r',"[a-e]', '', regex = True)
both['taylor_submit3'] = both['taylor_submit3'].str.replace('}', '')


both.filter(regex = r'submit[12345]')



limit_only[['drop11',
   'var_submit1',
   'var_submit2']] = limit_only['quiz_4_variance_submitted_answer'].str.split('variance_part_', expand = True)
limit_only['var_submit1'] = limit_only['var_submit1'].str.replace('a":', '')
limit_only['var_submit1'] = limit_only['var_submit1'].str.replace(',"', '')
limit_only['var_submit2'] = limit_only['var_submit2'].str.replace('b":', '')
limit_only['var_submit2'] = limit_only['var_submit2'].str.replace('}', '')

limit_only[['drop12',
   'rate_submit1_fast',
   'rate_submit2_slow',
   'rate_submit3_2fast',
   'rate_submit4_2slow']] = limit_only['quiz_4_convergence_rates_submitted_answer'].str.split('":"', expand = True)
limit_only['rate_submit1_fast'] = limit_only['rate_submit1_fast'].str[0]
limit_only['rate_submit2_slow'] = limit_only['rate_submit2_slow'].str[0]
limit_only['rate_submit3_2fast'] = limit_only['rate_submit3_2fast'].str[0]
limit_only['rate_submit4_2slow'] = limit_only['rate_submit4_2slow'].str[0]

limit_only[['hosp_submit1f',
   'hosp_submit2g']] = limit_only['quiz_4_l_hopitals_rule_submitted_answer'].str.split('","', expand = True)
limit_only['hosp_submit1f'] = limit_only['hosp_submit1f'].str.replace('{"f":"', '')
limit_only['hosp_submit2g'] = limit_only['hosp_submit2g'].str.replace('g":"', '')
limit_only['hosp_submit2g'] = limit_only['hosp_submit2g'].str.replace('"}', '')

limit_only[['test_submit1_ratio1',
            'test_submit1_ratio2',
            'test_submit1_ratio3',
            'test_submit1_limit1',
            'test_submit1_limit2',
            'test_submit1_limit3']] = limit_only['quiz_4_convergence_tests_submitted_answer'].str.split('","', expand = True)
limit_only['test_submit1_ratio1'] = limit_only['test_submit1_ratio1'].str.replace('{"a_ans":"', '')
limit_only['test_submit1_ratio2'] = limit_only['test_submit1_ratio2'].str.replace('b_ans":"', '')
limit_only['test_submit1_ratio3'] = limit_only['test_submit1_ratio3'].str.replace('c_ans":"', '')
limit_only['test_submit1_limit1'] = limit_only['test_submit1_limit1'].str.replace('limit_statements":', '')
limit_only['test_submit1_limit1'] = limit_only['test_submit1_limit1'].str.replace('"', '')
limit_only['test_submit1_limit1'] = limit_only['test_submit1_limit1'].str.replace('[', '')
limit_only['test_submit1_limit1'] = limit_only['test_submit1_limit1'].str.replace('}', '')
limit_only['test_submit1_limit2'] = limit_only['test_submit1_limit2'].str.replace('"]}', '')
limit_only['test_submit1_limit3'] = limit_only['test_submit1_limit3'].str.replace('"]}', '')

limit_only[['tail_submit1',
   'tail_submit2']] = limit_only['quiz_4_tail_plots_submitted_answer'].str.split('","', expand = True)
limit_only['tail_submit1'] = limit_only['tail_submit1'].str.replace('{"ans":"', '')
limit_only['tail_submit2'] = limit_only['tail_submit2'].str.replace('gamma":', '')
limit_only['tail_submit2'] = limit_only['tail_submit2'].str.replace('}', '')

limit_only[['geo_submit1',
   'geo_submit2']] = limit_only['quiz_4_geometric_series_submitted_answer'].str.split('","', expand = True)
limit_only['geo_submit1'] = limit_only['geo_submit1'].str.replace('{"converges":"', '')
limit_only['geo_submit2'] = limit_only['geo_submit2'].str.replace('geometric_series":', '')
limit_only['geo_submit2'] = limit_only['geo_submit2'].str.replace('}', '')

limit_only[['drop14',
   'taylor_submit1',
   'taylor_submit2',
   'taylor_submit3']] = limit_only['quiz_4_taylor_series_submitted_answer'].str.split('":', expand = True)
limit_only['taylor_submit1'] = limit_only['taylor_submit1'].str.replace(r',"[a-e]', '', regex = True)
limit_only['taylor_submit2'] = limit_only['taylor_submit2'].str.replace(r',"[a-e]', '', regex = True)
limit_only['taylor_submit3'] = limit_only['taylor_submit3'].str.replace('}', '')



int_only[['drop11',
   'var_submit1',
   'var_submit2']] = int_only['quiz_4_variance_submitted_answer'].str.split('variance_part_', expand = True)
int_only['var_submit1'] = int_only['var_submit1'].str.replace('a":', '')
int_only['var_submit1'] = int_only['var_submit1'].str.replace(',"', '')
int_only['var_submit2'] = int_only['var_submit2'].str.replace('b":', '')
int_only['var_submit2'] = int_only['var_submit2'].str.replace('}', '')

int_only[['drop12',
   'rate_submit1_fast',
   'rate_submit2_slow',
   'rate_submit3_2fast',
   'rate_submit4_2slow']] = int_only['quiz_4_convergence_rates_submitted_answer'].str.split('":"', expand = True)
int_only['rate_submit1_fast'] = int_only['rate_submit1_fast'].str[0]
int_only['rate_submit2_slow'] = int_only['rate_submit2_slow'].str[0]
int_only['rate_submit3_2fast'] = int_only['rate_submit3_2fast'].str[0]
int_only['rate_submit4_2slow'] = int_only['rate_submit4_2slow'].str[0]

int_only[['hosp_submit1f',
   'hosp_submit2g']] = int_only['quiz_4_l_hopitals_rule_submitted_answer'].str.split('","', expand = True)
int_only['hosp_submit1f'] = int_only['hosp_submit1f'].str.replace('{"f":"', '')
int_only['hosp_submit2g'] = int_only['hosp_submit2g'].str.replace('g":"', '')
int_only['hosp_submit2g'] = int_only['hosp_submit2g'].str.replace('"}', '')

int_only[['test_submit1_ratio1',
            'test_submit1_ratio2',
            'test_submit1_ratio3',
            'test_submit1_int1',
            'test_submit1_int2',
            'test_submit1_int3']] = int_only['quiz_4_convergence_tests_submitted_answer'].str.split('","', expand = True)
int_only['test_submit1_ratio1'] = int_only['test_submit1_ratio1'].str.replace('{"a_ans":"', '')
int_only['test_submit1_ratio2'] = int_only['test_submit1_ratio2'].str.replace('b_ans":"', '')
int_only['test_submit1_ratio3'] = int_only['test_submit1_ratio3'].str.replace('c_ans":"', '')
int_only['test_submit1_int1'] = int_only['test_submit1_int1'].str.replace('integral_statements":', '')
int_only['test_submit1_int1'] = int_only['test_submit1_int1'].str.replace('"', '')
int_only['test_submit1_int1'] = int_only['test_submit1_int1'].str.replace('[', '')
int_only['test_submit1_int1'] = int_only['test_submit1_int1'].str.replace('}', '')
int_only['test_submit1_int2'] = int_only['test_submit1_int2'].str.replace('"]}', '')
int_only['test_submit1_int3'] = int_only['test_submit1_int3'].str.replace('"]}', '')

int_only[['tail_submit1',
   'tail_submit2']] = int_only['quiz_4_tail_plots_submitted_answer'].str.split('","', expand = True)
int_only['tail_submit1'] = int_only['tail_submit1'].str.replace('{"ans":"', '')
int_only['tail_submit2'] = int_only['tail_submit2'].str.replace('gamma":', '')
int_only['tail_submit2'] = int_only['tail_submit2'].str.replace('}', '')

int_only[['geo_submit1',
   'geo_submit2']] = int_only['quiz_4_geometric_series_submitted_answer'].str.split('","', expand = True)
int_only['geo_submit1'] = int_only['geo_submit1'].str.replace('{"converges":"', '')
int_only['geo_submit2'] = int_only['geo_submit2'].str.replace('geometric_series":', '')
int_only['geo_submit2'] = int_only['geo_submit2'].str.replace('}', '')

int_only[['drop14',
   'taylor_submit1',
   'taylor_submit2',
   'taylor_submit3']] = int_only['quiz_4_taylor_series_submitted_answer'].str.split('":', expand = True)
int_only['taylor_submit1'] = int_only['taylor_submit1'].str.replace(r',"[a-e]', '', regex = True)
int_only['taylor_submit2'] = int_only['taylor_submit2'].str.replace(r',"[a-e]', '', regex = True)
int_only['taylor_submit3'] = int_only['taylor_submit3'].str.replace('}', '')


both.filter(regex = r'submit[12345]')
limit_only.filter(regex = r'submit[12345]')
int_only.filter(regex = r'submit[12345]')

# ---------------------- true and submitted answers ----------------------

both_true = both.filter(regex = r'true[12345]').columns.tolist()
both_submit = both.filter(regex = r'submit[12345]').columns.tolist()
both_sub = both[['anon_id'] + both_true + both_submit]

both_sub = both_sub.drop(columns = ['test_true1_ratio1',
                         'test_true1_ratio2',
                         'test_true1_ratio3',
                         'test_true2_limit',
                         'test_true2_int',
                         'test_submit2_limit',
                         'test_submit2_int'])
both_sub

limit_true = limit_only.filter(regex = r'true[12345]').columns.tolist()
limit_submit = limit_only.filter(regex = r'submit[12345]').columns.tolist()
limit_sub = limit_only[['anon_id'] + limit_true + limit_submit]
limit_sub = limit_sub.drop(columns = ['test_true1_ratiostate1',
                                      'test_true1_ratiostate2',
                                      'test_true1_ratiostate3'])
limit_sub

int_true = int_only.filter(regex = r'true[12345]').columns.tolist()
int_submit = int_only.filter(regex = r'submit[12345]').columns.tolist()
int_sub = int_only[['anon_id'] + int_true + int_submit]
int_sub = int_sub.drop(columns = ['test_true1_ratiostate1',
                                      'test_true1_ratiostate2',
                                      'test_true1_ratiostate3'])
int_sub

# ---------------------- cleaning ----------------------

def blank_to_na(df):
    return df.replace(r'^\s*$', np.nan, regex=True)
  
both_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']] = both_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']].astype(float)

both_sub = blank_to_na(both_sub)

limit_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']] = limit_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']].astype(float)

limit_sub = blank_to_na(limit_sub)

int_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']] = int_sub[['var_true1', 'var_true2', 'tail_true2', 'geo_true2', 'taylor_true1', 'taylor_true2', 'taylor_true3', 'var_submit1', 'var_submit2', 'tail_submit2', 'geo_submit2', 'taylor_submit1', 'taylor_submit2', 'taylor_submit3']].astype(float)

int_sub = blank_to_na(int_sub)

both_sub.columns
limit_sub.shape
int_sub.shape

# ---------------------- true and submitted answers ----------------------
# both
both_sub['item1'] = np.where(both_sub['var_submit1'] == both_sub['var_true1'], 1, 0)
both_sub['item2'] = np.where(both_sub['var_submit2'] == both_sub['var_true2'], 1, 0)
both_sub['item3'] = np.where(both_sub['rate_submit1_fast'] == both_sub['rate_true1_fast'], 1, 0)
both_sub['item4'] = np.where(both_sub['rate_submit2_slow'] == both_sub['rate_true2_slow'], 1, 0)
both_sub['item5'] = np.where(both_sub['rate_submit3_2fast'] == both_sub['rate_true3_2fast'], 1, 0)
both_sub['item6'] = np.where(both_sub['rate_submit4_2slow'] == both_sub['rate_true4_2slow'], 1, 0)
both_sub['item7'] = np.where(both_sub['hosp_submit1f'] == both_sub['hosp_true1f'], 1, 0)
both_sub['item8'] = np.where(both_sub['hosp_submit2g'] == both_sub['hosp_true2g'], 1, 0)
both_sub['item9'] = np.select([(both_sub['test_submit2_int1'] == both_sub['test_true2_int1']),
                               (both_sub['test_submit2_int1'] == both_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
both_sub['item10'] = np.select([(both_sub['test_submit2_int2'] == both_sub['test_true2_int1']),
                               (both_sub['test_submit2_int2'] == both_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
both_sub['item11'] = np.select([(both_sub['test_submit2_int3'] == both_sub['test_true2_int1']),
                               (both_sub['test_submit2_int3'] == both_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
both_sub['test_submit2_int4'] = np.nan
both_sub['test_true2_int4'] = np.nan
both_sub['item12'] = np.where(both_sub['test_submit2_int4'].isnull() & both_sub['test_true2_int4'].isnull(), 1, 0)

both_sub['item13'] = np.select([(both_sub['test_submit2_limit1'] == both_sub['test_true2_limit1']),
                               (both_sub['test_submit2_limit1'] == both_sub['test_true2_limit2']),
                               (both_sub['test_submit2_limit1'] == both_sub['test_true2_limit3'])],
                              [1, 1, 1],
                              default = 0)
both_sub['item14'] = np.select([(both_sub['test_submit2_limit2'] == both_sub['test_true2_limit1']),
                               (both_sub['test_submit2_limit2'] == both_sub['test_true2_limit2']),
                               (both_sub['test_submit2_limit2'] == both_sub['test_true2_limit3']),
                               (both_sub['test_submit2_limit2'].isnull() & both_sub['test_true2_limit2'].isnull())],
                              [1, 1, 1, 1],
                              default = 0)
both_sub['item15'] = np.select([(both_sub['test_submit2_limit3'] == both_sub['test_true2_limit1']),
                               (both_sub['test_submit2_limit3'] == both_sub['test_true2_limit2']),
                               (both_sub['test_submit2_limit3'] == both_sub['test_true2_limit3']),
                               (both_sub['test_submit2_limit3'].isnull() & both_sub['test_true2_limit3'].isnull())],
                              [1, 1, 1, 1],
                              default = 0)
both_sub['test_submit2_limit4'] = np.nan
both_sub['test_true2_limit4'] = np.nan
both_sub['test_submit2_limit5'] = np.nan
both_sub['test_true2_limit5'] = np.nan
both_sub['item16'] = np.where(both_sub['test_submit2_limit4'].isnull() & both_sub['test_true2_limit4'].isnull(), 1, 0)
both_sub['item17'] = np.where(both_sub['test_submit2_limit5'].isnull() & both_sub['test_true2_limit5'].isnull(), 1, 0)

both_sub['item21'] = np.where(both_sub['tail_submit1'] == both_sub['tail_true1'], 1, 0)
both_sub['item22'] = np.where(both_sub['tail_submit2'] == both_sub['tail_true2'], 1, 0)
both_sub['item23'] = np.where(both_sub['geo_submit1'] == both_sub['geo_true1'], 1, 0)
both_sub['item24'] = np.where(both_sub['geo_submit2'] == both_sub['geo_true2'], 1, 0)
both_sub['item25'] = np.where(both_sub['taylor_submit1'] == both_sub['taylor_true1'], 1, 0)
both_sub['item26'] = np.where(both_sub['taylor_submit2'] == both_sub['taylor_true2'], 1, 0)
both_sub['item27'] = np.where(both_sub['taylor_submit3'] == both_sub['taylor_true3'], 1, 0)

[both_sub[i].value_counts() for i in both_sub.filter(regex = 'item').columns]


# limit only data
limit_sub['item1'] = np.where(limit_sub['var_submit1'] == limit_sub['var_true1'], 1, 0)
limit_sub['item2'] = np.where(limit_sub['var_submit2'] == limit_sub['var_true2'], 1, 0)
limit_sub['item3'] = np.where(limit_sub['rate_submit1_fast'] == limit_sub['rate_true1_fast'], 1, 0)
limit_sub['item4'] = np.where(limit_sub['rate_submit2_slow'] == limit_sub['rate_true2_slow'], 1, 0)
limit_sub['item5'] = np.where(limit_sub['rate_submit3_2fast'] == limit_sub['rate_true3_2fast'], 1, 0)
limit_sub['item6'] = np.where(limit_sub['rate_submit4_2slow'] == limit_sub['rate_true4_2slow'], 1, 0)
limit_sub['item7'] = np.where(limit_sub['hosp_submit1f'] == limit_sub['hosp_true1f'], 1, 0)
limit_sub['item8'] = np.where(limit_sub['hosp_submit2g'] == limit_sub['hosp_true2g'], 1, 0)

limit_sub['item13'] = np.select([(limit_sub['test_submit1_limit1'] == limit_sub['test_true2_limit1']),
                               (limit_sub['test_submit1_limit1'] == limit_sub['test_true2_limit2']),
                               (limit_sub['test_submit1_limit1'] == limit_sub['test_true2_limit3'])],
                              [1, 1, 1],
                              default = 0)
limit_sub['item14'] = np.select([(limit_sub['test_submit1_limit2'] == limit_sub['test_true2_limit1']),
                               (limit_sub['test_submit1_limit2'] == limit_sub['test_true2_limit2']),
                               (limit_sub['test_submit1_limit2'] == limit_sub['test_true2_limit3']),
                               (limit_sub['test_submit1_limit2'].isnull() & limit_sub['test_true2_limit2'].isnull())],
                              [1, 1, 1, 1],
                              default = 0)
limit_sub['item15'] = np.select([(limit_sub['test_submit1_limit3'] == limit_sub['test_true2_limit1']),
                               (limit_sub['test_submit1_limit3'] == limit_sub['test_true2_limit2']),
                               (limit_sub['test_submit1_limit3'] == limit_sub['test_true2_limit3']),
                               (limit_sub['test_submit1_limit3'].isnull() & limit_sub['test_true2_limit3'].isnull())],
                              [1, 1, 1, 1],
                              default = 0)
limit_sub['test_submit2_limit4'] = np.nan
limit_sub['test_true2_limit4'] = np.nan
limit_sub['test_submit2_limit5'] = np.nan
limit_sub['test_true2_limit5'] = np.nan
limit_sub['item16'] = np.where(limit_sub['test_submit2_limit4'].isnull() & limit_sub['test_true2_limit4'].isnull(), 1, 0)
limit_sub['item17'] = np.where(limit_sub['test_submit2_limit5'].isnull() & limit_sub['test_true2_limit5'].isnull(), 1, 0)

limit_sub['item18'] = np.where(limit_sub['test_submit1_ratio1'] == limit_sub['test_true1_ratio1'], 1, 0)
limit_sub['item19'] = np.where(limit_sub['test_submit1_ratio2'] == limit_sub['test_true1_ratio2'], 1, 0)
limit_sub['item20'] = np.where(limit_sub['test_submit1_ratio3'] == limit_sub['test_true1_ratio3'], 1, 0)

# limit_sub['test_submit2_int4'] = np.nan
# limit_sub['test_true2_int4'] = np.nan
# limit_sub['item12'] = np.where(limit_sub['test_submit2_int4'].isnull() & limit_sub['test_true2_int4'].isnull(), 1, 0)

limit_sub['item21'] = np.where(limit_sub['tail_submit1'] == limit_sub['tail_true1'], 1, 0)
limit_sub['item22'] = np.where(limit_sub['tail_submit2'] == limit_sub['tail_true2'], 1, 0)
limit_sub['item23'] = np.where(limit_sub['geo_submit1'] == limit_sub['geo_true1'], 1, 0)
limit_sub['item24'] = np.where(limit_sub['geo_submit2'] == limit_sub['geo_true2'], 1, 0)
limit_sub['item25'] = np.where(limit_sub['taylor_submit1'] == limit_sub['taylor_true1'], 1, 0)
limit_sub['item26'] = np.where(limit_sub['taylor_submit2'] == limit_sub['taylor_true2'], 1, 0)
limit_sub['item27'] = np.where(limit_sub['taylor_submit3'] == limit_sub['taylor_true3'], 1, 0)

[limit_sub[i].value_counts() for i in limit_sub.filter(regex = 'item').columns]


# integral only data
int_sub['item1'] = np.where(int_sub['var_submit1'] == int_sub['var_true1'], 1, 0)
int_sub['item2'] = np.where(int_sub['var_submit2'] == int_sub['var_true2'], 1, 0)
int_sub['item3'] = np.where(int_sub['rate_submit1_fast'] == int_sub['rate_true1_fast'], 1, 0)
int_sub['item4'] = np.where(int_sub['rate_submit2_slow'] == int_sub['rate_true2_slow'], 1, 0)
int_sub['item5'] = np.where(int_sub['rate_submit3_2fast'] == int_sub['rate_true3_2fast'], 1, 0)
int_sub['item6'] = np.where(int_sub['rate_submit4_2slow'] == int_sub['rate_true4_2slow'], 1, 0)
int_sub['item7'] = np.where(int_sub['hosp_submit1f'] == int_sub['hosp_true1f'], 1, 0)
int_sub['item8'] = np.where(int_sub['hosp_submit2g'] == int_sub['hosp_true2g'], 1, 0)

int_sub['item9'] = np.select([(int_sub['test_submit1_int1'] == int_sub['test_true2_int1']),
                               (int_sub['test_submit1_int1'] == int_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
int_sub['item10'] = np.select([(int_sub['test_submit1_int2'] == int_sub['test_true2_int1']),
                               (int_sub['test_submit1_int2'] == int_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
int_sub['item11'] = np.select([(int_sub['test_submit1_int3'] == int_sub['test_true2_int1']),
                               (int_sub['test_submit1_int3'] == int_sub['test_true2_int2'])],
                              [1, 1],
                              default = 0)
int_sub['test_submit2_int4'] = np.nan
int_sub['test_true2_int4'] = np.nan
int_sub['item12'] = np.where(int_sub['test_submit2_int4'].isnull() & int_sub['test_true2_int4'].isnull(), 1, 0)

int_sub['item18'] = np.where(int_sub['test_submit1_ratio1'] == int_sub['test_true1_ratio1'], 1, 0)
int_sub['item19'] = np.where(int_sub['test_submit1_ratio2'] == int_sub['test_true1_ratio2'], 1, 0)
int_sub['item20'] = np.where(int_sub['test_submit1_ratio3'] == int_sub['test_true1_ratio3'], 1, 0)


# int_sub['test_submit2_limit4'] = np.nan
# int_sub['test_true2_limit4'] = np.nan
# int_sub['test_submit2_limit5'] = np.nan
# int_sub['test_true2_limit5'] = np.nan
# int_sub['item16'] = np.where(int_sub['test_submit2_limit4'].isnull() & int_sub['test_true2_limit4'].isnull(), 1, 0)
# int_sub['item17'] = np.where(int_sub['test_submit2_limit5'].isnull() & int_sub['test_true2_limit5'].isnull(), 1, 0)

int_sub['item21'] = np.where(int_sub['tail_submit1'] == int_sub['tail_true1'], 1, 0)
int_sub['item22'] = np.where(int_sub['tail_submit2'] == int_sub['tail_true2'], 1, 0)
int_sub['item23'] = np.where(int_sub['geo_submit1'] == int_sub['geo_true1'], 1, 0)
int_sub['item24'] = np.where(int_sub['geo_submit2'] == int_sub['geo_true2'], 1, 0)
int_sub['item25'] = np.where(int_sub['taylor_submit1'] == int_sub['taylor_true1'], 1, 0)
int_sub['item26'] = np.where(int_sub['taylor_submit2'] == int_sub['taylor_true2'], 1, 0)
int_sub['item27'] = np.where(int_sub['taylor_submit3'] == int_sub['taylor_true3'], 1, 0)

[int_sub[i].value_counts() for i in int_sub.filter(regex = 'item').columns]



both_sub2 = both_sub.filter(regex = r'item')
limit_sub2 = limit_sub.filter(regex = r'item')
int_sub2 = int_sub.filter(regex = r'item')

both_sub = both_sub[['anon_id'] + both_sub2.columns.tolist()]
limit_sub = limit_sub[['anon_id'] + limit_sub2.columns.tolist()]
int_sub = int_sub[['anon_id'] + int_sub2.columns.tolist()]

both_sub['type'] = 'both'
limit_sub['type'] = 'limit_ratio'
int_sub['type'] = 'integral_ratio'

y = pd.concat([both_sub, limit_sub, int_sub])
# y.to_csv(here('data/quiz_data/quiz4_ready_irt_separate_item.csv'))


y_item = y.filter(regex = 'item')

q = pd.read_csv(here('data/q_matrix/q4_granular.csv')).drop(columns=['Unnamed: 0'])

# only using retake data for 
stan_dict = {
  'J': y_item.shape[0],
  'I': y_item.shape[1],
  'C': alpha.shape[0],
  'K': q.shape[1],
  'Y': np.array(y_item),
  'Q': np.array(q),
  'alpha': np.array(alpha)
}

# np.mean(np.random.beta(30, 20, 200))
# np.mean(np.random.beta(10, 40, 200))

dcm_file = os.path.join(here(f'quiz_models/quiz4_model.stan'))
dcm_model = CmdStanModel(stan_file = dcm_file,
                         cpp_options={'STAN_THREADS': 'TRUE'})

np.random.seed(12345)
dcm_fit = dcm_model.sample(data = stan_dict,
                        show_console = True,
                        chains = 4,
                        adapt_delta = .95,
                        iter_warmup = 2000,
                        iter_sampling = 2000)
dcm_diagnose = pd.DataFrame(dcm_fit.summary())


dcm_prior_file = os.path.join(here(f'quiz_models/quiz4_model_prior_only.stan'))
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


dcm_diagnose.to_csv(here(f'diagnostics/quiz4_model.csv'))
(
  joblib.dump([dcm_model, dcm_fit],
              here(f'joblib_models/quiz4_modfit.joblib'),
              compress = 3)
)

dcm_prior_diagnose.to_csv(here(f'diagnostics/quiz4_model_prior_only.csv'))
(
  joblib.dump([dcm_prior_model, dcm_prior_fit],
              here(f'joblib_models/quiz4_modfit_prior_only.joblib'),
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

az.plot_trace(idcm, var_names = ['nu'])
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

# attr_mast.to_csv(here('student_data/attr_mastery_quiz2.csv'))


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
ydcm_wide.columns = ['stu', 'draw', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'item7', 'item8', 'item9', 'item10', 'item11', 'item12', 'item13', 'item14', 'item15', 'item16']

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