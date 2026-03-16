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

check = pd.read_csv(here('data/checkin/checkin2.csv'), skiprows = 2).clean_names(case_type = 'snake')
check = check.loc[(check['{"_import_id"_"status"}'] == 'IP Address')]

check = check.loc[:, '{"_import_id"_"qid3"}':]

check.columns = ['retake_quiz', 
                 'understand_logic', 
                 'understand_rules', 
                 'understand_bounds', 
                 'understand_prob_types', 
                 'understand_compute', 
                 'to_do_success', 
                 'capable_do_success', 
                 'understand_random_var_dist',
                 'understand_compute_vis_dist',
                 'understand_specific_mod',
                 'understand_cont_mod']

to_do_cond = [
  (check['to_do_success'] == 'Strongly Disagree'),
  (check['to_do_success'] == 'Disagree'),
  (check['to_do_success'] == 'Slightly Disagree'),
  (check['to_do_success'] == 'Slightly Agree'),
  (check['to_do_success'] == 'Agree'),
  (check['to_do_success'] == 'Strongly Agree')
]

capable_cond = [
  (check['capable_do_success'] == 'Strongly Disagree'),
  (check['capable_do_success'] == 'Disagree'),
  (check['capable_do_success'] == 'Slightly Disagree'),
  (check['capable_do_success'] == 'Slightly Agree'),
  (check['capable_do_success'] == 'Agree'),
  (check['capable_do_success'] == 'Strongly Agree')
]

success_choice = [1, 2, 3, 4, 5, 6]

check['num_to_do_success'] = np.select(to_do_cond, success_choice, default = 0)
check['num_capable_do_success'] = np.select(capable_cond, success_choice, default = 0)

check['to_do_success'] = pd.Categorical(check['to_do_success'], ordered = True, categories = ['Strongly Disagree', 'Disagree', 'Slightly Disagree', 'Slightly Agree', 'Agree', 'Strongly Agree'])

check['capable_do_success'] = pd.Categorical(check['capable_do_success'], ordered = True, categories = ['Strongly Disagree', 'Disagree', 'Slightly Disagree', 'Slightly Agree', 'Agree', 'Strongly Agree'])

check['success_na'] = np.where(check['to_do_success'].isna(), 1, 0)
check['capable_na'] = np.where(check['capable_do_success'].isna(), 1, 0)

check['success_na'] = pd.Categorical(check['success_na'])
check['capable_na'] = pd.Categorical(check['capable_na'])

pn.ggplot.show(
  pn.ggplot(check,
            pn.aes('to_do_success'))
  + pn.geom_bar(pn.aes(fill = 'success_na'),
                color = 'black')
  + pn.coord_flip()
  + pn.scale_y_continuous(limits = [0, 10],
                          breaks = np.arange(0, 11, 1))
  + pn.labs(title = 'I know what I would need to do in order to stay on track...')
)

pn.ggplot.show(
  pn.ggplot(check,
            pn.aes('capable_do_success'))
  + pn.geom_bar(pn.aes(fill = 'capable_na'),
                color = 'black')
  + pn.coord_flip()
  + pn.scale_y_continuous(limits = [0, 10],
                          breaks = np.arange(0, 11, 1))
  + pn.labs(title = 'I am capable of doing what I need to do in order to stay on track...')
)

check.filter(regex = '^num').melt().groupby('variable')['value'].agg(['mean', 'std']).round(3)

check['retake_quiz'].value_counts()

(
  check
  .loc[check['retake_quiz'] == 'Yes',
       ['understand_logic', 
        'understand_rules', 
        'understand_bounds', 
        'understand_prob_types', 
        'understand_compute']]
  .melt()
  .groupby('variable')['value']
  .agg(['mean', 'std'])
  .round(3)
)


(
  check
  .loc[check['retake_quiz'] == 'Yes',
       ['understand_random_var_dist',
       'understand_compute_vis_dist',
       'understand_specific_mod',
       'understand_cont_mod']]
  .melt()
  .groupby('variable')['value']
  .agg(['mean', 'std'])
  .round(3)
)

