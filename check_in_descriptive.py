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

pre = pd.read_csv(here('data/pre/pre_label.csv'), skiprows = 2).clean_names(case_type = 'snake')

pre.columns = ['start_date', 'end_date', 'status', 'ip_address', 'progress', 'duration', 'finished', 'record_date', 'record_id', 'drop1', 'drop2', 'drop3', 'drop4', 'drop5', 'drop6', 'drop7', 'drop8', 'to_do_success', 'capable_do_success', 'understand_logic', 'understand_rules', 'understand_bounds', 'understand_prob_types', 'understand_compute', 'mseaq1', 'mseaq2', 'mseaq3', 'mseaq4', 'mseaq5', 'mseaq6', 'mseaq7', 'mseaq8', 'mseaq9', 'mseaq10', 'mseaq11', 'mseaq12', 'mseaq13', 'mseaq14', 'maeaq15']

pre_drop = pre.filter(regex = '^drop').columns

pre = pre.drop(columns = pre_drop)

clean = pre.loc[~pre['ip_address'].isna()]

clean = clean.dropna()

to_do_cond = [
  (clean['to_do_success'] == 'Strongly Disagree'),
  (clean['to_do_success'] == 'Disagree'),
  (clean['to_do_success'] == 'Slightly Disagree'),
  (clean['to_do_success'] == 'Slightly Agree'),
  (clean['to_do_success'] == 'Agree'),
  (clean['to_do_success'] == 'Strongly Agree')
]

capable_cond = [
  (clean['capable_do_success'] == 'Strongly Disagree'),
  (clean['capable_do_success'] == 'Disagree'),
  (clean['capable_do_success'] == 'Slightly Disagree'),
  (clean['capable_do_success'] == 'Slightly Agree'),
  (clean['capable_do_success'] == 'Agree'),
  (clean['capable_do_success'] == 'Strongly Agree')
]

success_choice = [1, 2, 3, 4, 5, 6]

clean['num_to_do_success'] = np.select(to_do_cond, success_choice, default = 0)
clean['num_capable_do_success'] = np.select(capable_cond, success_choice, default = 0)

clean['to_do_success'] = pd.Categorical(clean['to_do_success'], ordered = True, categories = ['Strongly Disagree', 'Disagree', 'Slightly Disagree', 'Slightly Agree', 'Agree', 'Strongly Agree'])

clean['capable_do_success'] = pd.Categorical(clean['capable_do_success'], ordered = True, categories = ['Strongly Disagree', 'Disagree', 'Slightly Disagree', 'Slightly Agree', 'Agree', 'Strongly Agree'])

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('to_do_success'))
  + pn.geom_bar(color = 'black',
                fill = jpcolor)
  + pn.labs(title = 'I know what I would need to do in order to stay on track...')
)

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('capable_do_success'))
  + pn.geom_bar(color = 'black',
                fill = jpcolor)
  + pn.labs(title = 'I am capable of doing what I need to do in order to stay on track...')
)

clean.filter(regex = '^num').melt().groupby('variable')['value'].agg(['mean', 'std']).round(3)

clean.filter(regex = '^understand').melt().groupby('variable')['value'].agg(['mean', 'std']).round(3)

mseaq1_cond = [(clean['mseaq1'] == 'Never'),
  (clean['mseaq1'] == 'Seldom'),
  (clean['mseaq1'] == 'Sometimes'),
  (clean['mseaq1'] == 'Often'),
  (clean['mseaq1'] == 'Usually')
  ]

mseaq2_cond = [(clean['mseaq2'] == 'Never'),
  (clean['mseaq2'] == 'Seldom'),
  (clean['mseaq2'] == 'Sometimes'),
  (clean['mseaq2'] == 'Often'),
  (clean['mseaq2'] == 'Usually')
  ]

mseaq3_cond = [(clean['mseaq3'] == 'Never'),
  (clean['mseaq3'] == 'Seldom'),
  (clean['mseaq3'] == 'Sometimes'),
  (clean['mseaq3'] == 'Often'),
  (clean['mseaq3'] == 'Usually')
  ]

mseaq4_cond = [(clean['mseaq4'] == 'Never'),
  (clean['mseaq4'] == 'Seldom'),
  (clean['mseaq4'] == 'Sometimes'),
  (clean['mseaq4'] == 'Often'),
  (clean['mseaq4'] == 'Usually')
  ]

mseaq5_cond = [(clean['mseaq5'] == 'Never'),
  (clean['mseaq5'] == 'Seldom'),
  (clean['mseaq5'] == 'Sometimes'),
  (clean['mseaq5'] == 'Often'),
  (clean['mseaq5'] == 'Usually')
  ]

mseaq6_cond = [(clean['mseaq6'] == 'Never'),
  (clean['mseaq6'] == 'Seldom'),
  (clean['mseaq6'] == 'Sometimes'),
  (clean['mseaq6'] == 'Often'),
  (clean['mseaq6'] == 'Usually')
  ]

mseaq7_cond = [(clean['mseaq7'] == 'Never'),
  (clean['mseaq7'] == 'Seldom'),
  (clean['mseaq7'] == 'Sometimes'),
  (clean['mseaq7'] == 'Often'),
  (clean['mseaq7'] == 'Usually')
  ]

mseaq8_cond = [(clean['mseaq8'] == 'Never'),
  (clean['mseaq8'] == 'Seldom'),
  (clean['mseaq8'] == 'Sometimes'),
  (clean['mseaq8'] == 'Often'),
  (clean['mseaq8'] == 'Usually')
  ]

mseaq9_cond = [(clean['mseaq9'] == 'Never'),
  (clean['mseaq9'] == 'Seldom'),
  (clean['mseaq9'] == 'Sometimes'),
  (clean['mseaq9'] == 'Often'),
  (clean['mseaq9'] == 'Usually')
  ]

mseaq10_cond = [(clean['mseaq10'] == 'Never'),
  (clean['mseaq10'] == 'Seldom'),
  (clean['mseaq10'] == 'Sometimes'),
  (clean['mseaq10'] == 'Often'),
  (clean['mseaq10'] == 'Usually')
  ]

mseaq11_cond = [(clean['mseaq11'] == 'Never'),
  (clean['mseaq11'] == 'Seldom'),
  (clean['mseaq11'] == 'Sometimes'),
  (clean['mseaq11'] == 'Often'),
  (clean['mseaq11'] == 'Usually')
  ]

mseaq12_cond = [(clean['mseaq12'] == 'Never'),
  (clean['mseaq12'] == 'Seldom'),
  (clean['mseaq12'] == 'Sometimes'),
  (clean['mseaq12'] == 'Often'),
  (clean['mseaq12'] == 'Usually')
  ]

mseaq13_cond = [(clean['mseaq13'] == 'Never'),
  (clean['mseaq13'] == 'Seldom'),
  (clean['mseaq13'] == 'Sometimes'),
  (clean['mseaq13'] == 'Often'),
  (clean['mseaq13'] == 'Usually')
  ]

mseaq14_cond = [(clean['mseaq14'] == 'Never'),
  (clean['mseaq14'] == 'Seldom'),
  (clean['mseaq14'] == 'Sometimes'),
  (clean['mseaq14'] == 'Often'),
  (clean['mseaq14'] == 'Usually')
  ]

mseaq15_cond = [(clean['maeaq15'] == 'Never'),
  (clean['maeaq15'] == 'Seldom'),
  (clean['maeaq15'] == 'Sometimes'),
  (clean['maeaq15'] == 'Often'),
  (clean['maeaq15'] == 'Usually')
  ]

mseaq_choice = [0, 1, 2, 3, 4]
reverse_choice = [4, 3, 2, 1, 0]

clean['mseaq1'] = np.select(mseaq1_cond, mseaq_choice, default = 0)
clean['mseaq2'] = np.select(mseaq2_cond, reverse_choice, default = 0)
clean['mseaq3'] = np.select(mseaq3_cond, mseaq_choice, default = 0)
clean['mseaq4'] = np.select(mseaq4_cond, reverse_choice, default = 0)
clean['mseaq5'] = np.select(mseaq5_cond, mseaq_choice, default = 0)
clean['mseaq6'] = np.select(mseaq6_cond, mseaq_choice, default = 0)
clean['mseaq7'] = np.select(mseaq7_cond, mseaq_choice, default = 0)
clean['mseaq8'] = np.select(mseaq8_cond, reverse_choice, default = 0)
clean['mseaq9'] = np.select(mseaq9_cond, mseaq_choice, default = 0)
clean['mseaq10'] = np.select(mseaq10_cond, mseaq_choice, default = 0)
clean['mseaq11'] = np.select(mseaq11_cond, mseaq_choice, default = 0)
clean['mseaq12'] = np.select(mseaq12_cond, mseaq_choice, default = 0)
clean['mseaq13'] = np.select(mseaq13_cond, mseaq_choice, default = 0)
clean['mseaq14'] = np.select(mseaq14_cond, reverse_choice, default = 0)
clean['mseaq15'] = np.select(mseaq15_cond, reverse_choice, default = 0)

# higher score means more confidence in test taking
clean['test_anxiety'] = clean[['mseaq2', 'mseaq3', 'mseaq4', 'mseaq10', 'mseaq14']].sum(axis = 1)

# higher scores means more confidence in asking questions
clean['in_class'] = clean[['mseaq1', 'mseaq8', 'mseaq15']].sum(axis = 1)

# higher scores means more self-efficacy
clean['self_efficacy'] = clean[['mseaq5', 'mseaq6', 'mseaq7', 'mseaq9', 'mseaq11', 'mseaq12', 'mseaq13']].sum(axis = 1)

clean['test_anxiety'].agg(['mean', 'std', 'min', 'max']).round(3)

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('test_anxiety'))
  + pn.geom_histogram(color = 'black',
                      fill = 'seagreen',
                      bins = 11)
  + pn.labs(
    title = 'Higher scores indicate more confidence in test taking',
    caption = 'test anxiety (M = 7.7, SD = 3.48, Min = 1, Max = 16)'
  )
)

clean['in_class'].agg(['mean', 'std', 'min', 'max']).round(3)

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('in_class'))
  + pn.geom_histogram(color = 'black',
                      fill = 'seagreen',
                      bins = 11)
  + pn.labs(
    title = 'Higher scores indicate more confidence in asking questions',
    caption = 'test anxiety (M = 5.56, SD = 2.07, Min = 1, Max = 11)'
  )
)

clean['self_efficacy'].agg(['mean', 'std', 'min', 'max']).round(3)

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('self_efficacy'))
  + pn.geom_histogram(color = 'black',
                      fill = 'seagreen',
                      bins = 15)
  + pn.labs(
    title = 'Higher scores indicate greater self-efficacy',
    caption = 'test anxiety (M = 15.94, SD = 5.35, Min = 4, Max = 28)'
  )
)
