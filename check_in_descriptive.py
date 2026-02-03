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

pre = pd.read_csv(here('data/pre/pre.csv'), skiprows = 2).clean_names(case_type = 'snake')

clean = pre.dropna()

clean.columns = ['date_time', 'to_do_success', 'capable_do_success', 'understand_logic', 'understand_rules', 'understand_bounds', 'understand_prob_types', 'understand_compute', 'mseaq1', 'mseaq2', 'mseaq3', 'mseaq4', 'mseaq5', 'mseaq6', 'mseaq7']

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

clean = clean[['to_do_success', 'num_to_do_success', 'capable_do_success', 'num_capable_do_success', 'understand_logic', 'understand_rules', 'understand_bounds', 'understand_prob_types', 'understand_compute']]

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

pn.ggplot.show(
  pn.ggplot(clean,
            pn.aes('capable_do_success'))
  + pn.geom_histogram()
  + pn.labs(title = 'I am capable of doing what I need to do in order to stay on track...')
)

clean_long = clean.drop(columns = {'to_do_success', 'num_to_do_success', 'capable_do_success', 'num_capable_do_success'}).melt()

clean_long.groupby('variable')['value'].agg(['mean', 'std']).round(2)

pn.ggplot.show(
  pn.ggplot(clean_long,
            pn.aes('value'))
  + pn.geom_histogram(color = 'black',
                      fill = jpcolor)
  + pn.facet_wrap('variable')
)

clean_success = clean.filter(regex = '^num').melt()

clean_success.groupby('variable').agg(['mean', 'std']).round(2)