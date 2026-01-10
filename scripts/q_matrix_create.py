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

os.environ['QT_API'] = 'PyQt6'

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

here()

q1 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q1.csv'))
)

q2 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q2.csv'))
)

q3 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q3.csv'))
)

q4 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q4.csv'))
)

q5 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q5.csv'))
)

q6 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q6.csv'))
)

q7 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q7.csv'))
)

q8 = (
  pd
  .DataFrame({'hold1': [],
              'hold2': []})
  .to_csv(here('data/q_matrix/q8.csv'))
)