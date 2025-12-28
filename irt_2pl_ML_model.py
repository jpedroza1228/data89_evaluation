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

os.environ['QT_API'] = 'PyQt6'

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

link = 'https://raw.githubusercontent.com/jpedroza1228/projects_portfolio_and_practice/refs/heads/main/projects/dcm/lcdm_py/ecpe_data.csv'

import requests
from io import StringIO

response = requests.get(link, verify = False)
y = pd.read_csv(StringIO(response.text)).clean_names(case_type = 'snake')
# y = pd.read_csv(link)

np.random.seed(12345)
y = y.sample(n = 200)

# y_tf = np.where(y == 1, True, False)
# y_tf = pd.DataFrame(y_tf)
# items x participant
y_tf = y.transpose()


from girth import twopl_mml

estimates = twopl_mml(np.array(y_tf))

# Unpack estimates
pd.DataFrame({
  'item': np.arange(1, y_tf.shape[0] + 1),
  'discrimination': estimates['Discrimination'],
  'difficulty': estimates['Difficulty']
})
