# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from dotenv import load_dotenv, find_dotenv
dot_env = find_dotenv()
if dot_env == '': dot_env = find_dotenv(filename='dot.env')
load_dotenv(dot_env, override=True)

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

from urllib.request import urlopen
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import json
import datetime
import requests
import json
import math
import os
import seaborn as sns
import glob
from dateutil.parser import parse as parsedate
import functools
import matplotlib.colors as colors
#from types import SimpleNamespace

if 'KNACK_APP_ID' in os.environ:
  from knack import Knack
  knack = Knack(app_id = os.environ['KNACK_APP_ID'], api_key = os.environ['KNACK_API_KEY'])
else:
  knack = None

def run(*args):
  if len(args) == 1 and callable(args[0]):
    return args[0]()
  else:
    print(*args)
    return lambda func: func()

# https://www.cbs.nl/nl-nl/onze-diensten/open-data/open-data-v4/snelstartgids-odata-v4
def get_odata(url):
  data = pd.DataFrame()
  top = 100
  skip = 0
  sep = '?' if not '?' in url else '&'
  while True:
    r = requests.get(f'{url}{sep}$top={top}&$skip={skip}').json()
    if len(r['value']) == 0:
      break
    data = data.append(pd.DataFrame(r['value']))
    skip += top
  return data

