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
  while url:
    print(url)
    r = requests.get(url)
    try:
      r = r.json()
    except json.JSONDecodeError:
      raise ValueError(r.content.decode('utf-8'))

    data = data.append(pd.DataFrame(r['value']))

    if '@odata.nextLink' in r:
      url = r['@odata.nextLink']
    else:
      url = None
  return data

def rivm_cijfers(naam, n=0):
  os.makedirs('downloads', exist_ok = True)
  url = f'https://data.rivm.nl/covid-19/{naam}.csv'
  rivm = requests.head(url)
  latest = os.path.join('downloads', parsedate(rivm.headers['last-modified']).strftime(naam + '-%Y-%m-%d@%H-%M.csv'))
  if not os.path.exists(latest):
    print('downloading', latest)
    urlretrieve(url, latest)
  elif n == 0:
    print(latest, 'exists')
  history = sorted(glob.glob(os.path.join('downloads', f'{naam}*.csv')), reverse=True)
  for f in history[7:]:
    print('removing', f)
    os.remove(f)
  for f in history:
    if os.path.exists(f):
      print('keeping', f)
  print('loading', history[n])
  return pd.read_csv(history[n], sep=';', header=0 )
