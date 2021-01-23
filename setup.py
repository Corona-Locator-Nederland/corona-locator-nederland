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
import gzip
import shutil
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
  os.makedirs('rivm', exist_ok = True)
  url = f'https://data.rivm.nl/covid-19/{naam}.csv'
  rivm = requests.head(url)
  latest = os.path.join('rivm', parsedate(rivm.headers['last-modified']).strftime(naam + '-%Y-%m-%d@%H-%M.csv'))
  if not os.path.exists(latest) and not os.path.exists(latest + '.gz'):
    print('downloading', latest)
    urlretrieve(url, latest)
  elif n == 0:
    print(latest, 'exists')

  if 'CI' in os.environ:
    for f in glob.glob(os.path.join('rivm', f'{naam}*.csv')):
      with open(f, 'rb') as f_in, gzip.open(f + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        os.remove(f)

  history = sorted(glob.glob(os.path.join('rivm', f'{naam}*.csv*')), reverse=True)
  history = [f for f in history if not f + '.gz' in history]
  print('loading', history[n])
  return pd.read_csv(history[n], sep=';', header=0 )

def cbs_bevolking():
  def roundup(x):
    return int(math.ceil(x / 10.0)) * 10
  def rounddown(x):
    return int(math.floor(x / 10.0)) * 10

  cbs = 'https://opendata.cbs.nl/ODataApi/OData/83482NED'

  leeftijden = get_odata(cbs + "/Leeftijd?$select=Key, Title&$filter=CategoryGroupID eq 3")
  leeftijden.set_index('Key', inplace=True)
  # zet de Title om naar begin-eind paar
  leeftijden_range = leeftijden['Title'].replace(r'^(\d+) tot (\d+) jaar$', r'\1-\2', regex=True).replace(r'^(\d+) jaar of ouder$', r'\1-1000', regex=True)
  # splits die paren in van-tot
  leeftijden_range = leeftijden_range.str.split('-', expand=True).astype(int)
  # rond the "van" naar beneden op tientallen, "tot" naar boven op tientallen, en knip af naar "90+" om de ranges uit de covid tabel te matchen
  leeftijden_range[0] = leeftijden_range[0].apply(lambda x: rounddown(x)).apply(lambda x: str(min(x, 90)))
  leeftijden_range[1] = (leeftijden_range[1].apply(lambda x: roundup(x)) - 1).apply(lambda x: f'-{x}' if x < 90 else '+')
  # en plak ze aan elkaar
  leeftijden['Range'] = leeftijden_range[0] + leeftijden_range[1]
  del leeftijden['Title']

  def query(f):
    if f == 'Leeftijd':
      # alle leeftijds categerien zoals hierboven opgehaald
      return '(' + ' or '.join([f"{f} eq '{k}'" for k in leeftijden.index.values]) + ')'
    if f in ['Geslacht', 'Migratieachtergrond', 'Generatie']:
      # pak hier de key die overeenkomt met "totaal"
      ids = get_odata(cbs + '/' + f)
      return f + " eq '" + ids[ids['Title'].str.contains('totaal', na=False, case=False)]['Key'].values[0] + "'"
    if f == 'Perioden':
      # voor perioden pak de laatste
      periode = get_odata(cbs + '/Perioden').iloc[[-1]]['Key'].values[0]
      return f"{f} eq '{periode}'"
    raise ValueError(f)
  # haal alle properties op waar op kan worden gefiltered en stel de query samen. Als we niet alle termen expliciet benoemen is
  # de default namelijk "alles"; dus als we "Geslacht" niet benoemen krijgen we de data voor *alle categorien* binnen geslacht.
  filter = get_odata(cbs + '/DataProperties')
  filter = ' and '.join([query(f) for f in filter[filter.Type != 'Topic']['Key'].values])

  bevolking = get_odata(cbs + f"/TypedDataSet?$filter={filter}&$select=Leeftijd, BevolkingOpDeEersteVanDeMaand_1")
  # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
  bevolking.rename(columns = {'BevolkingOpDeEersteVanDeMaand_1': 'BevolkingOpDeEersteVanDeMaand'}, inplace = True)
  # merge de categoriecodes met de van-tot waarden
  bevolking = bevolking.merge(leeftijden, left_on = 'Leeftijd', right_index = True)
  # optellen om de leeftijds categorien bij elkaar te vegen zodat we de "agegroups" uit "covid" kunnen matchen
  bevolking = bevolking.groupby('Range')['BevolkingOpDeEersteVanDeMaand'].sum().to_frame()
  # deze factor hebben we vaker nodig
  bevolking['per 100k'] = 100000 / bevolking['BevolkingOpDeEersteVanDeMaand']

  return bevolking
