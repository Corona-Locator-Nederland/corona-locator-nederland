from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from dotenv import load_dotenv, find_dotenv
dot_env = find_dotenv()
if dot_env == '': dot_env = find_dotenv(filename='dot.env')
load_dotenv(dot_env, override=True)

from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import gspread
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.parse import urlparse
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

if 'GSHEET' in os.environ:
  def gsheet(df):
    print('updating GSheet')
    gc = gspread.service_account()
    sh = gc.open_by_key(os.environ['GSHEET'])
    ws = sh.get_worksheet(0)

    sh.values_clear("'Regios'!A1:ZZ10000")
    ws.update('A1', [df.columns.values.tolist()] + df.values.tolist())

def run(*args):
  if len(args) == 1 and callable(args[0]):
    return args[0]()
  else:
    print(*args)
    return lambda func: func()

# https://www.cbs.nl/nl-nl/onze-diensten/open-data/open-data-v4/snelstartgids-odata-v4
# which os a bit of a lie, since their odata implementation is broken in very imaginitive ways
class CBS:
  @classmethod
  def odata(cls, url):
    data = pd.DataFrame()
    while url:
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

  @classmethod
  def leeftijdsgroepen_bevolking(cls):
    def roundup(x):
      return int(math.ceil(x / 10.0)) * 10
    def rounddown(x):
      return int(math.floor(x / 10.0)) * 10

    cbs = 'https://opendata.cbs.nl/ODataApi/OData/83482NED'

    leeftijden = cls.odata(cbs + "/Leeftijd?$select=Key, Title&$filter=CategoryGroupID eq 3")
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
        ids = cls.odata(cbs + '/' + f)
        return f + " eq '" + ids[ids['Title'].str.contains('totaal', na=False, case=False)]['Key'].values[0] + "'"
      if f == 'Perioden':
        # voor perioden pak de laatste
        periode = cls.odata(cbs + '/Perioden').iloc[[-1]]['Key'].values[0]
        return f"{f} eq '{periode}'"
      raise ValueError(f)
    # haal alle properties op waar op kan worden gefiltered en stel de query samen. Als we niet alle termen expliciet benoemen is
    # de default namelijk "alles"; dus als we "Geslacht" niet benoemen krijgen we de data voor *alle categorien* binnen geslacht.
    filter = cls.odata(cbs + '/DataProperties')
    filter = ' and '.join([query(f) for f in filter[filter.Type != 'Topic']['Key'].values])

    bevolking = cls.odata(cbs + f"/TypedDataSet?$filter={filter}&$select=Leeftijd, BevolkingOpDeEersteVanDeMaand_1")
    # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
    bevolking.rename(columns = {'BevolkingOpDeEersteVanDeMaand_1': 'BevolkingOpDeEersteVanDeMaand'}, inplace = True)
    # merge de categoriecodes met de van-tot waarden
    bevolking = bevolking.merge(leeftijden, left_on = 'Leeftijd', right_index = True)
    # optellen om de leeftijds categorien bij elkaar te vegen zodat we de "agegroups" uit "covid" kunnen matchen
    bevolking = bevolking.groupby('Range')['BevolkingOpDeEersteVanDeMaand'].sum().to_frame()
    # deze factor hebben we vaker nodig
    bevolking['per 100k'] = 100000 / bevolking['BevolkingOpDeEersteVanDeMaand']

    return bevolking

def load_and_cache(url, n=0):
  domain = urlparse(url).netloc
  provider = domain.split('.')[-2]
  name, ext = os.path.splitext(os.path.basename(url))

  os.makedirs(provider, exist_ok = True)
  # without the user agent, LCPS won't answer HEAD requests
  resource = requests.head(url, allow_redirects=True, headers={ 'User-Agent': 'curl/7.64.1'})
  latest = os.path.join(provider, parsedate(resource.headers['last-modified']).strftime(f'{name}-%Y-%m-%d@%H-%M{ext}'))
  if not os.path.exists(latest) and not os.path.exists(latest + '.gz'):
    print('downloading', latest)
    urlretrieve(url, latest)
  elif n == 0:
    print(latest, 'exists')

  if 'CI' in os.environ:
    for f in glob.glob(os.path.join(provider, f'{name}*{ext}')):
      with open(f, 'rb') as f_in, gzip.open(f + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        os.remove(f)

  datafiles = os.path.join(provider, f'{name}*{ext}*')
  # delete local duplicates
  for f in glob.glob(datafiles):
    if os.path.exists(f + '.gz'):
      os.remove(f)
  history = sorted(glob.glob(datafiles), reverse=True)
  return history[n]

class RIVM:
  @classmethod
  def csv(cls, naam, n=0):
    data = load_and_cache(f'https://data.rivm.nl/covid-19/{naam}.csv', n)
    print('loading', data)
    return pd.read_csv(data, sep=';', header=0)
  @classmethod
  def json(cls, naam, n=0):
    data = load_and_cache(f'https://data.rivm.nl/covid-19/{naam}.json', n)
    print('loading', data)
    return pd.read_json(data)

class LCPS:
  @classmethod
  def csv(cls, naam, n=0):
    data = load_and_cache(f'https://lcps.nu/wp-content/uploads/{naam}.csv', n)
    print('loading', data)
    return pd.read_csv(data, header=0)
