#!/usr/bin/env python3

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
import re
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
from jsonpath import JSONPath

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

class CLN:
  @classmethod
  def register(cls, timestamp):
    if not hasattr(cls, 'active'):
      cls.active = 'default'
    if not hasattr(cls, 'updated'):
      cls.updated = {}
    if not hasattr(cls, 'ignore'):
      cls.ignore = False
    if not cls.ignore and cls.active not in cls.updated or timestamp > cls.updated[cls.active]:
      cls.updated[cls.active] = timestamp

  @classmethod
  def run(cls, *args, timestamp=None):
    cls.ignore = (timestamp == False)
    if timestamp == True:
      timestamp = None
    if not cls.ignore and timestamp is not None:
      cls.active = timestamp
    if not hasattr(cls, 'updated'):
      cls.updated = {}

    if len(args) == 1 and callable(args[0]):
      return args[0]()
    else:
      print(*args)
      return lambda func: func()

# https://www.cbs.nl/nl-nl/onze-diensten/open-data/open-data-v4/snelstartgids-odata-v4
# which is a bit of a lie, since their odata implementation is broken in very imaginitive ways
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
  def bevolking(cls, leeftijdsgroepen=False):
    def roundup(x):
      return int(math.ceil(x / 10.0)) * 10
    def rounddown(x):
      return int(math.floor(x / 10.0)) * 10

    cbs = 'https://opendata.cbs.nl/ODataApi/OData/83482NED'

    if leeftijdsgroepen:
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
      if f == 'Leeftijd' and leeftijdsgroepen:
        # alle leeftijds categerien zoals hierboven opgehaald
        return '(' + ' or '.join([f"{f} eq '{k}'" for k in leeftijden.index.values]) + ')'
      if f in ['Geslacht', 'Migratieachtergrond', 'Generatie', 'Leeftijd']:
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
    flt = cls.odata(cbs + '/DataProperties')
    flt = ' and '.join([query(f) for f in flt[flt.Type != 'Topic']['Key'].values])

    query = cbs + f"/TypedDataSet?$filter={flt}&$select=Leeftijd, BevolkingOpDeEersteVanDeMaand_1"
    bevolking = cls.odata(query)
    # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
    bevolking.rename(columns = {'BevolkingOpDeEersteVanDeMaand_1': 'BevolkingOpDeEersteVanDeMaand'}, inplace = True)

    if leeftijdsgroepen:
      # merge de categoriecodes met de van-tot waarden
      bevolking = bevolking.merge(leeftijden, left_on = 'Leeftijd', right_index = True)
      # optellen om de leeftijds categorien bij elkaar te vegen zodat we de "agegroups" uit "covid" kunnen matchen
      bevolking = bevolking.groupby('Range')['BevolkingOpDeEersteVanDeMaand'].sum().to_frame()

    # deze factor hebben we vaker nodig
    bevolking['per 100k'] = 100000 / bevolking['BevolkingOpDeEersteVanDeMaand']

    return bevolking

def download_and_cache(url, n=0, headers={}, keep=None, provider=None, name=None):
  if provider is None:
    domain = urlparse(url).netloc
    provider = domain.split('.')[-2]
  if name is None:
    name = os.path.basename(url)
  name, ext = os.path.splitext(name)

  datafiles = os.path.join(provider, f'{name}*{ext}*')

  os.makedirs(provider, exist_ok = True)
  # without the user agent, LCPS won't answer HEAD requests
  headers['User-Agent'] = 'curl/7.64.1'
  resource = requests.head(url, allow_redirects=True, headers=headers)
  if 'last-modified' in resource.headers:
    lastmodified = parsedate(resource.headers['last-modified'])
  else:
    # without last-modified, only update once an hour
    lastmodified = datetime.datetime.utcnow().replace(minute=0)
  latest = os.path.join(provider, lastmodified.strftime(f'{name}-%Y-%m-%d@%H-%M{ext}'))
  CLN.register((lastmodified + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H-%M'))

  if not os.path.exists(latest) and not os.path.exists(latest + '.gz'):
    print('downloading', latest)
    with requests.get(url, headers=headers, stream=True) as r, open(latest, 'wb') as f:
      r.raise_for_status()
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
  elif n == 0:
    print(latest, 'exists')

  if 'CI' in os.environ:
    for f in glob.glob(os.path.join(provider, f'{name}*{ext}')):
      with open(f, 'rb') as f_in, gzip.open(f + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        os.remove(f)

  # delete local duplicates
  for f in glob.glob(datafiles):
    if os.path.exists(f + '.gz'):
      os.remove(f)
  dated = {}
  # RIVM publiceert soms 2 keer per dag blijkbaar
  for f in sorted(glob.glob(datafiles), reverse=True):
    ts = re.search(r'-([0-9]{4}-[0-9]{2}-[0-9]{2})@[0-9]{2}-[0-9]{2}\.', f).group(1)
    if ts not in dated:
      dated[ts] = f
    else:
      os.remove(f)

  if keep is not None:
    for f in sorted(glob.glob(datafiles), reverse=True)[:-keep]:
      os.remove(f)
  history = sorted(glob.glob(datafiles), reverse=True)
  return history[n]

class RIVM:
  @classmethod
  def csv(cls, naam, n=0):
    data = download_and_cache(f'https://data.rivm.nl/covid-19/{naam}.csv', n)
    print('loading', data)
    return pd.read_csv(data, sep=';', header=0)
  @classmethod
  def json(cls, naam, n=0):
    data = download_and_cache(f'https://data.rivm.nl/covid-19/{naam}.json', n)
    print('loading', data)
    return pd.read_json(data)

class LCPS:
  @classmethod
  def csv(cls, naam, n=0):
    data = download_and_cache(f'https://lcps.nu/wp-content/uploads/{naam}.csv', n)
    print('loading', data)
    return pd.read_csv(data, header=0)

class GitHub:
  @classmethod
  def csv(cls, path):
    headers = {
      'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
      'Accept': 'application/vnd.github.v3.raw',
    }
    url = 'https://api.github.com/repos'
    if path[0] != '/':
      url += '/'
    url += path
    print(url)
    return pd.read_csv(download_and_cache(url, keep=1, headers=headers))

class NICE:
  @classmethod
  def json(cls, name, jsonpath=None):
    cached = download_and_cache(f'https://www.stichting-nice.nl/covid-19/public/{name}/', keep=1, provider='nice', name=f'{name.replace("/", "-")}.json')
    if jsonpath is None:
      print('loading', cached)
      return pd.read_json(cached)
    else:
      print('loading', jsonpath, 'from', cached)
      with open(cached) as f:
        data = JSONPath(jsonpath).parse(json.load(f))
        assert type(data) == list, type(data)
        assert all([type(row) == dict for row in data]), [type(row) for row in data if type(row) != dict]
        return pd.DataFrame(data)

class ArcGIS:
  @classmethod
  def csv(cls, naam, n=0):
    data = download_and_cache(f'https://opendata.arcgis.com/datasets/{naam}_0.csv', n, keep=1)
    print('loading', data)
    return pd.read_csv(data, header=0)

# if __name__ == "__main__": does not work, notebooks run in main
from IPython import get_ipython
if get_ipython() is None:
  # execute only if run as a script
  # just grab latest -- run this as a separate job that's unlikely to fail so that we know for sure we grab the history we need.
  RIVM.csv('COVID-19_aantallen_gemeente_per_dag')
  RIVM.csv('COVID-19_casus_landelijk')
  RIVM.json('COVID-19_prevalentie')
  RIVM.json('COVID-19_reproductiegetal')
  RIVM.csv('COVID-19_uitgevoerde_testen')
  RIVM.csv('COVID-19_ziekenhuisopnames')
  LCPS.csv('covid-19')
  ArcGIS.csv('f27f743476a142538e8054f7a7ce12e1')
