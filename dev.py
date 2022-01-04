import pandas as pd
import json
import requests
import math

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

    # query = cbs + f"/TypedDataSet?$filter={flt}&$select=Leeftijd, BevolkingOpDeEersteVanDeMaand_1"
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

bevolking = CBS.bevolking(leeftijdsgroepen=True)
print(bevolking)
print(type(bevolking))
bevolking.to_csv('cbs_leeftijden.csv')
print("bevolking verwerkt")

bevoltest = pd.read_csv('cbs_leeftijden.csv')
bevoltest.set_index('Range', inplace=True)
print(bevoltest)
print(type(bevoltest))
print("bevoltest verwerkt")
