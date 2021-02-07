# %%
from IPython import get_ipython
from IPython.core.display import display
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')

bevolking = CBS.bevolking().iloc[0]
#display(bevolking.BevolkingOpDeEersteVanDeMaand)
#display(bevolking['per 100k'])

def addstats(df):
  global dagoverzicht

  # de aanname is dat df al gegroepeerd is op datum, en dat de kolommen de dagtotalen zijn
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for stat in df.columns:
    # nieuw = dagtotaal, zet missende dagtotalen op 0
    dagoverzicht[f'{stat} (nieuw)'] = dagoverzicht[stat].fillna(0).astype(int)
    # kolom met eerder de dagtotalen = cumulatieve som over dagtotalen
    dagoverzicht[stat] = dagoverzicht[f'{stat} (nieuw)'].cumsum()
    # factor voor 100k
    dagoverzicht[f'{stat} per 100.000'] = dagoverzicht[stat] * bevolking['per 100k']
    # verschil met 7 dagen terug
    dagoverzicht[f'{stat} 7d'] = (dagoverzicht[stat] - dagoverzicht[stat].shift(7)).fillna(0).astype(int)
    # en weer factor 100 k
    dagoverzicht[f'{stat} 7d per 100.000'] = dagoverzicht[f'{stat} 7d'] * bevolking['per 100k']

# %%
@run('set up base frame + overleden + positief getest')
def cell():
  # 2 vliegen in 1 klap -- aantallen pos + overleden, en laatste datum met data voor de datum range
  df = RIVM.csv('COVID-19_aantallen_gemeente_per_dag').rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
    'Date_of_report': 'Today',
  })
  # sloop tijd van de datum en zet om in datetime object
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  df['Today'] = pd.to_datetime(df.Today.str.replace(' .*', '', regex=True))

  global dagoverzicht
  # maak leeg dataframe met een rij voor elke dag van 2020-02-27 tm Date_of_report
  dagoverzicht = pd.DataFrame(index=pd.date_range(start='2020-02-27', end=df.Today.max()))
  # noem de index Key
  dagoverzicht.index.name='Key'
  # vul de datum kolom
  dagoverzicht['Datum'] = dagoverzicht.index.strftime('%Y-%m-%d')
  # vaste waarde voor LandCode
  dagoverzicht['LandCode'] = 'NL'

  # sommeer pos en overl op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Positief getest': 'sum', 'Overleden': 'sum'}))
  display(dagoverzicht.head(10))

# %%
@run('ziekenhuisopnames')
def cell():
  df = RIVM.csv('COVID-19_ziekenhuisopnames').rename(columns={
    'Hospital_admission': 'Ziekenhuisopnames',
    'Date_of_statistics': 'Datum',
  })
  # datum naar datetime
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  # sommeer op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Ziekenhuisopnames': 'sum'}))
  display(dagoverzicht.head())

# %%
@run('reproductiegetal en besmettelijkheid')
def cell():
  global dagoverzicht

  datasets = [
    ('COVID-19_reproductiegetal', 'Rt_avg', 'Reproductiegetal'),
    ('COVID-19_prevalentie', 'prev_avg', 'Besmettelijk'),
  ]
  for dataset, source, target in datasets:
    # laad de dataset
    df = RIVM.json(dataset).rename(columns={source: target})
    # Date naar datetime index voor de merge
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # merge alleen de target kolom
    df = df[[target]]
    # voeg to aan dagoverzicht
    dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
    # vul lege waarden met 0
    dagoverzicht[target] = dagoverzicht[target].fillna(0)
  # per 100k factor
  dagoverzicht['Besmettelijk per 100.000'] = (dagoverzicht['Besmettelijk']  * bevolking['per 100k']).round(0)
  display(dagoverzicht)

@run('LCPS')
def cell():
  # laad dataset
  df = LCPS.csv('covid-19').rename(columns={
    'IC_Bedden_COVID': 'LCPS IC Bedden COVID',
    'IC_Bedden_Non_COVID': 'LCPS IC Bedden Non COVID',
    'Kliniek_Bedden': 'LCPS Kliniek Bedden COVID',
    'IC_Nieuwe_Opnames_COVID': 'LCPS IC Nieuwe Opnames COVID',
    'Kliniek_Nieuwe_Opnames_COVID': 'LCPS Kliniek Nieuwe Opnames COVID',
  })
  # datum naar datetime index voor merge
  df['Datum'] = pd.to_datetime(df['Datum'])
  df.set_index('Datum', inplace=True)

  # sommeer op datum
  df = df.groupby(['Datum']).agg({col: 'sum' for col in df.columns})

  # toename = waarde - vorige
  df['LCPS IC Bedden COVID (toename)'] = (df['LCPS IC Bedden COVID'] - df['LCPS IC Bedden COVID'].shift(1)).fillna(0)
  df['LCPS Kliniek Bedden COVID (toename)'] = (df['LCPS Kliniek Bedden COVID'] - df['LCPS Kliniek Bedden COVID'].shift(1)).fillna(0)

  global dagoverzicht
  # voeg toe aan dagoverzicht
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for col in df.columns:
    # vervang lege waarden door 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0).astype(int)

# %%
@run('corrections')
def cell():
  # laad corrections van mzelst
  df = GitHub.csv('mzelst/covid-19/contents/corrections/corrections_perday.csv')
  # date naar datetime index voor merge
  df['date'] = pd.to_datetime(df.date)
  df.set_index('date', inplace=True)
  columns =  {
    'net.infection': 'Positief getest (toename)',
    'net.hospitals': 'Ziekenhuisopnames (toename)',
    'net.deaths': 'Overleden (toename)'
  }
  # rename kolommen naar onze namen
  df = df.rename(columns=columns)[list(columns.values())]

  global dagoverzicht
  # voeg toe aan dagoverxicht
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for col in columns.values():
    # set lege waarden op 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0).astype(int)

# %%
@run('uitgevoerde testen')
def cell():
  df = RIVM.csv('COVID-19_uitgevoerde_testen').rename(columns={
    'Date_of_statistics': 'Datum',
    'Tested_with_result': 'Getest',
    'Tested_positive': 'Positief',
  })
  df['Datum'] = pd.to_datetime(df.Datum)
  df = df.groupby(['Datum']).agg({'Getest': 'sum', 'Positief': 'sum'})
  display(df)

# %%
async def publish():
  global dagoverzicht

  m = (dagoverzicht == np.inf)
  df = dagoverzicht.loc[m.any(axis=1), m.any(axis=0)]
  display(df.head())

  os.makedirs('artifacts', exist_ok = True)
  dagoverzicht.to_csv('artifacts/DagOverzicht.csv', index=True)

  if knack:
    print('updating knack')
    df = dagoverzicht.assign(Key=dagoverzicht.index.strftime('%Y-%m-%d'))
    await knack.update(objectName='Dagoverzicht', df=df)
await publish()
