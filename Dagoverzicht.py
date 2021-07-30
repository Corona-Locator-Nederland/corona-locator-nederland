# %% setup
from IPython import get_ipython
from IPython.core.display import display
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
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
    dagoverzicht[f'{stat} 7d'] = dagoverzicht[f'{stat} (nieuw)'].rolling(7).sum().fillna(0)
    # en weer factor 100 k
    dagoverzicht[f'{stat} 7d per 100.000'] = dagoverzicht[f'{stat} 7d'] * bevolking['per 100k']

# %% set up base frame from ESRI -> NICE
@run
def cell():
  df = ArcGIS.nice('f27f743476a142538e8054f7a7ce12e1')

  df['date'] = pd.to_datetime(df.date.str.replace(' .*', '', regex=True))
  df.set_index('date', inplace=True)

  # base date range uit NICE
  global dagoverzicht
  # maak leeg dataframe met een rij voor elke dag van 2020-02-27 tm Date_of_publication
  dagoverzicht = pd.DataFrame(index=pd.date_range(start='2020-02-27', end=df.index.max()))
  # noem de index Key
  dagoverzicht.index.name='Key'
  # vul de datum kolom
  dagoverzicht['Datum'] = dagoverzicht.index.strftime('%Y-%m-%d')
  # vaste waarde voor LandCode
  dagoverzicht['LandCode'] = 'NL'

  # the NICE data as prepared by ESRI can just be used as-is, so just rename the columns
  for prefix, kind in [ ('ic', 'IC'), ('zkh', 'Ziekenhuis') ]:
    df = df.rename(columns={
      f'{prefix}NewIntake': f'NICE {kind} Bedden (intake)',
      f'{prefix}IntakeCount': f'NICE {kind} Bedden',
      f'{prefix}IntakeCumulative': f'NICE {kind} Bedden (cumulatief)',
      f'{prefix}DiedCumulative': f'NICE {kind} Overleden',
    })

  # remove the columns we don't use
  df = df[[col for col in df.columns if 'NICE' in col]]
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)

  # after the merge, dates missing in the ESRI/NICE set will be `nan`, so fill these
  for col in df.columns:
    # for cumulatief columns, fill-forward from last known value
    if 'cumulatief' in col:
      dagoverzicht[col] = dagoverzicht[col].ffill()
    # for non-cumulative and whatever remains in cumulative after fill forward (which will be leading nan's), fill with 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0)

  for kind in [ 'IC', 'Ziekenhuis' ]:
    dagoverzicht[f'NICE {kind} Bedden (toename)'] = (dagoverzicht[f'NICE {kind} Bedden'] - dagoverzicht[f'NICE {kind} Bedden'].shift(1)).fillna(0)

  for window, shift, past in [(3, 1, 'afgelopen '), (7, 0, '')]:
    dagoverzicht[f'NICE IC Bedden (intake) {past}{window}d'] = dagoverzicht['NICE IC Bedden (intake)'].shift(shift).rolling(window).sum().fillna(0)

  dagoverzicht['NICE IC Bedden (intake) week-1'] = dagoverzicht['NICE IC Bedden (intake) 7d'].shift(7).fillna(0)

# %% overleden + positief getest
@run
def cell():
  df = RIVM.csv('COVID-19_aantallen_gemeente_per_dag').rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
    'Date_of_report': 'Today',
  })
  # sloop tijd van de datum en zet om in datetime object
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))

  # sommeer pos en overl op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Positief getest': 'sum', 'Overleden': 'sum'}))
  global dagoverzicht
  for col in ['Overleden', 'Positief getest']:
    dagoverzicht[f'{col} week-1'] = dagoverzicht[f'{col} 7d'].shift(7).fillna(0)
  display(dagoverzicht.head(10))

# %% ziekenhuisopnames
@run
def cell():
  df = RIVM.csv('COVID-19_ziekenhuisopnames').rename(columns={
    'Hospital_admission': 'Ziekenhuisopnames',
    'Date_of_statistics': 'Datum',
  })
  # datum naar datetime
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  # sommeer op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Ziekenhuisopnames': 'sum'}))
  dagoverzicht['Ziekenhuisopnames week-1'] = dagoverzicht['Ziekenhuisopnames 7d'].shift(7).fillna(0)
  display(dagoverzicht.head())

# %% reproductiegetal en besmettelijkheid
@run
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

# %% uitgevoerde testen
@run
def cell():
  df = RIVM.csv('COVID-19_uitgevoerde_testen').rename(columns={
    'Date_of_statistics': 'Datum',
    'Tested_with_result': 'GGD getest',
    'Tested_positive': 'GGD getest positief',
  })
  df['Datum'] = pd.to_datetime(df.Datum)
  df = df.groupby(['Datum']).agg({'GGD getest': 'sum', 'GGD getest positief': 'sum'})
  display(df)

  global dagoverzicht
  columns = dagoverzicht.columns

  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)

  dagoverzicht['GGD percentage positief'] = (dagoverzicht['GGD getest positief'] / dagoverzicht['GGD getest']).fillna(0)

  dagoverzicht['GGD getest (7 daags)'] = dagoverzicht['GGD getest'].rolling(7).sum().fillna(0)
  dagoverzicht['GGD getest positief (7 daags)'] = dagoverzicht['GGD getest positief'].rolling(7).sum().fillna(0)

  dagoverzicht['GGD percentage positief (7 daags)'] = (dagoverzicht['GGD getest positief (7 daags)'] / dagoverzicht['GGD getest (7 daags)']).fillna(0)

  dagoverzicht['GGD getest (cumulatief)'] = dagoverzicht['GGD getest'].cumsum()
  dagoverzicht['GGD getest positief (cumulatief)'] = dagoverzicht['GGD getest positief'].cumsum()

  dagoverzicht['GGD percentage positief (cumulatief)'] = (dagoverzicht['GGD getest positief (cumulatief)'] / dagoverzicht['GGD getest (cumulatief)']).fillna(0)

  # fill 0 *after* the calculations above to prevent summing 'GGD getest (7 daags) from the starting date of the 'GGD getest' series
  for col in dagoverzicht.columns:
    # only zero out new columns
    if col in columns: continue
    dagoverzicht[col] = dagoverzicht[col].fillna(0)

  display(dagoverzicht.head(10))

# %% LCPS
@run
def cell():
  # laad dataset
  # df = LCPS.csv('covid-19').rename(columns={   vervangen ivm nieuwe filenaam er 26-7-2021
  df = LCPS.csv('covid-19-datafeed').rename(columns={  
    'IC_Bedden_COVID': 'LCPS IC Bedden COVID',
    'IC_Bedden_Non_COVID': 'LCPS IC Bedden Non COVID',
    'Kliniek_Bedden': 'LCPS Kliniek Bedden COVID',
    'IC_Nieuwe_Opnames_COVID': 'LCPS IC Nieuwe Opnames COVID',
    'Kliniek_Nieuwe_Opnames_COVID': 'LCPS Kliniek Nieuwe Opnames COVID',
  })
  # datum naar datetime index voor merge
  df['Datum'] = pd.to_datetime(df['Datum'], format='%d-%m-%Y')
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

# %% corrections
@run
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

# %% Personen en Key
@run
def cell():
  global dagoverzicht
  dagoverzicht['Personen'] = bevolking.BevolkingOpDeEersteVanDeMaand
  dagoverzicht['Key'] = dagoverzicht.index.strftime('%Y-%m-%d')
  display(dagoverzicht)

# %%
if knack:
  await knack.publish(dagoverzicht, 'Dagoverzicht', Cache)
