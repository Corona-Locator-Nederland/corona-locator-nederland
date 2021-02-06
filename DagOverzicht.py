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

  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for stat in df.columns:
    dagoverzicht[f'{stat} (niew)'] = dagoverzicht[stat].fillna(0).astype(int)
    dagoverzicht[stat] = dagoverzicht[f'{stat} (niew)'].cumsum()
    dagoverzicht[f'{stat} per 100.000'] = dagoverzicht[stat] * bevolking['per 100k']
    dagoverzicht[f'{stat} 7d'] = (dagoverzicht[stat] - dagoverzicht[stat].shift(7)).fillna(0).astype(int)
    dagoverzicht[f'{stat} 7d per 100.000'] = dagoverzicht[f'{stat} 7d'] * bevolking['per 100k']

# %%
@run('set up base frame + overleden + positief getest')
def cell():
  df = RIVM.csv('COVID-19_aantallen_gemeente_per_dag').rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
    'Date_of_report': 'Today',
  })
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  df['Today'] = pd.to_datetime(df.Today.str.replace(' .*', '', regex=True))

  global dagoverzicht
  dagoverzicht = pd.DataFrame(index=pd.date_range(start='2020-02-27', end=df.Today.max()))
  dagoverzicht.index.name='Key'
  dagoverzicht['Datum'] = dagoverzicht.index.strftime('%Y-%m-%d')
  dagoverzicht['LandCode'] = 'NL'

  addstats(df.groupby(['Datum']).agg({'Positief getest': 'sum', 'Overleden': 'sum'}))
  display(dagoverzicht.head(10))

# %%
@run('ziekenhuisopnames')
def cell():
  df = RIVM.csv('COVID-19_ziekenhuisopnames').rename(columns={
    'Hospital_admission': 'Ziekenhuisopnames',
    'Date_of_statistics': 'Datum',
  })
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
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
    df = RIVM.json(dataset).rename(columns={source: target})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[[target]]
    dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
    dagoverzicht[target] = dagoverzicht[target].fillna(0)
  dagoverzicht['Besmettelijk per 100.000'] = (dagoverzicht['Besmettelijk']  * bevolking['per 100k']).round(0)
  display(dagoverzicht)

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
# %%
