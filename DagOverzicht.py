# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import itertools

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

  df = (df
        .groupby(['Datum'])
        .agg({'Positief getest': 'sum', 'Overleden': 'sum'})
  )
  df['Positief getest'] = df['Positief getest'].cumsum()
  df['Positief getest (toename)'] = (df['Positief getest'] - df['Positief getest'].shift(1)).fillna(0).astype(int)
  df['Overleden'] = df['Overleden'].cumsum()
  df['Overleden (toename)'] = (df['Overleden'] - df['Overleden'].shift(1)).fillna(0).astype(int)
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  display(dagoverzicht)

# %%
@run('ziekenhuisopnames')
def cell():
  df = RIVM.csv('COVID-19_ziekenhuisopnames').rename(columns={
    'Hospital_admission': 'Ziekenhuisopnames',
    'Date_of_statistics': 'Datum',
  })
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  df = (df
        .groupby(['Datum'])
        .agg({'Ziekenhuisopnames': 'sum'})
  )
  display(df)
  global dagoverzicht
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  dagoverzicht['Ziekenhuisopnames'] = dagoverzicht['Ziekenhuisopnames'].fillna(0).astype(int).cumsum()
  dagoverzicht['Ziekenhuisopnames (toename)'] = (dagoverzicht['Ziekenhuisopnames'] - dagoverzicht['Ziekenhuisopnames'].shift(1)).fillna(0).astype(int)

  display(dagoverzicht)
# %%
@run('publish')
def cell():
  global dagoverzicht

  os.makedirs('artifacts', exist_ok = True)
  dagoverzicht.to_csv('artifacts/DagOverzicht.csv', index=True)
# %%
