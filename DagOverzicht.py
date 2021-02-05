# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import itertools

# %%
@run('regio: load regios en hun basisgegevens')
def cell():
  df = RIVM.json('COVID-19_prevalentie')
  df = RIVM.json('COVID-19_reproductiegetal')
  #df = CBS.leeftijdsgroepen_bevolking()
  df = LCPS.csv('covid-19')
  df = RIVM.csv('COVID-19_casus_landelijk')

  df = RIVM.csv('COVID-19_aantallen_gemeente_per_dag')
  df = df.rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
  })

  df = (df
        .groupby(['Datum'])
        .agg({'Positief getest': 'sum', 'Overleden': 'sum'})
        .reset_index()
  )
  df['Key'] = df['Datum']
  df['LandCode'] = 'NL'
  df['cumsum'] = df['Positief getest'].cumsum()
  display(df.head())

# %%
