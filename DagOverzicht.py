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
  df = (df
        .groupby(['Date_of_publication'])['Total_reported']
        .sum()
        .reset_index()
  )
  df['cumsum'] = df['Total_reported'].cumsum()
  display(df.head())

# %%
