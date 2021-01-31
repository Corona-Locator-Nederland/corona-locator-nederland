# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import itertools

# %%
@run('regio: load regios en hun basisgegevens')
def cell():
  df = RIVM.json('COVID-19_prevalentie')
  display(df.head())
  df = RIVM.json('COVID-19_reproductiegetal')
  display(df.head())
  df = CBS.leeftijdsgroepen_bevolking()
  display(df.head())
  df = LCPS.csv('covid-19')
  display(df.head())
# %%
