# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import itertools

# %%
@run('set up base frame + overleden + positief getest')
def cell():
  #df = RIVM.json('COVID-19_prevalentie')
  #df = RIVM.json('COVID-19_reproductiegetal')
  #df = CBS.leeftijdsgroepen_bevolking()
  #df = LCPS.csv('covid-19')
  #df = RIVM.csv('COVID-19_casus_landelijk')

  ag = RIVM.csv('COVID-19_aantallen_gemeente_per_dag')
  ag = ag.rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
    'Date_of_report': 'Today',
  })
  ag['Datum'] = pd.to_datetime(ag.Datum.str.replace(' .*', '', regex=True))
  ag['Today'] = pd.to_datetime(ag.Today.str.replace(' .*', '', regex=True))

  global dagoverzicht
  dagoverzicht = pd.DataFrame(index=pd.date_range(start='2020-02-27', end=ag.Today.max()))
  dagoverzicht.index.name='Key'
  dagoverzicht['Datum'] = dagoverzicht.index.strftime('%Y-%m-%d')
  dagoverzicht['LandCode'] = 'NL'
  #display(dagoverzicht)

  ag = (ag
        .groupby(['Datum'])
        .agg({'Positief getest': 'sum', 'Overleden': 'sum'})
  )
  ag['Positief getest'] = ag['Positief getest'].cumsum()
  ag['Overleden'] = ag['Overleden'].cumsum()
  dagoverzicht = dagoverzicht.merge(ag, how='left', left_index=True, right_index=True)
  display(dagoverzicht)

# %%
@run('publish')
def cell():
  global dagoverzicht

  os.makedirs('artifacts', exist_ok = True)
  dagoverzicht.to_csv('artifacts/DagOverzicht.csv', index=True)
# %%
