# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
get_ipython().run_line_magic('run', 'setup')
import gspread
import df2gspread as d2g

gc = gspread.service_account()
sh = gc.open_by_key('1OOXPXubXqnsOdNckUvVqt5M6b6q9oMwJMKhSbK4BQyk')
ws = sh.get_worksheet(0)

def publish(df):
  sh.values_clear("'Regios'!A1:ZZ10000")
  sh.values_clear("'Regios'!A1:ZZ10000")
  ws.update([df.columns.values.tolist()] + df.values.tolist())

# %%
@run('regios: download gemeenten en hun codes')
def cell():
  global gemeenten
  gemeenten = pd.read_csv('gemeenten.csv')

  base = 'https://opendata.cbs.nl/ODataApi/OData/37230ned'

  # voor perioden pak de laatste
  periode = get_odata(base + '/Perioden').iloc[[-1]]['Key'].values[0]

  # startsWith would have been better to do in the filter but then the CBS "odata-ish" server responds with
  # "Object reference not set to an instance of an object."
  bevolking = get_odata(base + f"/TypedDataSet?$filter=(Perioden eq '{periode}')&$select=RegioS, BevolkingAanHetBeginVanDePeriode_1")
  bevolking = bevolking[bevolking.RegioS.str.startswith('GM')]
  # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
  bevolking.rename(columns = {'BevolkingAanHetBeginVanDePeriode_1': 'BevolkingAanHetBeginVanDePeriode'}, inplace = True)
  gemeenten = gemeenten.merge(bevolking, how='left', left_on='Code', right_on='RegioS')
  gemeenten = gemeenten[~np.isnan(gemeenten.BevolkingAanHetBeginVanDePeriode)]
  publish(gemeenten)
# %%
# Download de bevolkings cijfers van CBS, uitgesplitst op gemeenten
@run('regios: download demografische data van CBS')
def cell():
  

# %%
gc = gspread.service_account()
sh = gc.open_by_key('1OOXPXubXqnsOdNckUvVqt5M6b6q9oMwJMKhSbK4BQyk')
ws = sh.get_worksheet(0)
sh.values_clear("'Regios'!A1:ZZ10000")
ws.update([gemeenten.columns.values.tolist()] + gemeenten.values.tolist())

