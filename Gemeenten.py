# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from IPython import get_ipython
get_ipython().run_line_magic('run', 'setup')
import gspread
import df2gspread as d2
import itertools

if not 'CI' in os.environ:
  gc = gspread.service_account()
  sh = gc.open_by_key('1OOXPXubXqnsOdNckUvVqt5M6b6q9oMwJMKhSbK4BQyk')
  ws = sh.get_worksheet(0)

def publish(df):
  if 'CI' in os.environ:
    os.makedirs('artifacts', exist_ok = True)
    df.to_csv('artifacts/gemeenten.csv', index=True)
  else:
    df = df.reset_index(level=0).rename(columns={'index': 'Code'})
    sh.values_clear("'Regios'!A1:ZZ10000")
    sh.values_clear("'Regios'!A1:ZZ10000")
    ws.update([df.columns.values.tolist()] + df.values.tolist())


# %%
@run('gemeenten: download gemeenten en hun codes')
def cell():
  global gemeenten
  global bevolking
  gemeenten = pd.read_csv('gemeenten.csv')
  regiocodes = 

  base = 'https://opendata.cbs.nl/ODataApi/OData/37230ned'

  # voor perioden pak de laatste
  periode = get_odata(base + '/Perioden').iloc[[-1]]['Key'].values[0]

  # startsWith would have been better to do in the filter but then the CBS "odata-ish" server responds with
  # "Object reference not set to an instance of an object."
  bevolking = get_odata(base + f"/TypedDataSet?$filter=(Perioden eq '{periode}')&$select=RegioS, BevolkingAanHetBeginVanDePeriode_1")
  # want de CBS odata API snap startsWith niet...
  bevolking = bevolking[bevolking.RegioS.str.startswith('GM')]
  # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
  bevolking.rename(columns={'RegioS': 'Code', 'BevolkingAanHetBeginVanDePeriode_1': 'BevolkingAanHetBeginVanDePeriode'}, inplace=True)
  bevolking.set_index('Code', inplace=True)

  gemeenten = gemeenten.merge(bevolking, how='left', left_on='Code', right_index=True)
  gemeenten.loc[gemeenten.Personen == 0, 'Personen'] = gemeenten.BevolkingAanHetBeginVanDePeriode
  gemeenten.set_index('Code', inplace=True)

  gemeenten = gemeenten[['Type', 'Landcode', 'GGD regio', 'Veiligheidsregio', 'Veiligheidsregio Code', 'Provincie', 'Landsdeel', 'Schoolregio', 'Personen', 'Opp land km2', 'Naam']]
  datasets = [
    ('aantallen_gemeenten', 'COVID-19_aantallen_gemeente_per_dag', 0),
    ('ziekenhuisopnames', 'COVID-19_ziekenhuisopnames', 0),
    ('ziekenhuisopnames_gisteren', 'COVID-19_ziekenhuisopnames', 1),
  ]
  for df, dataset, day in datasets:
    globals()[df] = rivm_cijfers(dataset, day)
    # vervang lege gemeentecodes door de fallback 'GM0000'
    globals()[df]['Municipality_code'] = globals()[df]['Municipality_code'].fillna('GM0000')
    # knip de tijd van de datum af
    globals()[df].Date_of_report = globals()[df].Date_of_report.str.replace(' .*', '', regex=True)
    globals()[df]['Date_of_report_date'] = pd.to_datetime(globals()[df].Date_of_report.str.replace(' .*', '', regex=True))

    globals()[df]['Date_of_report_date'] = pd.to_datetime(globals()[df]['Date_of_report_date'])
    for when in ['Date_of_statistics', 'Date_of_publication']:
      if when in globals()[df]:
        globals()[df][f'{when}_date'] = pd.to_datetime(globals()[df][when])

# %%
@run('gemeenten: absolute aantallen per gemeente')
def cell():
  def groepeer_op_gemeente(ag, columns):
    # simpele sum over groepering op gemeentecode, met rename
    df = ag.groupby(['Municipality_code'])[list(columns.keys())].sum()
    df.rename(columns=columns, inplace=True)
    return df

  positief_overleden = groepeer_op_gemeente(aantallen_gemeenten, {'Total_reported':'Positief getest', 'Deceased':'Overleden'})
  # beperk tot records op de datum van publicatie
  positief_overleden_toename = groepeer_op_gemeente(
    aantallen_gemeenten[aantallen_gemeenten.Date_of_report == aantallen_gemeenten.Date_of_publication],
    {'Total_reported':'Positief getest (toename)', 'Deceased':'Overleden (toename)'}
  )
  #print(positief_overleden.head())
  #print(positief_overleden_toename.head())

  admissions = groepeer_op_gemeente(ziekenhuisopnames, {'Hospital_admission':'Ziekenhuisopname'})
  admissions_gisteren = groepeer_op_gemeente(ziekenhuisopnames_gisteren, {'Hospital_admission':'Ziekenhuisopname_gisteren'})
  admissions_toename = admissions.merge(admissions_gisteren, how='left', on='Municipality_code')

  admissions_toename['Ziekenhuisopname (toename)'] = admissions_toename.Ziekenhuisopname - admissions_toename.Ziekenhuisopname_gisteren
  del admissions_toename['Ziekenhuisopname']
  del admissions_toename['Ziekenhuisopname_gisteren']

  # en plak het zwik aan elkaar
  global gemeenten
  gemeenten = (gemeenten
    .merge(positief_overleden, how='left', left_index=True, right_index=True)
    .merge(admissions, how='left', left_index=True, right_index=True)
    .merge(positief_overleden_toename, how='left', left_index=True, right_index=True)
    .merge(admissions_toename, how='left', left_index=True, right_index=True)
    .fillna(0)
  )

  # per 100k voor de absolute kolommen
  for df in [positief_overleden, admissions]:
    for col in df.columns:
      gemeenten[col + ' per 100.000'] = (gemeenten[col] * (100000 / gemeenten.Personen)).replace(np.inf, 0)

  gemeenten['Positief getest 1d/100k'] = gemeenten['Positief getest (toename)'] / gemeenten['Personen']

  gemeenten['Positief getest percentage'] = (gemeenten['Positief getest'] / gemeenten['Personen']).replace(np.inf, 0)
  gemeenten['Positief getest per km2'] = (gemeenten['Positief getest'] / gemeenten['Opp land km2']).replace(np.inf, 0)

# %%
@run('gemeenten: historie')
def cell():
  def add_history(df, column, colors, label):
    weeks = 26
    if 'scale' in df:
      df = df.assign(count=df[column] * df.scale)
    else:
      df = df.assign(count=df[column])

    if 'Date_of_statistics' in df:
      datefield = 'Date_of_statistics_date'
    else:
      datefield = 'Date_of_publication_date'

    historie = df[['Municipality_code', 'count' ]].assign(wekenterug=np.floor(
        (
          df.Date_of_report_date
          -
          df[datefield]
        )
        /
        np.timedelta64(7, 'D')
      ).astype(np.int)
    )
    historie = historie[historie.wekenterug < weeks]
  
    # voeg regels met 0 voor elke gemeente/week terug zodat we zeker weten dat elke week bestaat
    fill = pd.DataFrame(
      index=pd.MultiIndex.from_product(
        [ df.Municipality_code.unique(), np.arange(weeks) ],
        names = ['Municipality_code', 'wekenterug']
      )
    ).reset_index()
    fill['count'] = 0
    historie = pd.concat([historie, fill[historie.columns]], axis=0)
    # en dan kantelen en optellen
    historie = (historie
      .groupby(['Municipality_code', 'wekenterug'])['count']
      .sum()
      .unstack(fill_value=np.nan)
      .rename_axis(None, axis=1)
    )
    # must be done *before* the rename
    positief_hoogste_week = historie.idxmax(axis=1).to_frame().rename(columns={0: f'{label} hoogste week' })
    historie.rename(columns={ n: f'{label} w{-n}' for n in range(weeks) }, inplace=True)
  
    historie_kleuren = (historie.divide(historie.max(axis=1), axis=0) * 1000).rename(columns={col:col.replace('w', 'cw') for col in historie})
  
    global gemeenten
    gemeenten = (gemeenten.merge(historie, left_index=True, right_index=True))
    if colors:
      gemeenten = gemeenten.merge(historie_kleuren, left_index=True, right_index=True)
    gemeenten = gemeenten.merge(positief_hoogste_week, left_index=True, right_index=True)
    # bij ontbreken van w-1 vaste waarde 9.99
    gemeenten[f'{label} t.o.v. vorige week'] = (gemeenten[f'{label} w0'] / gemeenten[f'{label} w-1']).replace(np.inf, 9.99)

  add_history(
    aantallen_gemeenten,
    colors=True,
    label='Positief getest',
    column='Total_reported'
  )
  add_history(
    aantallen_gemeenten.merge(
      gemeenten.assign(scale=100000 / gemeenten.Personen)[['scale']], left_on='Municipality_code', right_index=True
    ),
    colors=False,
    label='Positief getest per 100.000',
    column='Total_reported'
  )
  add_history(
    ziekenhuisopnames,
    colors=True,
    label='Ziekenhuisopname',
    column='Hospital_admission'
  )
  add_history(
    aantallen_gemeenten,
    colors=True,
    label='Overleden',
    column='Deceased'
  )
# %%
display(gemeenten.head())
publish(gemeenten.fillna(0).replace(np.inf, 0))

# %%
regiocodes = pd.read_csv('regiocodes.csv')
regios = regiocodes[regiocodes.Type == 'GGD']
df = aantallen_gemeenten.merge(regiocodes, how='left', left_on='Municipal_health_service', right_on='GGD Regio')
df[pd.isnull(df['GGD Regio'])]['Municipal_health_service'].unique()
gemeenten['GGD regio'].unique()
# %%
