# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import gspread
import itertools

if 'GSHEET' in os.environ:
  gc = gspread.service_account()
  sh = gc.open_by_key(os.environ['GSHEET'])
  ws = sh.get_worksheet(0)

def sortcolumns(df):
  return df[sorted(df.columns)]

def orderedprefix(df, prefix, exclude=[]):
  return df[prefix + [col for col in df.columns if col not in prefix and col not in exclude]]

def publish(df):
  display(df.head())
  if 'GSHEET' in os.environ:
    sh.values_clear("'Regios'!A1:ZZ10000")
    ws.update('A1', [df.columns.values.tolist()] + df.values.tolist())
  else:
    os.makedirs('artifacts', exist_ok = True)
    df.to_csv('artifacts/gemeenten.csv', index=True)


# %%
@run('regio: load regios en hun basisgegevens')
def cell():
  global gemeenten
  gemeenten = pd.read_csv('gemeenten.csv').rename(columns={
    'Code': 'GemeenteCode',
    'Naam': 'Gemeente',
    'Veiligheidsregio Code': 'VeiligheidsregioCode',
    'GGD regio': 'GGDregio',
    'Landcode': 'LandCode',
  })
  del gemeenten['Type']
  
  global regiocodes
  regiocodes = pd.read_csv('regiocodes.csv')
  regiocodes = regiocodes.rename(columns={'Landcode': 'LandCode'})
  regiocodes.loc[regiocodes.Type == 'GGD', 'Type'] = 'GGDregio'
  
  for regiotype in ['GGDregio', 'Provincie', 'Landsdeel', 'Schoolregio']:
    gemeenten = gemeenten.merge(
      regiocodes[regiocodes.Type == regiotype][['LandCode', 'Regio', 'Code']].rename(columns={'Code': regiotype + 'Code', 'Regio': regiotype}),
      how='left',
      on=[regiotype, 'LandCode'],
    )
  gemeenten = gemeenten.merge(
    regiocodes[regiocodes.Type == 'Land'][['LandCode', 'Regio']].rename(columns={'Regio': 'Land'}),
    how='left',
    on='LandCode'
  )
  
  for regiotype, prefix in [('GGDregio', 'GG'), ('Veiligheidsregio', 'VR'), ('Provincie', 'PV'), ('Landsdeel', 'LD'), ('Schoolregio', 'SR')]:
    gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', regiotype] = ''
    gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', f'{regiotype}Code'] = f'{prefix}00'
  gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', 'LandCode'] = 'NL'
  
  base = 'https://opendata.cbs.nl/ODataApi/OData/37230ned'
  
  # voor perioden pak de laatste
  periode = get_odata(base + '/Perioden').iloc[[-1]]['Key'].values[0]
  
  # startsWith would have been better to do in the filter but then the CBS "odata-ish" server responds with
  # "Object reference not set to an instance of an object."
  bevolking = get_odata(base + f"/TypedDataSet?$filter=(Perioden eq '{periode}')&$select=RegioS, BevolkingAanHetBeginVanDePeriode_1")
  # want de CBS odata API snap startsWith niet...
  bevolking = bevolking[bevolking.RegioS.str.startswith('GM')]
  # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
  bevolking.rename(columns={'RegioS': 'GemeenteCode', 'BevolkingAanHetBeginVanDePeriode_1': 'BevolkingAanHetBeginVanDePeriode'}, inplace=True)
  
  gemeenten = gemeenten.merge(bevolking, how='left', on='GemeenteCode')
  # default naar gegeven waarde bij ontbrekende data
  gemeenten.loc[gemeenten.Personen == 0, 'Personen'] = gemeenten.BevolkingAanHetBeginVanDePeriode
  del gemeenten['BevolkingAanHetBeginVanDePeriode']
  gemeenten = sortcolumns(gemeenten)
  
  def prepare(dataset, day):
    df = rivm_cijfers(dataset, day)
    df[['GemeenteCode', 'VeiligheidsregioCode', 'Veiligheidsregio']] = df[['Municipality_code', 'Security_region_code', 'Security_region_name']]
    df['GemeenteCode'] = df['GemeenteCode'].fillna('GM0000')
  
    for regiotype in ['GGDregio', 'Provincie', 'Landsdeel', 'Schoolregio']:
      df = df.merge(gemeenten[['GemeenteCode', f'{regiotype}Code']].drop_duplicates(), on='GemeenteCode')
    df['LandCode'] = 'NL'
  
    # knip de tijd van de datum af
    df['Today'] = pd.to_datetime(df.Date_of_report.str.replace(' .*', '', regex=True))
  
    for when in ['Date_of_statistics', 'Date_of_publication']:
      if when in df:
        df['Date'] = pd.to_datetime(df[when])
        df['WekenTerug'] = ((df.Today - df.Date) / np.timedelta64(7, 'D')).astype(np.int)
    return sortcolumns(df)

  global aantallen, ziekenhuisopnames, ziekenhuisopnames_1
  aantallen = prepare('COVID-19_aantallen_gemeente_per_dag', 0)
  ziekenhuisopnames = prepare('COVID-19_ziekenhuisopnames', 0)
  ziekenhuisopnames_1 = prepare('COVID-19_ziekenhuisopnames', 1)

# %%
def groupregio(regiotype):
  global gemeenten

  grouping = [ f'{regiotype}Code', regiotype]
  if regiotype != 'Land':
    grouping += [ 'LandCode' ]

  columns = [
    'GGDregio',
    'Veiligheidsregio',
    'VeiligheidsregioCode',
    'Provincie',
    'Landsdeel',
    'Schoolregio',
    'Land',
  ]

  if regiotype == 'Gemeente':
    df = gemeenten[grouping + columns + ['Personen', 'Opp land km2']].rename(columns={'Gemeente': 'Naam', 'GemeenteCode': 'Code'})
  else:
    df = (gemeenten[gemeenten.GemeenteCode != 'GM0000']
      .groupby(grouping).agg({ 'Personen': 'sum', 'Opp land km2': 'sum' })
      .reset_index()
      .rename(columns={regiotype: 'Naam', f'{regiotype}Code': 'Code'})
      .assign(**{ col: '' for col in columns })
    )
    if regiotype == 'Land':
      df['Land'] = df['Naam']
      df['LandCode'] = df['Code']
  return df.assign(Type=regiotype)

def sumcols(df, regiotype, columns):
  regiotype_code = f'{regiotype}Code'
  return (df
    .groupby([regiotype_code])[list(columns.keys())]
    .sum()
    .rename(columns=columns)
  )

def histcols(df, regiotype, maxweeks, column, colors=False, highestweek=False):
  assert len(column) == 1
  label = list(column.values())[0]
  datacolumn = list(column.keys())[0]
  regiotype_code = f'{regiotype}Code'

  df = df[df.WekenTerug < maxweeks]
  if 'scale' in df:
    df = df.assign(count=df[datacolumn] * df.scale)
  else:
    df = df.assign(count=df[datacolumn])

  df = (df
    .groupby([regiotype_code, 'WekenTerug'])[datacolumn]
    .sum()
    .unstack(level=-1)
    .reindex(columns=np.arange(maxweeks), fill_value=0)
  )

  merges = []
  # must be done before colors is merged and before the columns are renamed
  if highestweek:
    merges.append(df.idxmax(axis=1).to_frame().rename(columns={0: f'{label} hoogste week'}))

  df.columns = [f'{label} w{-col}' for col in df.columns.values]

  # must be done before highestweek is merged but after the columns are renamed
  if colors:
    merges.append((df.divide(df.max(axis=1), axis=0) * 1000).rename(columns={col:col.replace(' w', ' cw') for col in df}))

  for extra in merges:
    df = df.merge(extra, left_index=True, right_index=True)

  # bij ontbreken van w-1 vaste waarde 9.99
  df[f'{label} t.o.v. vorige week'] = (df[f'{label} w0'] / df[f'{label} w-1']).replace(np.inf, 9.99)
  return df

def collect(regiotype):
  regiotype_code = f'{regiotype}Code'

  global aantallen, ziekenhuisopnames, ziekenhuisopnames_1

  pos_dec = sumcols(aantallen, regiotype, {'Total_reported':'Positief getest', 'Deceased':'Overleden'})
  pos_dec_delta = sumcols(
    aantallen[aantallen.Date == aantallen.Today],
    regiotype,
    {'Total_reported':'Positief getest (toename)', 'Deceased':'Overleden (toename)'}
  )
  admissions = sumcols(ziekenhuisopnames, regiotype, {'Hospital_admission':'Ziekenhuisopname'})
  admissions_1 = sumcols(ziekenhuisopnames_1, regiotype, {'Hospital_admission':'Ziekenhuisopname_1'})
  admissions_delta = admissions.merge(admissions_1, how='left', on=regiotype_code)
  admissions_delta['Ziekenhuisopname (toename)'] = admissions_delta.Ziekenhuisopname - admissions_delta.Ziekenhuisopname_1
  del admissions_delta['Ziekenhuisopname']
  del admissions_delta['Ziekenhuisopname_1']

  df = (groupregio(regiotype)
    .merge(pos_dec, how='left', left_on='Code', right_index=True)
    .merge(admissions, how='left', left_on='Code', right_index=True)
    .merge(pos_dec_delta, how='left', left_on='Code', right_index=True)
    .merge(admissions_delta, how='left', left_on='Code', right_index=True)
    .fillna(0)
  )
  # per 100k voor de absolute kolommen
  for cat in [pos_dec, admissions]:
    for col in cat.columns:
      df[col + ' per 100.000'] = ((df[col] * 100000) / df.Personen).replace(np.inf, 0)

  df['Positief getest 1d/100k'] = (df['Positief getest (toename)'] / df.Personen).replace(np.inf, 0)
  df['Positief getest percentage'] = (df['Positief getest'] / df.Personen).replace(np.inf, 0)
  df['Positief getest per km2'] = (df['Positief getest'] / df['Opp land km2']).replace(np.inf, 0)

  maxweeks = 26
  df = (df
    .merge(histcols(
      aantallen,
      regiotype,
      maxweeks=maxweeks,
      colors=True,
      highestweek=True,
      column={'Total_reported':'Positief getest'}), how='left', left_on='Code', right_index=True)
    .merge(histcols(
      aantallen.merge(df.assign(scale=100000 / df.Personen)[['Code', 'scale']], left_on=regiotype_code, right_on='Code'),
      regiotype,
      maxweeks=maxweeks,
      column={'Total_reported':'Positief getest 7d/100k'}), how='left', left_on='Code', right_index=True)
    .merge(histcols(ziekenhuisopnames,
      regiotype,
      maxweeks=maxweeks,
      colors=True,
      column={'Hospital_admission':'Ziekenhuisopname'}), how='left', left_on='Code', right_index=True)
    .merge(histcols(
      aantallen,
      regiotype,
      maxweeks=maxweeks,
      colors=True,
      column={'Deceased':'Overleden'}), how='left', left_on='Code', right_index=True)
  )

  return df

# %%
@run('regio: load verschillende aggregatie niveaus')
def cell():
  global regios
  regios = pd.concat([
    collect(regiotype)
    for regiotype in
    [ 'Gemeente', 'GGDregio', 'Veiligheidsregio', 'Provincie', 'Landsdeel', 'Schoolregio', 'Land' ]
  ])
  regios.loc[regios.Code == 'GM0000']['VeiligheidsregioCode'] = ''
  regios = regios.rename(columns={
    'LandCode': 'Landcode',
    'VeiligheidsregioCode': 'Veiligheidsregio Code',
    'GGDregio': 'GGD regio'
  })



# %%
order = pd.read_csv('columnorder.csv')
publish(regios[order.columns.values].fillna(0))

# %%
