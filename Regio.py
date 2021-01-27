# %%
from IPython import get_ipython
from IPython.display import clear_output
get_ipython().run_line_magic('run', 'setup')
import itertools

# sorteert de kolommen alfabetisch, is makkelijker visueel te debuggen.
def sortcolumns(df):
  return df[sorted(df.columns)]

# %%
@run('regio: load regios en hun basisgegevens')
def cell():
  global gemeenten
  # rename de kolommen naar "Naam" + "NaamCode" voor makkelijker uniforme data bewerking
  gemeenten = pd.read_csv('gemeenten.csv').rename(columns={
    'Code': 'GemeenteCode',
    'Naam': 'Gemeente',
    'Veiligheidsregio Code': 'VeiligheidsregioCode',
    'GGD regio': 'GGDregio',
    'Landcode': 'LandCode',
  })
  # niet nodig want die voegen we vanzelf toe bij de per-type constructie van de cijfers
  del gemeenten['Type']
  
  global leeftijdsgroepen
  leeftijdsgroepen = pd.read_csv('leeftijdsgroepen.csv')
  del leeftijdsgroepen['Type']
  lgb = CBS.leeftijdsgroepen_bevolking().reset_index()
  lgb['Code'] = 'LE' + lgb['Range'].replace({'0-9': '00-09'}).replace('-', '', regex=True).astype(str)
  lgb = lgb.rename(columns={'BevolkingOpDeEersteVanDeMaand': 'Personen'})
  leeftijdsgroepen = leeftijdsgroepen.merge(lgb[['Code', 'Personen']], how='left', on='Code')

  global regiocodes
  regiocodes = pd.read_csv('regiocodes.csv')
  # sluit aan bij de uniforme naamgeving van hierboven
  regiocodes = regiocodes.rename(columns={'Landcode': 'LandCode'})
  regiocodes.loc[regiocodes.Type == 'GGD', 'Type'] = 'GGDregio'
  
  # voeg de regiocodes toe aan de gemeenten tabel voor makkelijker uniforme data bewerking
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

  # lege regel voor GM0000
  for regiotype, prefix in [('GGDregio', 'GG'), ('Veiligheidsregio', 'VR'), ('Provincie', 'PV'), ('Landsdeel', 'LD'), ('Schoolregio', 'SR')]:
    gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', regiotype] = ''
    gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', f'{regiotype}Code'] = f'{prefix}00'
  gemeenten.loc[gemeenten.GemeenteCode == 'GM0000', 'LandCode'] = 'NL'
  
  base = 'https://opendata.cbs.nl/ODataApi/OData/37230ned'
  
  # voor perioden pak de laatste
  periode = CBS.odata(base + '/Perioden').iloc[[-1]]['Key'].values[0]
  
  # startsWith would have been better to do in the filter but then the CBS "odata-ish" server responds with
  # "Object reference not set to an instance of an object."
  bevolking = CBS.odata(base + f"/TypedDataSet?$filter=(Perioden eq '{periode}')&$select=RegioS, BevolkingAanHetBeginVanDePeriode_1")
  # want de CBS odata API snap startsWith niet...
  bevolking = bevolking[bevolking.RegioS.str.startswith('GM')]
  # die _1 betekent waarschijnlijk dat het gedrag ooit gewijzigd is en er een nieuwe "versie" van die kolom is gepubliceerd
  bevolking.rename(columns={'RegioS': 'GemeenteCode', 'BevolkingAanHetBeginVanDePeriode_1': 'BevolkingAanHetBeginVanDePeriode'}, inplace=True)
  
  gemeenten = gemeenten.merge(bevolking, how='left', on='GemeenteCode')
  # default naar gegeven waarde bij ontbrekende data
  gemeenten.loc[gemeenten.Personen == 0, 'Personen'] = gemeenten.BevolkingAanHetBeginVanDePeriode
  del gemeenten['BevolkingAanHetBeginVanDePeriode']
  gemeenten = sortcolumns(gemeenten)

  # prepareer een RIVM dataset
  def prepare(dataset, day):
    df = RIVM.load(dataset, day)
    # hernoem kolommen voor makkelijker uniforme data bewerking
    for old, new in [('Municipality_code', 'GemeenteCode'), ('Security_region_code', 'VeiligheidsregioCode'), ('Security_region_name', 'Veiligheidsregio')]:
      if old in df:
        df[new] = df[old]
    if 'GemeenteCode' in df:
      df['GemeenteCode'] = df['GemeenteCode'].fillna('GM0000')

    if 'Agegroup' in df:
      df['LeeftijdCode'] = 'LE' + df['Agegroup'].replace({'0-9': '00-09', '<50': '00-00', 'Unknown': '00-00', 'Onbekend': '00-00'}).replace('-', '', regex=True).astype(str)
      df['Total_reported'] = 1 # impliciet in casus-landelijk
      df = df.replace({'Hospital_admission': {'Yes': 1, 'No': 0, 'Unknown': 0}, 'Deceased': {'Yes': 1, 'No': 0, 'Unknown': 0}})

    # voeg regiocodes to aan elke regel in de dataset
    if 'GemeenteCode' in df:
      for regiotype in ['GGDregio', 'Provincie', 'Landsdeel', 'Schoolregio']:
        df = df.merge(gemeenten[['GemeenteCode', f'{regiotype}Code']].drop_duplicates(), on='GemeenteCode')

    # als er geen gemeentecode is, maar misschien wel een VR code, vervang die door VR00
    if 'GemeenteCode' in df and 'VeiligheidsregioCode' in df:
      df.loc[df.GemeenteCode == 'GM0000', 'VeiligheidsregioCode'] = 'VR00'
      df.loc[df.GemeenteCode == 'GM0000', 'Veiligheidsregio'] = ''

    df['LandCode'] = 'NL'
    df['Land'] = 'Nederland'
  
    # knip de tijd van de datum af, en stop hem in 'Today' (referentiepunt metingen)
    if 'Date_of_report' in df:
      df['Datum'] = df.Date_of_report.str.replace(' .*', '', regex=True)
    elif 'Date_file' in df:
      df['Datum'] = df.Date_file.str.replace(' .*', '', regex=True)
    df['Today'] = pd.to_datetime(df.Datum)
  
    # zet 'Date' naar de bij de betreffende dataset horende meetdatum-kolom
    for when in ['Date_statistics', 'Date_of_statistics', 'Date_of_publication']:
      if when in df:
        df['Date'] = pd.to_datetime(df[when])
        # en direct maar weken terug, die hebben we vaker nodig
        df['WekenTerug'] = ((df.Today - df.Date) / np.timedelta64(7, 'D')).astype(np.int)

    return sortcolumns(df)

  global aantallen, ziekenhuisopnames, ziekenhuisopnames_1, casus_landelijk, casus_landelijk_1
  aantallen = prepare('COVID-19_aantallen_gemeente_per_dag', 0)
  ziekenhuisopnames = prepare('COVID-19_ziekenhuisopnames', 0)
  ziekenhuisopnames_1 = prepare('COVID-19_ziekenhuisopnames', 1)
  casus_landelijk = prepare('COVID-19_casus_landelijk', 0)
  casus_landelijk_1 = prepare('COVID-19_casus_landelijk', 1)

# %% Ondersteunde code voor de regio berekeningen
def groupregio(regiotype):
  """
    Groepeer de gemeenten tabel op gegeven regiotypen en sommeer personen/oppervlakte
  """
  global gemeenten

  grouping = [ f'{regiotype}Code', regiotype]
  if regiotype != 'Land':
    grouping += [ 'LandCode' ]

  # deze kolommen willen we voor de resultset, ook al zijn ze voor alles behalve 'Gemeente' leeg
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
    # hier hoeven we niets te doen dan de juiste kolommen te selecteren
    df = gemeenten[grouping + columns + ['Personen', 'Opp land km2']].rename(columns={'Gemeente': 'Naam', 'GemeenteCode': 'Code'})
  elif regiotype == 'Leeftijd':
    df = (leeftijdsgroepen
      # voeg de lege kolommen toe
      .assign(**{ col: '' for col in columns})
      .assign(**{'Opp land km2': 0})
    )
  else:
    df = (gemeenten[gemeenten.GemeenteCode != 'GM0000']
      # groupeer op regiotype, sommeer oppervlakte en personen
      .groupby(grouping).agg({ 'Personen': 'sum', 'Opp land km2': 'sum' })
      .reset_index()
      # standardiseerd de 'Naam' en 'Code' kolommen zodat ze klaar zijn voor output.
      .rename(columns={regiotype: 'Naam', f'{regiotype}Code': 'Code'})
      # voeg de lege kolommen toe
      .assign(**{ col: '' for col in columns })
    )
    if regiotype == 'Land':
      df['Land'] = df['Naam']
      df['LandCode'] = df['Code']
  # voeg het regiotype toe in de Type kolom
  return df.assign(Type=regiotype)

def sumcols(df, regiotype, columns):
  """
    groepeer en sommeer deceased/admission/positive
  """
  regiotype_code = f'{regiotype}Code'
  return (df
    # groepeer op regiotype en selecteer de gewenste kolommen
    .groupby([regiotype_code])[list(columns.keys())]
    # sommeer
    .sum()
    # rename naar gewenste output kolommen
    .rename(columns=columns)
  )

def histcols(df, regiotype, maxweeks, column, colors=False, highestweek=False):
  """
    voeg week historie toe:
    - regiotype
    - maxweeks = hoeveel weken
    - column = deceased/admission/positive naam => output kolom
    - colors = toevoegen schaalverdeling kleuren
    - highestweek = toevoegen op welke week het maximum was behaald
  """
  # in principe zouden we kunnen groeperen/sommeren op meerdere kolommen tegelijk maar dan worden colors en highestweek weer heel complex
  assert len(column) == 1
  label = list(column.values())[0]
  datacolumn = list(column.keys())[0]
  regiotype_code = f'{regiotype}Code'

  # knip alle data van >= maxweeks eruit
  df = df[df.WekenTerug < maxweeks]

  # als we schalen naar 100.000, voor hoeveel telt elke melding dan mee
  if 'scale' in df:
    df = df.assign(count=df[datacolumn] * df.scale).replace(np.inf, 0)
  else:
    df = df.assign(count=df[datacolumn])

  df = (df
    # groepeer op reguitype en wekenterug
    .groupby([regiotype_code, 'WekenTerug'])['count']
    # optellen (de aantallen zijn eventueel hierboven al geschaald)
    .sum()
    # maak de week nummers kolommen
    .unstack(level=-1)
    # en vul de kolommen uit zodat als een week helemaal geen meldingen heeft dat die niet weg blijft maar 0 bevat
    .reindex(columns=np.arange(maxweeks), fill_value=0)
  )

  merges = []
  # must be done before colors is merged and before the columns are renamed
  if highestweek:
    merges.append(df.idxmax(axis=1).to_frame().rename(columns={0: f'{label} hoogste week'}))

  # hernoem de kolommen van de getallen die ze nu zijn
  df.columns = [f'{label} w{-col}' for col in df.columns.values]

  # must be done before highestweek is merged but after the columns are renamed
  if colors:
    # kleurkolommen zijn waarde van de week gedeeld door het maximum over de weken heen
    merges.append((df.divide(df.max(axis=1), axis=0) * 1000).rename(columns={col:col.replace(' w', ' cw') for col in df}))

  for extra in merges:
    df = df.merge(extra, left_index=True, right_index=True)

  # bij ontbreken van w-1 vaste waarde 9.99
  df[f'{label} t.o.v. vorige week'] = (df[f'{label} w0'] / df[f'{label} w-1']).replace(np.inf, 9.99)
  return df

def collect(regiotype):
  """
    verzamel alle kolommen voor gegeven regiotype
  """
  regiotype_code = f'{regiotype}Code'

  def datasets():
    if regiotype == 'Leeftijd':
      global casus_landelijk, casus_landelijk_1
      return (casus_landelijk, casus_landelijk, casus_landelijk_1)
    else:
      global aantallen, ziekenhuisopnames, ziekenhuisopnames_1
      return (aantallen, ziekenhuisopnames, ziekenhuisopnames_1)

  aantallen, ziekenhuisopnames, ziekenhuisopnames_1 = datasets()

  assert len(aantallen.Datum.unique()) == 1
  assert len(ziekenhuisopnames_1.Datum.unique()) == 1
  assert list(aantallen.Datum.unique()) == list(ziekenhuisopnames.Datum.unique()), list(aantallen.Datum.unique()) + list(ziekenhuisopnames.Datum.unique())

  # sommeer Total_reported en Deceased voor gegeven regiotype
  pos_dec = sumcols(aantallen, regiotype, {'Total_reported':'Positief getest', 'Deceased':'Overleden'})
  # toename is precies hetzelfde maar dan alleen voor de metingen van 'vandaag'
  pos_dec_delta = sumcols(
    aantallen[aantallen.Date == aantallen.Today],
    regiotype,
    {'Total_reported':'Positief getest (toename)', 'Deceased':'Overleden (toename)'}
  )
  # sommeer Hospital_admission voor gegeven regiotype
  admissions = sumcols(ziekenhuisopnames, regiotype, {'Hospital_admission':'Ziekenhuisopname'})
  # sommeer Hospital_admission van 'vorige dag' voor gegeven regiotype
  admissions_1 = sumcols(ziekenhuisopnames_1, regiotype, {'Hospital_admission':'Ziekenhuisopname_1'})
  # voeg ze bij elkaar en trek ze van elkaar af
  admissions_delta = admissions.merge(admissions_1, how='left', on=regiotype_code)
  admissions_delta['Ziekenhuisopname (toename)'] = admissions_delta.Ziekenhuisopname - admissions_delta.Ziekenhuisopname_1
  # niet meer nodig en vervuilen anders het resultaat
  del admissions_delta['Ziekenhuisopname']
  del admissions_delta['Ziekenhuisopname_1']

  # groupeer de gemeenten op gegeven regiotype
  df = (groupregio(regiotype)
    # en voeg de berekende kolommen toe
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

  df['Positief getest 1d/100k'] = ((df['Positief getest (toename)'] * 100000) / df.Personen).replace(np.inf, 0)
  df['Positief getest percentage'] = (df['Positief getest'] / df.Personen).replace(np.inf, 0)
  df['Positief getest per km2'] = (df['Positief getest'] / df['Opp land km2']).replace(np.inf, 0)

  # weekhistories
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
  df['Datum'] = aantallen.Datum.unique()[0]

  return df

# %%
@run('regio: load verschillende aggregatie niveaus')
def cell():
  global regios
  # verzamel de data voor de gegeven regiotypes en plak ze onder elkaar
  regios = pd.concat([
    collect(regiotype)
    for regiotype in
    [
      'Gemeente',
      'GGDregio',
      'Veiligheidsregio',
      'Provincie',
      'Landsdeel',
      'Schoolregio',
      'Land',
      'Leeftijd',
    ]
  ])
  # maak de kolommen leeg voor GM0000
  # hernoem de eerder geuniformeerde kolomen terug naar de gewenste output
  regios = regios.rename(columns={
    'LandCode': 'Landcode',
    'VeiligheidsregioCode': 'Veiligheidsregio Code',
    'GGDregio': 'GGD regio'
  })

# %%
# load de gewenste kolom volgorde uit een file en publiceer
async def publish(df):
  df2 = df.set_index('Code')
  m = (df2 == np.inf)
  df2 = df2.loc[m.any(axis=1), m.any(axis=0)]
  display(df2.head())

  os.makedirs('artifacts', exist_ok = True)
  df.to_csv('artifacts/gemeenten.csv', index=True)

  if knack:
    print('updating knack')
    await knack.update(objectName='RegioV2', df=df)

order = pd.read_csv('columnorder.csv')
await publish(regios[order.columns.values].fillna(0))
