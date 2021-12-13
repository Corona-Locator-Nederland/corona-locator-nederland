# %%
from IPython import get_ipython
from IPython.core.display import display
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('run', 'setup')

# %% leeftijdsgroepen: download RIVM data
#leeftijdsgroepen = SimpleNamespace()
@run
def cell():
  global rivm
  rivm = RIVM.csv('COVID-19_casus_landelijk')
  display(rivm.head())

# %% Download de bevolkings cijfers van CBS, uitgesplitst op de leeftijds categorien in de dataset van het RIVM
@run
def cell():
  global bevolking
  bevolking = CBS.bevolking(leeftijdsgroepen=True)

# %% leeftijdsgroepen: prepareer tabel
# Bereken de stand van zaken van besmettingen / hospitalisaties / overlijden, per cohort in absolute aantallen en aantallen per 100k, met een kleur indicator voor de aantallen.
# vervang <50 en Unknown door Onbekend
@run
def cell():
  rivm['Cohort'] = rivm['Agegroup'].replace({'<50': 'Onbekend', 'Unknown': 'Onbekend'})
  # aangenomen 'gemiddelde' leeftijd van een cohort: minimum waarde + 5
  assumed_cohort_age = [(cohort, [int(n) for n in cohort.replace('+', '').split('-')]) for cohort in rivm['Cohort'].unique() if cohort[0].isdigit()]
  assumed_cohort_age = { cohort: min(rng) + 5 for cohort, rng in assumed_cohort_age }
  rivm['Gemiddelde leeftijd'] = rivm['Cohort'].apply(lambda x: assumed_cohort_age.get(x, np.nan))

  # verwijder tijd
  rivm['Date_file_date'] = pd.to_datetime(rivm['Date_file'].replace(r' .*', '', regex=True))

  rivm['Date_statistics_date'] = pd.to_datetime(rivm['Date_statistics'])

  # weken terug = verschil tussen Date_file en Date_statistcs, gedeeld door 7 dagen
  rivm['Weken terug'] = np.floor((rivm['Date_file_date'] - rivm['Date_statistics_date'])/np.timedelta64(7, 'D')).astype(int)

  # voeg key, gem leeftijd, kleurnummer en totaal toe
  Date_file = rivm['Date_file_date'].unique()[0].astype('M8[D]').astype('O')
  cohorten = list(bevolking.index) + ['Onbekend']
  def summarize(df, category, prefix):
    # aangezien we hier de dataframe in-place wijzigen (bijv door toevoegen kolommen)
    # en we het 'rivm' frame later nog clean nodig hebben
    df = df.copy(deep=True)

    df = (df
          .groupby(['Weken terug', 'Cohort'])['count']
          .sum()
          .unstack(fill_value=np.nan)
          .reset_index()
          .rename_axis(None, axis=1)
        ).merge(df
          # we voegen hier gemiddelde leeftijd toe, want die willen we op een ander
          # niveau aggregeren voor 'df' overschreven word
          .groupby(['Weken terug'])['Gemiddelde leeftijd']
          .mean()
          .to_frame(), on='Weken terug'
        )

    # altijd 52 rijen
    df = pd.Series(np.arange(52), name='Weken terug').to_frame().merge(df, how='left', on='Weken terug')

    # toevoegen missende cohorten
    for col in cohorten:
      if not col in df:
        df[col] = np.nan

    # sommeer per rij (axis=1) over de cohorten om een totaal te krijgen
    df['Totaal'] = df[cohorten].sum(axis=1)

    # voeg periode en datum toe
    # periode afgeleid van weken-terug (= de index voor deze dataframe)
    df['Datum'] = pd.to_datetime(Date_file)
    df['Periode'] = (df
      .index.to_series()
      .apply(
        lambda x: (
          (Date_file + datetime.timedelta(weeks=-(x+1), days=1)).strftime('%d/%m')
          + '-'
          + (Date_file + datetime.timedelta(weeks=-x)).strftime('%d/%m')
        )
      )
    )

    # voeg 'Key' en 'Type' kolom toe. Variabele 'type' kan niet, is een language primitive.
    df['Key'] = prefix + df.index.astype(str).str.rjust(3, fillchar='0')
    df['Type'] = category

    # voeg de kleur kolommen toe
    for col in cohorten:
      df['c' + col] = ((df[col] / df[[col for col in cohorten]].max(axis=1)) * 1000).fillna(0).astype(int)

    # herschikken van de kolommen
    colorder = ['Key', 'Weken terug', 'Datum', 'Periode', 'Gemiddelde leeftijd', 'Totaal', 'Type']
    return df[colorder + [col for col in df if col not in colorder]]

  factor = bevolking.to_dict()['per 100k']
  global tabel
  tabel = pd.concat(
    # flatten the result list zodat pd.concat ze onder elkaar kan plakken
    functools.reduce(lambda a, b: a + b, [
      [summarize(df.assign(count=1), label, prefix), summarize(df.assign(count=df['Cohort'].apply(lambda x: factor.get(x, np.nan))), label + ' per 100.000', prefix + '100k')]
      for df, label, prefix in [
        (rivm, 'Positief getest', 'p'), # volledige count per cohort
        (rivm[rivm.Hospital_admission == 'Yes'], 'Ziekenhuisopname', 'h'), # count van cohort voor Hospital_admission == 'Yes'
        (rivm[rivm.Deceased == 'Yes'], 'Overleden', 'd'), # count van cohort voor Deceased == 'Yes'
      ]
    ])
  )

  # rood -> groen
  cdict = {
    'red':   ((0.0, 0.0, 0.0),   # no red at 0
              (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
              (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1
    'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
              (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
              (1.0, 0.0, 0.0)),  # no green at 1
    'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
              (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
              (1.0, 0.0, 0.0))   # no blue at 1
  }
  cm = colors.LinearSegmentedColormap('GnRd', cdict)
  # geel -> paars
  cm = sns.color_palette('viridis_r', as_cmap=True)
  display(tabel
    .fillna(0)
    .head()
    .round(1)
    .reset_index(drop=True)
    .style.background_gradient(cmap=cm, axis=1, subset=cohorten)
  )

# %% publish
if knack:
  await knack.publish(tabel.fillna(0).assign(Datum=tabel.Datum.dt.strftime('%Y-%m-%d')), 'Leeftijdsgroep', Cache)
# %% setup
from IPython import get_ipython
from IPython.core.display import display
from typing_extensions import runtime
import numpy as np
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('run', 'setup')

# sorteert de kolommen alfabetisch, is makkelijker visueel te debuggen.
def sortcolumns(df):
  return df[sorted(df.columns)]

# prepareer een RIVM dataset
def prepare(dataset, day=0):
  # hernoem kolommen voor makkelijker uniforme data bewerking
  df = RIVM.csv(dataset, day).rename(columns={
    'Municipality_code': 'GemeenteCode',
    'Municipality_name': 'Gemeente',
    'Security_region_code': 'VeiligheidsregioCode',
    'Security_region_name': 'Veiligheidsregio',
  })
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
      df['WekenTerug'] = ((df.Today - df.Date) / np.timedelta64(7, 'D')).astype(int)

  return sortcolumns(df).sort_values(by=['Date'])

# %% regio: load regios en hun basisgegevens
@run
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
  lgb = CBS.bevolking(leeftijdsgroepen=True).reset_index()
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

# %% regio: load verschillende aggregatie niveaus
@run
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

# %% publiceer RegioV2
if knack:
  await knack.publish(regios[[col for col in regios.columns if col != 'Land']].fillna(0), 'RegioV2', Cache)

# %% Regioposten  
@ignore
def cell():
  # just so the timestamps update OK
  Cache.reset()

  import sys
  aantallen = prepare('COVID-19_aantallen_gemeente_per_dag').assign(Hospital_admission=np.nan)
  aantallen['Week'] = aantallen['Date'].dt.strftime('%Y-%V')
  ziekenhuisopnames = prepare('COVID-19_ziekenhuisopnames').assign(Total_reported=np.nan, Deceased=np.nan)
  ziekenhuisopnames['Week'] = ziekenhuisopnames['Date'].dt.strftime('%Y-%V')

  regiotypes = [ 'GGDregio', 'Gemeente', 'Land', 'Landsdeel', 'Provincie', 'Schoolregio', 'Veiligheidsregio' ]
  regiocodecols = [f'{regiotype}Code' for regiotype in regiotypes]
  datafields = [ 'Date', 'Week', 'Hospital_admission', 'Total_reported', 'Deceased']
  rivm = pd.concat([
    aantallen[regiocodecols + datafields],
    ziekenhuisopnames[regiocodecols + datafields],
  ], axis=0)

  weken = list(pd.date_range(
    start=min(aantallen.Date.min(), ziekenhuisopnames.Date.min()),
    end=max(aantallen.Date.max(), ziekenhuisopnames.Date.max())
  ).strftime('%Y-%V').unique())
  print(len(weken), 'weken')
  week = ['ma', 'di', 'wo', 'do', 'vr', 'za', 'zo']
  metrics = [
    'Total_reported',
    'Deceased',
    'Hospital_admission',
  ]

  global regioposten
  regioposten = []
  for regiotype in regiotypes:
    print(regiotype)
    sys.stdout.flush()
    code = regiotype + 'Code'

    df = (rivm
      .sort_values(by=[code, 'Date'])
      .groupby([code, 'Week', 'Date'])
      .agg({ metric: 'sum' for metric in metrics })
      .reset_index()
    )
    for dow, dayname in enumerate(week):
      for metric in metrics:
        df.loc[df['Date'].dt.dayofweek == dow, f'{metric} {dayname}'] = df[metric]

    df = (df
      .sort_values(by=[code, 'Date'])
      .groupby([code, 'Week'])
      .agg({
        'Date': [ 'max', 'nunique' ],
        'Total_reported': [ 'sum', 'last' ],
        'Deceased': [ 'sum', 'last' ],
        'Hospital_admission': [ 'sum', 'last' ],
        **{
          f'{metric} {day}': 'sum'
          for metric in metrics
          for day in week
        }
      })
      .reset_index()
    )
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.rename(columns={
      code: 'Code',
      'Date max': 'Datum',
      'Date nunique': 'Dagen',
      'Total_reported sum': 'Positief getest week',
      'Total_reported last': 'Positief getest laatste dag',
      'Deceased sum': 'Overleden week',
      'Deceased last': 'Overleden laatste dag',
      'Hospital_admission sum': 'Ziekenhuisopname week',
      'Hospital_admission last': 'Ziekenhuisopname laatste dag',
      **{ f'Total_reported {day} sum': f'Positief getest {day}' for day in week },
      **{ f'Deceased {day} sum': f'Overleden {day}' for day in week },
      **{ f'Hospital_admission {day} sum': f'Ziekenhuisopname {day}' for day in week },
    }, inplace=True)

    regio = groupregio(regiotype)
    regio = regio[[col for col in regio.columns if col == 'Code' or 'Code' not in col and 'km2' not in col]]
    df = df.merge(regio, how='left', on='Code')
    for col, coltype in zip(regio.columns, regio.dtypes):
      if col == 'Personen':
        df[col] = df[col].fillna(0).astype(int)
      elif col == 'Type':
        df[col] = df[col].fillna(regiotype)
      elif coltype == np.float64:
        df[col] = df[col].fillna(0)
      elif coltype == object:
        df[col] = df[col].fillna('')
      else:
        raise KeyError(col)
    df['Landcode'] = 'NL'

    base = [
      (code, week)
      for code in list(df.Code.unique())
      for week in weken
    ]
    base = pd.DataFrame({
        'Week': [week for code, week in base],
        'Code': [code for code, week in base],
      },
      index=['-'.join(codeweek) for codeweek in base]
    )

    df['Naam + Type'] = df['Naam'] + ' (' + df['Type'] + ')'
    df['Key'] = df['Code'] + '-' + df['Week']
    df.set_index('Key', inplace=True)
    df.drop(columns=['Code', 'Week'], inplace=True)

    df = base.join(df)

    for col in ['Positief getest', 'Overleden', 'Ziekenhuisopname']:
      # vul missende waarden met 0 vanaf de eerstgevonden waarde zodat cumsum goed werkt
      #df[f'{col} week'] = df[f'{col} week'].mask(df[f'{col} week'].ffill().isnull(), 0)
      #df[f'{col} cumulatief'] = df[f'{col} week'].cumsum().fillna(0)
      #df[f'{col} week -1'] = df[f'{col} cumulatief'].shift(1).fillna(0)
      df[f'{col} week'] = df[f'{col} week'].fillna(0)
      df[f'{col} week -1'] = df[f'{col} week'].shift(1).fillna(0)

    df.index.rename('Key', inplace=True)
    df = df[[col for col in df.columns if col != 'Land']]
    regioposten.append(df.reset_index())
  regioposten = pd.concat(regioposten, axis=0)
  regioposten['Datum'] = regioposten['Datum'].dt.strftime('%Y-%m-%d')
  regioposten.rename(columns={'GGDregio': 'GGD regio'}, inplace=True)

  display(regioposten[['Week'] + [col for col in regioposten.columns if 'Positief getest' in col]].head())
# %%
# if knack:
#   await knack.publish(regioposten, 'Regioposten', Cache)

# %%
@ignore
def cell():
  def diff(naam):
    def make(dag):
      df = RIVM.csv(naam, dag).drop(columns=['Date_of_report'])
      for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, int]:
          df[col] = df[col].fillna(0)
        else:
          df[col] = df[col].fillna('')
      df.insert(0, 'id', df.index)
      return df

    vandaag = make(0)
    if 'Date_of_publication' in vandaag:
      dop = 'Date_of_publication'
    else:
      dop = 'Date_of_statistics'
    delta = vandaag[dop].max()
    vandaag = vandaag[vandaag[dop]!= delta]
    gisteren = make(1)

    df_all = gisteren.merge(vandaag, how='outer', on=[str(col) for col in gisteren.columns], indicator=True)
    gisteren=df_all[df_all._merge == 'left_only'].drop(columns=['_merge'])
    vandaag=df_all[df_all._merge == 'right_only'].drop(columns=['_merge'])

    assert len(vandaag) == len(gisteren), (len(vandaag), len(gisteren))

    df_all = pd.concat([gisteren.set_index('id'), vandaag.set_index('id')], axis='columns', keys=['Gisteren', 'Vandaag'])
    df_all = df_all.swaplevel(axis='columns')[gisteren.columns[1:]]
    def highlight_diff(data):
      attr = 'background-color: yellow'
      other = data.xs('Gisteren', axis='columns', level=-1)
      return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''), index=data.index, columns=data.columns)

    df_all.style.apply(highlight_diff, axis=None).to_excel(f'diff-{naam}.xlsx')

  diff('COVID-19_aantallen_gemeente_per_dag')
  diff('COVID-19_ziekenhuisopnames')
# %% setup
from IPython import get_ipython
from IPython.core.display import display
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('run', 'setup')

bevolking = CBS.bevolking().iloc[0]
#display(bevolking.BevolkingOpDeEersteVanDeMaand)
#display(bevolking['per 100k'])

def addstats(df):
  global dagoverzicht

  # de aanname is dat df al gegroepeerd is op datum, en dat de kolommen de dagtotalen zijn
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for stat in df.columns:
    # nieuw = dagtotaal, zet missende dagtotalen op 0
    dagoverzicht[f'{stat} (nieuw)'] = dagoverzicht[stat].fillna(0).astype(int)
    # kolom met eerder de dagtotalen = cumulatieve som over dagtotalen
    dagoverzicht[stat] = dagoverzicht[f'{stat} (nieuw)'].cumsum()
    # factor voor 100k
    dagoverzicht[f'{stat} per 100.000'] = dagoverzicht[stat] * bevolking['per 100k']
    # verschil met 7 dagen terug
    dagoverzicht[f'{stat} 7d'] = dagoverzicht[f'{stat} (nieuw)'].rolling(7).sum().fillna(0)
    # en weer factor 100 k
    dagoverzicht[f'{stat} 7d per 100.000'] = dagoverzicht[f'{stat} 7d'] * bevolking['per 100k']

# %% set up base frame from ESRI -> NICE
@run
def cell():
  df = ArcGIS.nice('f27f743476a142538e8054f7a7ce12e1')

  df['date'] = pd.to_datetime(df.date.str.replace(' .*', '', regex=True))
  df.set_index('date', inplace=True)

  # base date range uit NICE
  global dagoverzicht
  # maak leeg dataframe met een rij voor elke dag van 2020-02-27 tm Date_of_publication
  dagoverzicht = pd.DataFrame(index=pd.date_range(start='2020-02-27', end=df.index.max()))
  # noem de index Key
  dagoverzicht.index.name='Key'
  # vul de datum kolom
  dagoverzicht['Datum'] = dagoverzicht.index.strftime('%Y-%m-%d')
  # vaste waarde voor LandCode
  dagoverzicht['LandCode'] = 'NL'

  # the NICE data as prepared by ESRI can just be used as-is, so just rename the columns
  for prefix, kind in [ ('ic', 'IC'), ('zkh', 'Ziekenhuis') ]:
    df = df.rename(columns={
      f'{prefix}NewIntake': f'NICE {kind} Bedden (intake)',
      f'{prefix}IntakeCount': f'NICE {kind} Bedden',
      f'{prefix}IntakeCumulative': f'NICE {kind} Bedden (cumulatief)',
      f'{prefix}DiedCumulative': f'NICE {kind} Overleden',
    })

  # remove the columns we don't use
  df = df[[col for col in df.columns if 'NICE' in col]]
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)

  # after the merge, dates missing in the ESRI/NICE set will be `nan`, so fill these
  for col in df.columns:
    # for cumulatief columns, fill-forward from last known value
    if 'cumulatief' in col:
      dagoverzicht[col] = dagoverzicht[col].ffill()
    # for non-cumulative and whatever remains in cumulative after fill forward (which will be leading nan's), fill with 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0)

  for kind in [ 'IC', 'Ziekenhuis' ]:
    dagoverzicht[f'NICE {kind} Bedden (toename)'] = (dagoverzicht[f'NICE {kind} Bedden'] - dagoverzicht[f'NICE {kind} Bedden'].shift(1)).fillna(0)

  for window, shift, past in [(3, 1, 'afgelopen '), (7, 0, '')]:
    dagoverzicht[f'NICE IC Bedden (intake) {past}{window}d'] = dagoverzicht['NICE IC Bedden (intake)'].shift(shift).rolling(window).sum().fillna(0)

  dagoverzicht['NICE IC Bedden (intake) week-1'] = dagoverzicht['NICE IC Bedden (intake) 7d'].shift(7).fillna(0)

# %% overleden + positief getest
@run
def cell():
  df = RIVM.csv('COVID-19_aantallen_gemeente_per_dag').rename(columns={
    'Total_reported': 'Positief getest',
    'Deceased': 'Overleden',
    'Date_of_publication': 'Datum',
    'Date_of_report': 'Today',
  })
  # sloop tijd van de datum en zet om in datetime object
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))

  # sommeer pos en overl op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Positief getest': 'sum', 'Overleden': 'sum'}))
  global dagoverzicht
  for col in ['Overleden', 'Positief getest']:
    dagoverzicht[f'{col} week-1'] = dagoverzicht[f'{col} 7d'].shift(7).fillna(0)
  display(dagoverzicht.head(10))

# %% ziekenhuisopnames
@run
def cell():
  df = RIVM.csv('COVID-19_ziekenhuisopnames').rename(columns={
    'Hospital_admission': 'Ziekenhuisopnames',
    'Date_of_statistics': 'Datum',
  })
  # datum naar datetime
  df['Datum'] = pd.to_datetime(df.Datum.str.replace(' .*', '', regex=True))
  # sommeer op datum en voeg toe aan dagoverzicht
  addstats(df.groupby(['Datum']).agg({'Ziekenhuisopnames': 'sum'}))
  dagoverzicht['Ziekenhuisopnames week-1'] = dagoverzicht['Ziekenhuisopnames 7d'].shift(7).fillna(0)
  display(dagoverzicht.head())

# %% reproductiegetal en besmettelijkheid
@run
def cell():
  global dagoverzicht

  datasets = [
    ('COVID-19_reproductiegetal', 'Rt_avg', 'Reproductiegetal'),
    ('COVID-19_prevalentie', 'prev_avg', 'Besmettelijk'),
  ]
  for dataset, source, target in datasets:
    # laad de dataset
    df = RIVM.json(dataset).rename(columns={source: target})
    # Date naar datetime index voor de merge
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # merge alleen de target kolom
    df = df[[target]]
    # voeg to aan dagoverzicht
    dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
    # vul lege waarden met 0
    dagoverzicht[target] = dagoverzicht[target].fillna(0)
  # per 100k factor
  dagoverzicht['Besmettelijk per 100.000'] = (dagoverzicht['Besmettelijk']  * bevolking['per 100k']).round(0)
  display(dagoverzicht)

# %% uitgevoerde testen
@run
def cell():
  df = RIVM.csv('COVID-19_uitgevoerde_testen').rename(columns={
    'Date_of_statistics': 'Datum',
    'Tested_with_result': 'GGD getest',
    'Tested_positive': 'GGD getest positief',
  })
  df['Datum'] = pd.to_datetime(df.Datum)
  df = df.groupby(['Datum']).agg({'GGD getest': 'sum', 'GGD getest positief': 'sum'})
  display(df)

  global dagoverzicht
  columns = dagoverzicht.columns

  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)

  dagoverzicht['GGD percentage positief'] = (dagoverzicht['GGD getest positief'] / dagoverzicht['GGD getest']).fillna(0)

  dagoverzicht['GGD getest (7 daags)'] = dagoverzicht['GGD getest'].rolling(7).sum().fillna(0)
  dagoverzicht['GGD getest positief (7 daags)'] = dagoverzicht['GGD getest positief'].rolling(7).sum().fillna(0)

  dagoverzicht['GGD percentage positief (7 daags)'] = (dagoverzicht['GGD getest positief (7 daags)'] / dagoverzicht['GGD getest (7 daags)']).fillna(0)

  dagoverzicht['GGD getest (cumulatief)'] = dagoverzicht['GGD getest'].cumsum()
  dagoverzicht['GGD getest positief (cumulatief)'] = dagoverzicht['GGD getest positief'].cumsum()

  dagoverzicht['GGD percentage positief (cumulatief)'] = (dagoverzicht['GGD getest positief (cumulatief)'] / dagoverzicht['GGD getest (cumulatief)']).fillna(0)

  # fill 0 *after* the calculations above to prevent summing 'GGD getest (7 daags) from the starting date of the 'GGD getest' series
  for col in dagoverzicht.columns:
    # only zero out new columns
    if col in columns: continue
    dagoverzicht[col] = dagoverzicht[col].fillna(0)

  display(dagoverzicht.head(10))

# %% LCPS
@run
def cell():
  # laad dataset
  df = LCPS.csv('covid-19-datafeed').rename(columns={  
    'IC_Bedden_COVID_Nederland': 'LCPS IC Bedden COVID',
    'IC_Bedden_COVID_Internationaal': 'LCPS IC Bedden COVID Internationaal',
    'IC_Bedden_Non_COVID_Nederland': 'LCPS IC Bedden Non COVID',
    'Kliniek_Bedden_Nederland': 'LCPS Kliniek Bedden COVID',
    'IC_Nieuwe_Opnames_COVID_Nederland': 'LCPS IC Nieuwe Opnames COVID',
    'Kliniek_Nieuwe_Opnames_COVID_Nederland': 'LCPS Kliniek Nieuwe Opnames COVID',
  })
  # datum naar datetime index voor merge
  df['Datum'] = pd.to_datetime(df['Datum'], format='%d-%m-%Y')
  df.set_index('Datum', inplace=True)

  # sommeer op datum
  df = df.groupby(['Datum']).agg({col: 'sum' for col in df.columns})

  # toename = waarde - vorige
  df['LCPS IC Bedden COVID (toename)'] = (df['LCPS IC Bedden COVID'] - df['LCPS IC Bedden COVID'].shift(1)).fillna(0)
  df['LCPS Kliniek Bedden COVID (toename)'] = (df['LCPS Kliniek Bedden COVID'] - df['LCPS Kliniek Bedden COVID'].shift(1)).fillna(0)

  global dagoverzicht
  # voeg toe aan dagoverzicht
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for col in df.columns:
    # vervang lege waarden door 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0).astype(int)

# %% corrections
@run
def cell():
  # laad corrections van mzelst
  df = GitHub.csv('mzelst/covid-19/contents/corrections/corrections_perday.csv')
  # date naar datetime index voor merge
  df['date'] = pd.to_datetime(df.date)
  df.set_index('date', inplace=True)
  columns =  {
    'net.infection': 'Positief getest (toename)',
    'net.hospitals': 'Ziekenhuisopnames (toename)',
    'net.deaths': 'Overleden (toename)'
  }
  # rename kolommen naar onze namen
  df = df.rename(columns=columns)[list(columns.values())]

  global dagoverzicht
  # voeg toe aan dagoverxicht
  dagoverzicht = dagoverzicht.merge(df, how='left', left_index=True, right_index=True)
  for col in columns.values():
    # set lege waarden op 0
    dagoverzicht[col] = dagoverzicht[col].fillna(0).astype(int)

# %% Personen en Key
@run
def cell():
  global dagoverzicht
  dagoverzicht['Personen'] = bevolking.BevolkingOpDeEersteVanDeMaand
  dagoverzicht['Key'] = dagoverzicht.index.strftime('%Y-%m-%d')
  display(dagoverzicht)

# %%
if knack:
  await knack.publish(dagoverzicht, 'Dagoverzicht', Cache)
