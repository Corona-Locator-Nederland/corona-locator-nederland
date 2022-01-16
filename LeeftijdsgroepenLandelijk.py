# %%
from IPython import get_ipython
from IPython.core.display import display
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('run', 'setup')

# %% Leeftijdsgroepen: download RIVM casus-data
# Leeftijdsgroepen: download RIVM casus-data
@run
def cell():
  global rivm
  rivm = RIVM.csv('COVID-19_casus_landelijk')
  display(rivm.head())

# %% Leeftijdsgroepen: download CBS bevolkingscijfers, uitgesplitst op de leeftijdscategorien zoals bij RIVM
# Leeftijdsgroepen: download CBS bevolkingscijfers, uitgesplitst op de leeftijdscategorien zoals bij RIVM
@run
def cell():
  global bevolking
  # probeer de laatste leeftijdsgroepen op te halen bij het CBS, ververs de fallback-file 
  try:
    bevolking = CBS.bevolking(leeftijdsgroepen=True)
    bevolking.to_csv("cbs/leeftijdsgroepen_cbs.csv")
  # als het niet is gelukt de CBS file op te halen, gebruik de fallback-file en pas de index klom aan
  except Exception as e:
    print(e)
    bevolking = pd.read_csv("cbs/leeftijdsgroepen_cbs.csv")
    bevolking.set_index('Range', inplace=True)
  display(bevolking.head())

# %% Leeftijdsgroepen: bereken per cohort besmettingen / opnamen / sterfte (incl. kleurcode), in aantal en per 100k. Vervang <50 en Unknown door Onbekend
# Leeftijdsgroepen: bereken per cohort besmettingen / opnamen / sterfte (incl. kleurcode), in aantal en per 100k. Vervang <50 en Unknown door Onbekend
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

  # bij /100k is het Totaal-veld geen optelsom maar wordt berekend obv de gehele bevolking
  totale_bevolking_per_cohort = bevolking.to_dict()['BevolkingOpDeEersteVanDeMaand']
  totale_bevolking = sum(totale_bevolking_per_cohort.values())
  keys = [ key for key in tabel["Key"]]
  totals = [ x for x in tabel["Totaal"]]
  for k in range(0, len(keys)):
    key = keys[k]
    if '100k' in key:
      abskey = key[0] + key[5:]
      kk = keys.index(abskey)
      if isinstance(totals[k], float):
        correctedtotal = totals[kk] * (100_000 / totale_bevolking)
        # print([k, totals[k], totals[kk], correctedtotal])
        totals[k] = correctedtotal
  tabel["Totaal"] = totals

  display(tabel.head())

# %% publish
if knack:
  await knack.publish(tabel.fillna(0).assign(Datum=tabel.Datum.dt.strftime('%Y-%m-%d')), 'Leeftijdsgroep', Cache)
