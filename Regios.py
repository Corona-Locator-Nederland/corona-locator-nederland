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
    df = df.reset_index(level=0)
    sh.values_clear("'Regios'!A1:ZZ10000")
    sh.values_clear("'Regios'!A1:ZZ10000")
    ws.update([df.columns.values.tolist()] + df.values.tolist())
  else:
    display(df)


# %%
@run('regios: download gemeenten en hun codes')
def cell():
  global gemeenten
  global bevolking
  gemeenten = pd.read_csv('gemeenten.csv')

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
  display(gemeenten)
  
# %%
@run('regios: RIVM cijfers ophalen')
def cell():
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


# %%
@run('regios: absolute aantallen per gemeente')
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
    replace = [
        'Positief getest',
        'Positief getest per 100.000',
        'Positief getest 1d/100k',
        'Positief getest per km2',
        'Overleden',
        'Overleden per 100.000',
        'Ziekenhuisopname',
        'Positief getest (toename)',
        'Overleden (toename)',
        'Ziekenhuisopname (toename)',
        'Positief getest percentage',
    ]
    gemeenten = (gemeenten[[col for col in gemeenten.columns if col not in replace]]
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
    
    days = (pd.to_datetime(aantallen_gemeenten['Date_of_publication']).max() - pd.to_datetime(aantallen_gemeenten['Date_of_publication']).min()) / np.timedelta64(1, 'D')
    gemeenten['Positief getest 1d/100k'] = gemeenten['Positief getest per 100.000'] / days

    gemeenten['Positief getest percentage'] = (gemeenten['Positief getest'] / gemeenten['Personen']).replace(np.inf, 0)
    gemeenten['Positief getest per km2'] = (gemeenten['Positief getest'] / gemeenten['Opp land km2']).replace(np.inf, 0)

    display(gemeenten.head())
    publish(gemeenten.fillna(0))


# %%



