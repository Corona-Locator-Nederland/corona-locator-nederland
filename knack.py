from urllib.request import urlopen
from munch import Munch
from jsonpath import JSONPath
import json
from retry import retry
from requests import HTTPError
import time
import requests
from json import JSONDecodeError

def in_notebook():
  from IPython import get_ipython
  return get_ipython() is not None
if in_notebook():
  from tqdm.notebook import tqdm as progress
else:
  from tqdm import tqdm as progress

class Knack:
  def __init__(self, app_id, api_key):
    self.app_id = app_id
    self.api_key = api_key
    self.headers = {
      'X-Knack-Application-Id': app_id,
      'X-Knack-REST-API-KEY': api_key,
    }
    with urlopen(f'https://loader.knack.com/v1/applications/{app_id}') as response:
      self.metadata = json.load(response)

  def find(self, path):
    found = JSONPath(path).parse(self.metadata)
    if len(found) == 1:
      return self.munch(found[0])
    raise ValueError(f'{path} yields {len(found)} results, expected 1')

  def _check(self, res):
    if res.status_code >= 200 and res.status_code < 300: return res
    try:
      msg = res.json()
      if type(msg) == dict and 'errors' in msg: raise ValueError(json.dumps(msg['errors']))
    except JSONDecodeError:
      pass
    res.raise_for_status()

  @retry(HTTPError, delay=1, tries=2) # absolutely ridiculous
  def _create(self, object_key, data):
    #curl = ' '.join(['-H ' + shlex.quote(f'{k}: {v}') for k, v in self.headers.items()])
    #curl = f'curl -X POST https://api.knack.com/v1/objects/{object_key}/records {curl} --data-binary {shlex.quote(json.dumps(data))}'
    #print(curl)
    #curl = subprocess.run(curl, shell=True)
    #if curl.returncode != 0: raise ValueError(str(curl.returncode))
    #return

    url = f'https://api.knack.com/v1/objects/{object_key}/records'
    self._check(requests.post(url=url, json=data, headers=self.headers))

  @retry(HTTPError, delay=1, tries=2) # absolutely ridiculous
  def _update(self, object_key, record_id, data):
    url = f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}'
    self._check(requests.put(url=url, json=data, headers=self.headers))

  @retry(HTTPError, delay=1, tries=2) # absolutely ridiculous
  def _delete(self, object_key, record_id):
    url = f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}'
    self._check(requests.delete(url=url, headers=self.headers))

  def _getall(self, object_key):
    url = f'https://api.knack.com/v1/objects/{object_key}/records?rows_per_page=1000'
    records = []
    page = 1
    while True:
      res = self._check(requests.get(url=f'{url}&page={page}', headers=self.headers)).json()
      records += res['records']
      if res['current_page'] != res['total_pages'] and res['total_pages'] != 0: # what the actual...
        page += 1
      else:
        break
    # echt mensen dit geloof je niet.
    for rec in records:
      for k, v in list(rec.items()):
        if not k.endswith('_raw'):
          continue
        if type(v) == dict and 'date' in v:
          v = v['iso_timestamp'].replace('T00:00:00.000Z', '')
        rec[k.replace('_raw', '')] = v
        rec.pop(k)

    return self.munch(records)

  def munch(self, records):
    if type(records) == list:
      return [ Munch.fromDict(rec) for rec in records ]
    elif type(records) == dict:
      return Munch.fromDict(records)
    else:
      raise ValueError(f'Unexpected type {str(type(records))}')

  def update(self, sceneName=None, viewName=None, objectName=None, df=None, verbose=False):
    assert df is not None, 'df parameter is required'

    assert (sceneName is not None and viewName is not None) != (objectName is not None), 'Specify either viewName and sceneName, or objectName'

    if objectName:
      obj = self.find(f'$.application.objects[?(@.name=={json.dumps(objectName)})]')
    else:
      view = self.find(f'$.application.scenes[?(@.name=={json.dumps(sceneName)})].views[?(@.name=={json.dumps(viewName)})]')
      source = view.source.object
      obj = self.find(f'$.application.objects[?(@.key=="{source}")]')
    mapping = { field.name: field.key for field in obj.fields }

    soll = { rec[obj.identifier]: Munch(record=rec, key=rec[obj.identifier]) for rec in df.rename(columns=mapping).to_dict('records') }
    ist = self._getall(obj.key)
    # fetch these separately as the key does not need to be unique?
    delete = [rec.id for rec in ist if rec.get(obj.identifier) not in soll]
    ist  = { rec[obj.identifier]: Munch(record=rec, key=rec[obj.identifier]) for rec in ist if not rec.id in delete }
   
    for record_id in progress(delete, desc='deleting obsolete records'):
      self._delete(object_key=obj.key, record_id=record_id)

    for data in progress([rec.record for rec in soll.values() if rec.key not in ist], desc='creating records'):
      self._create(object_key=obj.key, data=data)

    for record_id, data in progress([(ist[rec.key].record.id, rec.record) for rec in soll.values() if rec.key in ist and {**rec.record, 'id': None} != {**ist[rec.key].record, 'id': None}], desc='updating records'):
      self._update(object_key=obj.key, record_id=record_id, data=data)
