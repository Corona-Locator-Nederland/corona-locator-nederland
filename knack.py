from urllib.request import urlopen
from munch import Munch
from jsonpath import JSONPath
import json
from requests import HTTPError
import time
import requests
from json import JSONDecodeError
import hashlib

import asyncio
import aiohttp
import backoff

def in_notebook():
  from IPython import get_ipython
  return get_ipython() is not None
if in_notebook():
  #import tqdm.notebook as tqdm
  import tqdm.asyncio as tqdm
else:
  #import tqdm
  import tqdm.asyncio as tqdm

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
      with open('metadata.json', 'w') as f:
        json.dump(self.metadata, f, indent='  ')

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

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=8)
  async def _create(self, object_key, data):
    self.calls += 1
    async with self.semaphore:
      async with self.session.post(f'https://api.knack.com/v1/objects/{object_key}/records', json=data, headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=8)
  async def _update(self, object_key, record_id, data):
    self.calls += 1
    async with self.semaphore:
      async with self.session.put(f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}', json=data, headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=8)
  async def _delete(self, object_key, record_id):
    self.calls += 1
    async with self.semaphore:
      async with self.session.delete(f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}', headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  def _getall(self, object_key):
    url = f'https://api.knack.com/v1/objects/{object_key}/records?rows_per_page=1000'
    records = []
    page = 1
    while True:
      res = self._check(requests.get(url=f'{url}&page={page}', headers=self.headers)).json()
      self.calls += 1
      records += res['records']
      if res['current_page'] != res['total_pages'] and res['total_pages'] != 0: # what the actual...
        page += 1
      else:
        break
    # echt mensen dit geloof je niet.
    #for rec in records:
    #  for k, v in list(rec.items()):
    #    if not k.endswith('_raw'):
    #      continue
    #    if type(v) == dict and 'date' in v:
    #      v = v['iso_timestamp'].replace('T00:00:00.000Z', '')
    #    rec[k.replace('_raw', '')] = v
    #    rec.pop(k)

    return self.munch(records)

  def munch(self, records):
    if type(records) == list:
      return [ Munch.fromDict(rec) for rec in records ]
    elif type(records) == dict:
      return Munch.fromDict(records)
    else:
      raise ValueError(f'Unexpected type {str(type(records))}')

  def _dict(self, kv):
    m = Munch()
    for k, v in kv:
      if k in m:
        raise KeyError(f'duplicate key {json.dumps(k)}')
      m[k] = v
    return m

  async def update(self, sceneName=None, viewName=None, objectName=None, df=None, force=False):
    self.calls = 0
    assert df is not None, 'df parameter is required'

    assert (sceneName is not None and viewName is not None) != (objectName is not None), 'Specify either viewName and sceneName, or objectName'

    if objectName:
      obj = self.find(f'$.application.objects[?(@.name=={json.dumps(objectName)})]')
      with open(objectName + '.json', 'w') as f:
        json.dump(obj, f, indent='  ')
    else:
      view = self.find(f'$.application.scenes[?(@.name=={json.dumps(sceneName)})].views[?(@.name=={json.dumps(viewName)})]')
      source = view.source.object
      obj = self.find(f'$.application.objects[?(@.key=="{source}")]')

    mapping = self._dict([ (field.name, field.key) for field in obj.fields ])
    key = [field.name for field in obj.fields if field.get('unique')]
    assert len(key) == 1

    key = Munch(name=key[0])
    key.field = mapping[key.name]

    assert key.name in df, f'{json.dumps(key.name)} not present in {str(df.columns)}'
    assert df.rename(columns=mapping).loc[:, key.field].is_unique, f'{json.dumps(key.name)}/{json.dumps(key.field)} is not unique in the dataset'

    connections = {}
    for field in obj.fields:
      if field.type == 'connection':
        assert 'relationship' in field and field.relationship.get('has') == 'one' and field.relationship.get('belongs_to') == 'many'
        domain = [ f for f in self.find(f'$.application.objects[?(@.key=={json.dumps(field.relationship.object)})].fields') if f.get('unique') ]
        assert len(domain) == 1
        domain = domain[0]['key']
        connections[field.name] = { d.get(domain + '_raw', d[domain]): d['id'] for d in self._getall(field.relationship.object) }

    data = self.munch(df.replace(connections).rename(columns=mapping).to_dict('records'))
    if 'Hash' in mapping:
      for rec in data:
        assert mapping.Hash not in rec
        rec[mapping.Hash] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode('utf-8')).hexdigest()

      create = self._dict([ (rec[key.field], rec) for rec in data ])
      update = []
      delete = []
      for ist in self._getall(obj.key):
        if soll:= create.get(ist[key.field]):
          ist[mapping.Hash]
          soll[mapping.Hash]
          if force or ist[mapping.Hash] != soll[mapping.Hash]:
            update.append((ist.id, soll))
          del create[soll[key.field]]
        else:
          delete.append(rec.id)
    else:
      delete = [rec.id for rec in self._getall(obj.key)]

    self.semaphore = asyncio.Semaphore(9)
    async with aiohttp.ClientSession() as session:
      self.session = session
      tasks = [
        asyncio.create_task(self._delete(object_key=obj.key, record_id=ist)) for ist in delete
      ] + [
        asyncio.create_task(self._update(object_key=obj.key, record_id=ist, data=soll)) for ist, soll in update
      ] + [
        asyncio.create_task(self._create(object_key=obj.key, data=soll)) for soll in create.values()
      ]
      if len(tasks) == 0:
        print('nothing to do')
      else:
        responses = [await req for req in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
      print(self.calls, 'API calls')
    return len(tasks)
