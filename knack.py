from urllib.request import urlopen
from munch import Munch
from jsonpath import JSONPath
import json
from requests import HTTPError
import time
import requests
from json import JSONDecodeError
import hashlib
import logging
import sys, os

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

import aiolimiter
from aiolimiter.compat import get_running_loop
class AsyncLimiter(aiolimiter.AsyncLimiter):
  def pause(self):
    self._level = self.max_rate
    self._last_check = get_running_loop().time()

def on_backoff(details):
  ex_type, ex, ex_traceback = sys.exc_info()
  if ex.status == 429:
    details['args'][0].limiter.pause()
    details['args'][0].calls.backoff += 1
  else:
    details['args'][0].calls.error(ex.status, ex.message)
    #print('looks like Knack croaked again:', {'status': ex.status, 'message': ex.message})

class Knack:
  class Calls:
    def __init__(self):
      self.create = 0
      self.read = 0
      self.update = 0
      self.delete = 0
      self.backoff = 0
      self.errors = {}
    def error(self, status, message):
      if status not in self.errors:
        self.errors[status] = [message]
      else:
        self.errors[status].append(message)
    def __repr__(self):
      r = [', '.join([ f'{k}={v}' for k, v in self.__dict__.items() if k != 'errors' ])]
      for status, messages in self.errors.items():
        r.append(f'{status}:')
        for m in messages:
          r.append('  ' + m.replace('\n', ' '))
      return '\n'.join(r)

  def __init__(self, app_id, api_key):
    self.app_id = app_id
    self.api_key = api_key
    self.headers = {
      'X-Knack-Application-Id': app_id,
      'X-Knack-REST-API-KEY': api_key,
    }
    self.connection_field_map = {}
    with urlopen(f'https://loader.knack.com/v1/applications/{app_id}') as response:
      self.metadata = json.load(response)
      os.makedirs('metadata', exist_ok = True)
      with open('metadata/metadata.json', 'w') as f:
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

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60, on_backoff=on_backoff)
  async def _create(self, object_key, data):
    self.calls.create += 1
    async with self.limiter:
      async with self.session.post(f'https://api.knack.com/v1/objects/{object_key}/records', json=data, headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60, on_backoff=on_backoff)
  async def _update(self, object_key, record_id, data):
    self.calls.update += 1
    async with self.limiter:
      async with self.session.put(f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}', json=data, headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60, on_backoff=on_backoff)
  async def _delete(self, object_key, record_id):
    self.calls.delete += 1
    async with self.limiter:
      async with self.session.delete(f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}', headers=self.headers, raise_for_status=True) as response:
        return await response.read()

  def _getall(self, object_key):
    url = f'https://api.knack.com/v1/objects/{object_key}/records?rows_per_page=1000'
    records = []
    page = 1
    while True:
      res = self._check(requests.get(url=f'{url}&page={page}', headers=self.headers)).json()
      self.calls.read += 1
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

  async def update(self, sceneName=None, viewName=None, objectName=None, df=None, force=False, rate_limit=7):
    self.calls = self.Calls()
    assert df is not None, 'df parameter is required'

    assert (sceneName is not None and viewName is not None) != (objectName is not None), 'Specify either viewName and sceneName, or objectName'

    if objectName:
      obj = self.find(f'$.application.objects[?(@.name=={json.dumps(objectName)})]')
      os.makedirs('metadata', exist_ok = True)
      with open(os.path.join('metadata', objectName + '.json'), 'w') as f:
        json.dump(obj, f, indent='  ')
    else:
      view = self.find(f'$.application.scenes[?(@.name=={json.dumps(sceneName)})].views[?(@.name=={json.dumps(viewName)})]')
      source = view.source.object
      obj = self.find(f'$.application.objects[?(@.key=="{source}")]')

    self.mapping = self._dict([ (field.name, field.key) for field in obj.fields ])
    key = [field.name for field in obj.fields if field.get('unique')]
    assert len(key) == 1

    key = Munch(name=key[0])
    key.field = self.mapping[key.name]

    assert key.name in df, f'{json.dumps(key.name)} not present in {str(df.columns)}'
    assert df.rename(columns=self.mapping).loc[:, key.field].is_unique, f'{json.dumps(key.name)}/{json.dumps(key.field)} is not unique in the dataset'

    connections = {}
    for field in obj.fields:
      if field.type == 'connection':
        assert 'relationship' in field and field.relationship.get('has') == 'one' and field.relationship.get('belongs_to') == 'many'
        if field.relationship.object not in self.connection_field_map:
          domain = [ f for f in self.find(f'$.application.objects[?(@.key=={json.dumps(field.relationship.object)})].fields') if f.get('unique') ]
          assert len(domain) == 1
          domain = domain[0]['key']
          self.connection_field_map[field.relationship.object] = { d.get(domain + '_raw', d[domain]): d['id'] for d in self._getall(field.relationship.object) }
        connections[field.name] = self.connection_field_map[field.relationship.object]

    data = self.munch(df.replace(connections).rename(columns=self.mapping).to_dict('records'))
    if 'Hash' in self.mapping:
      for rec in data:
        assert self.mapping.Hash not in rec
        rec[self.mapping.Hash] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode('utf-8')).hexdigest()

      create = self._dict([ (rec[key.field], rec) for rec in data ])
      update = []
      delete = []
      for ist in self._getall(obj.key):
        if soll:= create.get(ist[key.field]):
          ist[self.mapping.Hash]
          soll[self.mapping.Hash]
          if force or ist[self.mapping.Hash] != soll[self.mapping.Hash]:
            update.append((ist.id, soll))
          del create[soll[key.field]]
        else:
          delete.append(rec.id)
    else:
      delete = [rec.id for rec in self._getall(obj.key)]

    # because the shoddy Knack platform cannot get to more than 2-3 calls per second without parallellism, but if you *do* use parallellism
    # to any significant extent you get immediate backoff errors. And lots of 'em
    self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=1)
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
      print('\nrate limit:', rate_limit, 'API calls:', self.calls)
    return len(tasks)
