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
import pandas as pd

import asyncio
import aiohttp
import backoff
from slack_webhook import Slack
import os
from datetime import datetime, timedelta

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
class AsyncLimiter(aiolimiter.AsyncLimiter): # fills the bucket, forcing a backoff
  def pause(self):
    self._level = self.max_rate
    self._last_check = get_running_loop().time()

def on_backoff(details):
  ex_type, ex, ex_traceback = sys.exc_info()
  self = details['args'][0]

  # on 429, pause and note the backoff
  if ex.status == 429:
    self.limiter.pause()
    self.calls.backoff()
  else:
    self.calls.error(ex.status, ex.message)
    #print('looks like Knack croaked again:', {'status': ex.status, 'message': ex.message})

def on_giveup(details):
  ex_type, ex, ex_traceback = sys.exc_info()
  task = details['args'][1]
  print(task.action)
  print(task.data)

class Knack:
  class Calls:
    def __init__(self):
      self.actions = Munch.fromDict({k: {} for k in ['create', 'read', 'update', 'delete']})
      self.actions.backoff = 0
      self.errors = {}

    # if we got a backoff, the actual CUD action that triggered it didn't go through
    def backoff(self):
      self.actions.backoff += 1

    def hit(self, action, _id=None):
      # no retry for this action, so every attempt is an unique id. Use negative IDs for this because those are not used by the re-attempted calls
      if _id is None:
        _id = -len(self.actions[action])
      self.actions[action][_id] = True

    def error(self, status, message):
      msg = f'{status}: {message}'
      if msg not in self.errors:
        self.errors[msg] = 0
      self.errors[msg] += 1

    def __repr__(self):
      calls = []
      for k, n in self.actions.items():
        if type(n) != int:
          n = len(n)
        calls.append(f'{k}: {n}')
      calls = ', '.join(calls) + "\n"
      if len(self.errors) > 0:
        calls += 'errors:\n'
        for msg, n in self.errors.items():
          msg = msg.replace("\n", " ")
          calls += f"  {msg}: {n}\n"
      return calls.strip()

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

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=60, on_backoff=on_backoff, on_giveup=on_giveup)
  async def execute(self, task):
    self.calls.hit(task.action, task.id)
    async with self.limiter:
      if task.action == 'create':
        async with self.session.post(f'https://api.knack.com/v1/objects/{task.object_key}/records', json=task.data, headers=self.headers, raise_for_status=True) as response:
          return await response.read()
      elif task.action == 'update':
        async with self.session.put(f'https://api.knack.com/v1/objects/{task.object_key}/records/{task.record_id}', json=task.data, headers=self.headers, raise_for_status=True) as response:
          return await response.read()
      elif task.action == 'delete':
        async with self.session.delete(f'https://api.knack.com/v1/objects/{task.object_key}/records/{task.record_id}', headers=self.headers, raise_for_status=True) as response:
          return await response.read()
      else:
        raise ValueError(f'Unexpected task action {json.dumps(task.action)}')

  def getall(self, object_key):
    url = f'https://api.knack.com/v1/objects/{object_key}/records?rows_per_page=1000'
    records = []
    page = 1
    while True:
      res = requests.get(url=f'{url}&page={page}', headers=self.headers)
      res.raise_for_status()
      self.calls.hit('read')

      if res.status_code >= 200 and res.status_code < 300:
        res = res.json()
        records += res['records']
        if res['current_page'] != res['total_pages'] and res['total_pages'] != 0: # what the actual...
          page += 1
        else:
          break
      else:
        msg = None
        try:
          msg = res.json()
          if type(msg) == dict and 'errors' in msg:
            msg = json.dumps(msg['errors'])
        except JSONDecodeError:
          pass
        if msg is None:
          msg = res.text

        self.calls.error(res.status_code, msg)

    return self.munch(records)

  def munch(self, records):
    if type(records) == list:
      return [ Munch.fromDict(rec) for rec in records ]
    elif type(records) == dict:
      return Munch.fromDict(records)
    else:
      raise ValueError(f'Unexpected type {str(type(records))}')

  def safe_dict(self, kv):
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

    self.mapping = self.safe_dict([ (field.name, field.key) for field in obj.fields ])
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
          self.connection_field_map[field.relationship.object] = { d.get(domain + '_raw', d[domain]): d['id'] for d in self.getall(field.relationship.object) }
        connections[field.name] = self.connection_field_map[field.relationship.object]

    data = self.munch(df.replace(connections).rename(columns=self.mapping).to_dict('records'))
    unmapped = [col for col in data[0].keys() if not col.startswith('field_')]
    assert len(unmapped) == 0, unmapped

    hashing = 'Hash' in self.mapping
    force = not hashing
    if hashing:
      for rec in data:
        assert self.mapping.Hash not in rec
        rec[self.mapping.Hash] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode('utf-8')).hexdigest()

    create = self.safe_dict([ (rec[key.field], rec) for rec in data ])
    update = []
    delete = []
    for ist in self.getall(obj.key):
      if soll:= create.get(ist[key.field]):
        if force or ist[self.mapping.Hash] != soll[self.mapping.Hash]:
          update.append((ist.id, soll))
        del create[soll[key.field]]
      else:
        delete.append(ist.id)

    # because the shoddy Knack platform cannot get to more than 2-3 calls per second without parallellism, but if you *do* use parallellism
    # to any significant extent you get immediate backoff errors. And lots of 'em
    self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=1)
    async with aiohttp.ClientSession() as session:
      self.session = session
      tasks = [
        Munch(action='delete', object_key=obj.key, record_id=ist) for ist in delete
      ] + [
        Munch(action='update', object_key=obj.key, record_id=ist, data=soll) for ist, soll in update
      ] + [
        Munch(action='create', object_key=obj.key, data=soll) for soll in create.values()
      ]
      for i, task in enumerate(tasks):
        task.id = i + 1 # skip 0 to avoid confusion between -0 and 0 in the call tracking
      tasks = [asyncio.create_task(self.execute(task)) for task in tasks]
      if len(tasks) == 0:
        print('nothing to do')
        self.slack('nothing to do', emoji=':sleeping:')
      else:
        responses = [await req for req in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        self.slack(f'API calls: {self.calls}', emoji=(':white_check_mark:' if hashing else ':game_die:'))
      print('\nrate limit:', rate_limit, '\nAPI calls:', self.calls)

    return len(tasks)

  async def timestamps(self, notebook, timestamps):
    msg = '*update timestamps*'
    for provider, ts in timestamps.items():
      msg += f"\n• *{provider}*: {ts}"
    self.slack(msg, emoji=':clock1:')
    await self.update(objectName='LaatsteUpdate', df=pd.DataFrame([{'Key': 1, **{ f'Timestamp {notebook} {provider}': ts for provider, ts in timestamps.items() }}]))

  def slack(self, msg, emoji=''):
    if 'SLACK_WEBHOOK' not in os.environ: return

    prefix = ''

    if 'GITHUB_RUN_ID' in os.environ:
      prefix += '<'
      prefix += os.environ['GITHUB_SERVER_URL'] + '/' + os.environ['GITHUB_REPOSITORY'] + '/actions/runs/' + os.environ['GITHUB_RUN_ID']
      prefix += '|'
      prefix += os.environ['GITHUB_RUN_NUMBER']
      prefix += '> '

    if nb := os.environ.get('NOTEBOOK'):
      prefix += f'*{nb}* '

    prefix += emoji + ' '

    prefix += (datetime.now() + timedelta(hours=1)).strftime(f'%Y-%m-%d %H:%M ')

    prefix = prefix.strip() + '\n'
    Slack(url=os.environ['SLACK_WEBHOOK']).post(text=prefix + msg)
