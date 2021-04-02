from urllib.request import urlopen
from munch import Munch
from jsonpath import JSONPath
import json
from requests import HTTPError
import time
import requests
from json import JSONDecodeError
import logging
import sys, os
import pandas as pd
import base64
import zlib
import hashlib

import asyncio
import aiohttp
import backoff
from slack_webhook import Slack
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import binascii
import copy
import getpass

def in_notebook():
  from IPython import get_ipython
  return get_ipython() is not None
if in_notebook():
  #import tqdm.notebook as tqdm
  import tqdm.asyncio as tqdm
else:
  #import tqdm
  import tqdm.asyncio as tqdm

class File(object):
  def __init__(self, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok = True)
    self.f = open(file_name, 'w')
  def __enter__(self):
    return self.f
  def __exit__(self, type, value, traceback):
    self.f.close()

import aiolimiter
from aiolimiter.compat import get_running_loop
class AsyncLimiter(aiolimiter.AsyncLimiter): # fills the bucket, forcing a backoff
  def pause(self):
    self._level = self.max_rate
    self._last_check = get_running_loop().time()

def sort(df):
  return df[sorted(df.columns)]

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
  print(task.action, flush=True)
  print(task.data, flush=True)

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
    #self.fill = {}
    self.all = {}
    self.connection_field_map = {}
    with urlopen(f'https://loader.knack.com/v1/applications/{app_id}') as response:
      self.metadata = json.load(response)
      with File('metadata/app.json') as f:
        json.dump(self.metadata, f, indent='  ')

  def hash(self, record):
    #rec[self.mapping.Hash] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode('utf-8')).hexdigest()
    return base64.b64encode(zlib.compress(json.dumps(record, sort_keys=True).encode('utf-8'), 9)).decode('utf-8')
  def unhash(self, record):
    return json.loads(zlib.decompress(base64.b64decode(record)))

  def find(self, paths):
    found = []
    if type(paths) != list:
      paths = [ paths ]
    for path in paths:
      found = found + JSONPath(path).parse(self.metadata)
    if len(found) == 1:
      return self.munch(found[0])
    raise ValueError(f'{path} yields {len(found)} results, expected 1')

  @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_time=180, on_backoff=on_backoff, on_giveup=on_giveup)
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
    if not object_key in self.all:
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

      obj = self.object_metadata(object_key)

      with File(os.path.join('artifacts', 'fetched', obj.meta.name + '.json')) as f:
        json.dump(records, f, indent='  ')

      if hashcol := obj.mapping.get('Hash'):
        try:
          restored = [{**self.unhash(record[hashcol]), 'id': record['id'], hashcol: record[hashcol]} for record in records]
          if len(records) > 0:
            assert all(key.startswith('field_') or key == 'id' for key in restored[0].keys())
          print('restored', obj.meta.name, 'from hash', flush=True)
          #self.fill[obj.meta.name] = True
          records = restored
          with File(os.path.join('artifacts', 'restored', obj.meta.name + '.json')) as f:
            json.dump(records, f, indent='  ')
        except (binascii.Error, zlib.error, AssertionError):
          print('failed to restore', obj.meta.name, 'from hash', flush=True)
          #self.fill[obj.meta.name] = False

      self.all[object_key] = records

    return self.munch(copy.deepcopy(self.all[object_key]))

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

  def object_metadata(self, object_name):
    paths = [f'$.application.objects[?(@.{field}=={json.dumps(object_name)})]' for field in ['name', 'key']]
    meta = self.find(paths)
    mapping = self.safe_dict([ (field.name, field.key) for field in meta.fields ])

    with File(os.path.join('metadata', 'mapping', meta.name + '.json')) as f:
      json.dump(mapping, f, indent='  ')

    return Munch(meta=meta, mapping=mapping)

  async def update(self, object_name, df, force=False, rate_limit=7, slack=Munch(msg='',emoji=None)):
    self.calls = self.Calls()
    assert df is not None, 'df parameter is required'
    assert 'Hash' not in df.columns

    obj = self.object_metadata(object_name)

    with File(os.path.join('metadata', obj.meta.name + '.json')) as f:
      json.dump(obj.meta, f, indent='  ')

    key = [field.name for field in obj.meta.fields if field.get('unique')]
    assert len(key) <= 1, len(key)
    if len(key) == 0:
      key = None
    else:
      key = Munch(name=key[0])
      key.field = obj.mapping[key.name]

      assert key.name in df, f'{json.dumps(key.name)} not present in {str(df.columns)}'
      assert df.rename(columns=obj.mapping).loc[:, key.field].is_unique, f'{json.dumps(key.name)}/{json.dumps(key.field)} is not unique in the dataset'

    connections = {}
    for field in obj.meta.fields:
      if field.type == 'connection':
        assert 'relationship' in field and field.relationship.get('has') == 'one' and field.relationship.get('belongs_to') == 'many'
        if field.relationship.object not in self.connection_field_map:
          domain = [ f for f in self.find(f'$.application.objects[?(@.key=={json.dumps(field.relationship.object)})].fields') if f.get('unique') ]
          assert len(domain) == 1
          domain = domain[0]['key']
          self.connection_field_map[field.relationship.object] = { d.get(domain + '_raw', d[domain]): d['id'] for d in self.getall(field.relationship.object) }
        connections[field.name] = self.connection_field_map[field.relationship.object]

    data = self.munch(df.replace(connections).rename(columns=obj.mapping).to_dict('records'))
    unmapped = [col for col in data[0].keys() if not col.startswith('field_')]
    assert len(unmapped) == 0, unmapped

    hashing = 'Hash' in obj.mapping
    force = force or not hashing

    update = []
    delete = []
    if not key:
      create = { id(rec): rec for rec in data }
    else:
      create = self.safe_dict([ (rec[key.field], rec) for rec in data ])
      for ist in self.getall(obj.meta.key):
        if soll:= create.pop(ist[key.field], None):
          if hashing:
            assert obj.mapping.Hash not in soll, (soll.keys(), obj.mapping)
            #if self.fill.get(obj.meta.name):
            #  soll = {**{k: v for k, v in ist.items() if k not in ['id', obj.mapping.Hash]}, **soll}
            soll[obj.mapping.Hash] = self.hash(soll)
          if force or ist[obj.mapping.Hash] != soll[obj.mapping.Hash]:
            update.append((ist.id, soll))
        else:
          delete.append(ist.id)

    if hashing:
      for soll in create.values():
        soll[obj.mapping.Hash] = self.hash(soll)

    tasks = len(create) + len(update) + len(delete)

    # mangle data for ridiculous knack upload format
    df = sort(df).copy(deep=True)
    print(df.dtypes)
    if hashing:
      df['Hash'] = [self.hash(rec) for rec in self.munch(df.replace(connections).rename(columns=obj.mapping).to_dict('records'))]
    artifact = os.path.join('artifacts', 'bulk', f'{obj.meta.name}-mangle-for-knack.csv')
    for col, coltype in zip(df.columns, df.dtypes):
      if coltype in (int, np.int64, np.float64, float):
        df[col] = df[col].astype(str).str.replace(".", ",", regex=False).fillna('')
      elif coltype == object:
        df[col] = df[col].fillna('')
      else:
        raise ValueError(str(coltype))
    with File(artifact) as f:
      df.to_csv(f, index=False)

    if tasks > 2000: # Knack can't deal with even miniscule amounts of data
      print('Not executing', tasks, obj.meta.name, 'to spare API quota. Please upload', artifact, flush=True)
      self.slack(slack.msg + f"Not executing {tasks} {obj.meta.name} actions to spare API quota. Please upload {artifact} to {obj.meta.name}", obj.meta.name, emoji=':no_entry:')
      return False

    # because the shoddy Knack platform cannot get to more than 2-3 calls per second without parallellism, but if you *do* use parallellism
    # to any significant extent you get immediate backoff errors. And lots of 'em.
    self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=1)

    with File(os.path.join('artifacts', 'api', os.environ.get('NOTEBOOK', ''), f"{os.environ.get('GITHUB_RUN_NUMBER', getpass.getuser())}-{obj.meta.name}-{datetime.now().isoformat().replace('T', '@').replace(':', '-')}.json")) as f:
      json.dump({ 'delete': delete, 'update': update, 'create': list(create.values()) }, f)
    async with aiohttp.ClientSession() as session:
      self.session = session
      tasks = [
        Munch(action='delete', object_key=obj.meta.key, record_id=ist) for ist in delete
      ] + [
        Munch(action='update', object_key=obj.meta.key, record_id=ist, data=soll) for ist, soll in update
      ] + [
        Munch(action='create', object_key=obj.meta.key, data=soll) for soll in create.values()
      ]
      for i, task in enumerate(tasks):
        task.id = i + 1 # skip 0 to avoid confusion between -0 and 0 in the call tracking

      tasks = [asyncio.create_task(self.execute(task)) for task in tasks]
      if slack.msg.strip() != '':
        slack.msg = slack.msg.strip() + '\n'
      if len(tasks) == 0:
        print(f'nothing to do for {obj.meta.name}', flush=True)
        self.slack(slack.msg + f'nothing to do for {obj.meta.name}', obj.meta.name, emoji=':sleeping:')
      else:
        responses = [await req for req in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        self.slack(slack.msg + f'{obj.meta.name} API calls: {self.calls}', obj.meta.name, emoji=slack.emoji or ':white_check_mark:')
      print('\nrate limit:', rate_limit, f'\n{obj.meta.name} API calls:', self.calls, flush=True)
    return len(tasks)

  async def updating(self, object_name, updating):
    df = [{'Key': 1, f'Updating {object_name}': str(updating).lower() }]
    print('updating:', df)
    df = pd.DataFrame(df)
    # https://www.webfx.com/tools/emoji-cheat-sheet/
    await self.update(object_name='LaatsteUpdate', df=df, slack=Munch(msg=('Updating ' if updating else 'Updated ') + object_name, emoji=':clapper:' if updating else ' :checkered_flag:'))
    
  async def timestamps(self, object_name, timestamps):
    print([{'Key': 1, **{ f'Timestamp {object_name} {provider}': ts for provider, ts in timestamps.items() }}], flush=True)
    msg = ''
    for provider, ts in timestamps.items():
      msg += f"• *{provider}*: {ts}\n"

    df = [{'Key': 1, **{ f'Timestamp {object_name} {provider}': ts for provider, ts in timestamps.items() }}]
    print('timestamps:', df)
    df = pd.DataFrame(df)
    await self.update(object_name='LaatsteUpdate', df=df, slack=Munch(msg=msg, emoji=':clock1:'))

    batch = os.environ.get('GITHUB_RUN_NUMBER', f'{getpass.getuser()} @ {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    df = [ {'BatchName': batch, 'ObjectName': object_name, 'Source': provider, 'Timestamp': ts } for provider, ts in timestamps.items() ]
    df = pd.DataFrame(df)
    await self.update(object_name='UpdateDetails', df=df, slack=Munch(msg=msg, emoji=':clock1:'))

  def slack(self, msg, object_name, emoji=''):
    if 'SLACK_WEBHOOK' not in os.environ: return

    prefix = ''

    if 'GITHUB_RUN_ID' in os.environ:
      prefix += '<'
      prefix += os.environ['GITHUB_SERVER_URL'] + '/' + os.environ['GITHUB_REPOSITORY'] + '/actions/runs/' + os.environ['GITHUB_RUN_ID']
      prefix += '|'
      prefix += os.environ['GITHUB_RUN_NUMBER']
      prefix += '> '

    nb = os.environ.get('NOTEBOOK')
    if nb and nb != object_name:
      prefix += f'*{nb}.{object_name}* '
    else:
      prefix += f'*{object_name}* '

    prefix += emoji + ' '

    prefix += (datetime.now() + timedelta(hours=1)).strftime(f'%Y-%m-%d %H:%M ')

    prefix = prefix.strip() + '\n'
    Slack(url=os.environ['SLACK_WEBHOOK']).post(text=prefix + msg)

  async def publish(self, df, object_name, downloads, update_timestamps=False):
    print('infinities:', flush=True)
    m = (df == np.inf)
    inf = df.loc[m.any(axis=1), m.any(axis=0)]
    print(inf.head(), flush=True)
    print('nan:', flush=True)
    m = (df == np.nan)
    nan = df.loc[m.any(axis=1), m.any(axis=0)]
    print(nan.head(), flush=True)
    print(df.dtypes)

    with File(f'artifacts/publish/{object_name}.csv') as f:
      sort(df).to_csv(f, index=False)

    print('updating knack', flush=True)
    await self.updating(object_name, True)
    await self.update(object_name=object_name, df=df, slack=Munch(msg='\n'.join(downloads.actions), emoji=None))
    await self.timestamps(object_name, downloads.timestamps)
    await self.updating(object_name, False)
