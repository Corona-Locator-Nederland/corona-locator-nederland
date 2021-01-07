from munch import Munch
import knackpy
from jsonpath import JSONPath

class Knack:
  def __init__(self, app_id, api_key):
    self.headers = {
      'X-Knack-Application-Id': app_id,
      'X-Knack-REST-API-KEY': api_key,
      'content-type': 'application/json',
    }
    self.app = knackpy.App(app_id=os.environ['KNACK_APP_ID'],  api_key=os.environ['KNACK_API_KEY'], tzinfo='Europe/Amsterdam')
    with urlopen(f'https://loader.knack.com/v1/applications/{os.environ["KNACK_APP_ID"]}') as response:
      self.metadata = json.load(response)

  def find(self, path):
    found = JSONPath(path).parse(self.metadata)
    assert len(found) == 1, (len(found), path)
    return self.munch(found[0])

  def create(self, object_key, data):
    url = f'https://api.knack.com/v1/objects/{object_key}/records'
    res = requests.put(url=url, data=data, headers=self.headers)

  def modify(self, object_key, record_id, data):
    url = f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}'
    res = requests.put(url=url, data=data, headers=self.headers)

  def delete(self, object_key, record_id):
    url = f'https://api.knack.com/v1/objects/{object_key}/records/{record_id}'
    res = requests.delete(url=url, headers=self.headers)

  def get(self, object_key):
    url = f'https://api.knack.com/v1/objects/{object_key}/records'
    records = []
    page = 1
    while True:
      res = requests.get(url=f'{url}?page={page}', headers=self.headers).json()
      records += res['records']
      if res['current_page'] != res['total_pages']:
        page += 1
      else:
        break
    return self.munch(records)

#    @retry(HTTPError, delay=1, tries=10) # absolutely ridiculous
#    def record(self, method, data, obj):
#      self.app.record(method=method, data=data, obj=obj)
  
  def munch(self, records):
    if type(records) == list:
      return [ Munch.fromDict(rec) for rec in records ]
    elif type(records) == dict:
      return Munch.fromDict(records)
    else:
      raise ValueError(f'Unexpected type {str(type(records))}')

  def update(self, scene, view, df):
    print('x')
    view = self.find(f'$.application.scenes[?(@.name=={json.dumps(scene)})].views[?(@.name=={json.dumps(view)})]')
    source = view.source.object
    obj = self.find(f'$.application.objects[?(@.key=="{source}")]')
    mapping = { field.name: field.key for field in obj.fields }
    id_field = obj.identifier

    #ist = [dict(rec) for rec in self.app.get(source)]
    ist = self.get(obj.key)
    soll = df.rename(columns=mapping).to_dict('records')
    keys = self.munch({
      'ist': { rec[id_field]: rec.id for rec in ist},
      'soll': [rec[id_field] for rec in soll],
    })
    print(keys)

    with timer() as t:
      for rec in ist:
        if rec[id_field] not in keys.soll:
          print('deleting', rec[id_field])
          #self.delete(method='delete', data=rec, obj=obj['key'])
          self.delete(object_key=obj.key, record_id=rec.id)
      for rec in soll:
        if record_id := keys.ist.get(rec[id_field]):
          print('updating', rec[id_field])
          #self.record(method='update', data={**rec, 'id': keys['ist'][rec[id_field]]}, obj=obj['key'])
          self.modify(object_key=obj.key, record_id=record_id, data=rec)
        else:
          print('creating', rec[id_field])
          self.create(objject_key=obj.key, data=rec)
          #self.record(method="create", data=rec, obj=obj['key'])
