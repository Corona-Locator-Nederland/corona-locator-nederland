#!/usr/bin/env python3

from ruamel.yaml import YAML
yaml=YAML()
import threading
import sys
from io import StringIO
import shlex

with open('.github/workflows/publish.yml', encoding='utf-8') as f:
  action = f.read()

script = ''
for notebook in yaml.load(StringIO(action))['jobs']['publish']['strategy']['matrix']['notebook']:
  for step in yaml.load(StringIO(action.replace('${{ matrix.notebook }}', notebook)))['jobs']['publish']['steps']:
    if step.get('id') == 'notebook':
      if 'env' in step:
        for var, val in step['env'].items():
          script += f"export {var}={shlex.quote(val)}\n"
      script += step['run'] + "\n"
print(script)

#  steps = action['jobs']['publish']['steps']
#  script = [step['run'] for step in steps if step.get('name') == 'Run notebooks'][0]
#
#  script = "set -x\nexport CI=true\ngit pull --rebase\n" + script
#  #print(script)
#
#  #checkin = [step['with']['file_pattern'] for step in steps if step.get('uses') == 'stefanzweifel/git-auto-commit-action@v4'][0]
#  #script += f"\ngit add -u\ngit add {checkin}\n"
#
#  print(script)
