#!/usr/bin/env python3

from ruamel.yaml import YAML
yaml=YAML()
import threading
import sys

with open('.github/workflows/publish.yml') as f:
  action = yaml.load(f)
  steps = action['jobs']['publish']['steps']
  script = [step['run'] for step in steps if step.get('name') == 'Run notebooks'][0]

  script = "set -x\nexport CI=true\ngit pull --rebase\n" + script
  #print(script)

  #checkin = [step['with']['file_pattern'] for step in steps if step.get('uses') == 'stefanzweifel/git-auto-commit-action@v4'][0]
  #script += f"\ngit add -u\ngit add {checkin}\n"

  print(script)
