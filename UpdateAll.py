#!/usr/bin/env python3

from ruamel.yaml import YAML
yaml=YAML()
import threading
import sys

with open('.github/workflows/publish.yml') as f:
  action = yaml.load(f)
  steps = action['jobs']['run']['steps']
  script = [step['run'] for step in steps if step.get('name') == 'Run notebooks'][0]
  script = "export CI=true\n" + script
  print(script)

  #result = run(script, shell=True, capture_output=True, text=True)
  #print(result.stdout)
  #print(result.stderr)
