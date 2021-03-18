# %%
from IPython import get_ipython
from IPython.core.display import display
ipy = get_ipython()
for nb in ['Dagoverzicht', 'LeeftijdsgroepenLandelijk', 'Regio']:
  ipy.run_line_magic('run', nb + '.ipynb')