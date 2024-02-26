#%%
import pandas as pd
from matplotlib import axes, pyplot as plt
from pylab import rcParams
import torch
rcParams['figure.figsize'] = 7,7 
import seaborn as sns
import numpy as np
sns.set(color_codes=True, font_scale=1.2)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2

from heatmap import heatmap, corrplot

#%%
plt.figure(figsize=(3.2, 3.2))
# Load the Automobile dataset.
# This gets a cleaned version of UCI dataset found at http://archive.ics.uci.edu/ml/datasets/automobile

data = pd.read_csv('./images/autos.clean.csv')
df2 = data.corr()
value = torch.load('matrices/weighting2-.tch').numpy()
df = pd.DataFrame(value[26], 
                columns=list('012345678'), index=list('012345678'))
corrplot(df, size_scale=100)
#%%
