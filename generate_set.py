import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random as rd
import pickle
from sklearn.preprocessing import StandardScaler

import itertools


import umap.umap_ as umap
reducer = umap.UMAP()

df=pd.DataFrame().astype(int)
values = range(1,5)

for l in range(1,5):
	for i in range(0,4):
		for j in range(0,4):
			for k in range(0,4):
				name='random'
				if l>2:
					name='longneck'
				if l<3:
					name='shortneck'
				new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
				df = df.append(new_row, ignore_index=True)
	print("done for l=" +str(l))

float_col = df.select_dtypes(include=['float64']) 
for col in float_col.columns.values:
	df[col] = df[col].astype('int64')

first_column = df.pop('category')
df.insert(0, 'category', first_column)

#df.category.value_counts()

#df=df.groupby('category').apply(lambda s: s.sample(100))

data = df[
    [
        "neck",
        "ears",
        "tail",
        "legs",
    ]
].values

scaled_data = StandardScaler().fit_transform(data)

embedding = reducer.fit_transform(scaled_data)
#embedding.shape


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

green_patch = mpatches.Patch(color='green', label='long neck')
blue_patch = mpatches.Patch(color='blue', label='short neck')
#orange_patch = mpatches.Patch(color='orange', label='medium neck')



colours=['green', 'blue', 'orange']
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[colours[x] for x in df.category.map({"longneck":0, "shortneck":1, "random":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.legend(handles=[green_patch, blue_patch])

plt.title('UMAP projection of the dataset', fontsize=24);


