import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random as rd
import pickle
from sklearn.preprocessing import StandardScaler
import itertools
import sys
from PIL import Image, ImageDraw

#preload images
body = Image.open('body.png', 'r')
body_mask = Image.open('body_mask.png', 'r')
head=body.resize((100,100))
head_mask=body_mask.resize((100,100))
limb = Image.open('limb.png', 'r')
limb_mask = Image.open('limb_mask.png', 'r')
limb_w, limb_h = limb.size
limb=limb.resize((limb_w, limb_h//2))
limb_mask=limb_mask.resize((limb_w, limb_h//2))
limb_w, limb_h = limb.size



import umap.umap_ as umap
reducer = umap.UMAP()



def draw_animal(ears_l, neck_l, legs_l, tail_l):

	ear=limb.resize((limb_w,limb_h*ears_l))
	ear_mask=limb_mask.resize((limb_w,limb_h*ears_l))
	leg=limb.resize((limb_w,limb_h*legs_l))
	leg_mask=limb_mask.resize((limb_w,limb_h*legs_l))
	tail=limb.resize((limb_w,limb_h*tail_l))
	tail_mask=limb_mask.resize((limb_w,limb_h*tail_l))

	right_ear=ear.rotate(-30, expand=True)
	right_ear_mask=ear_mask.rotate(-30, expand=True)
	left_ear=right_ear.transpose(Image.FLIP_LEFT_RIGHT)
	left_ear_mask=right_ear_mask.transpose(Image.FLIP_LEFT_RIGHT)

	tail=tail.transpose(Image.ROTATE_90)
	tail_mask = tail_mask.transpose(Image.ROTATE_90)

	front_leg=leg.transpose(Image.FLIP_TOP_BOTTOM).rotate(20, expand=True)
	front_leg_mask=leg_mask.transpose(Image.FLIP_TOP_BOTTOM).rotate(20, expand=True)
	back_leg=front_leg.transpose(Image.FLIP_LEFT_RIGHT)
	back_leg_mask=front_leg_mask.transpose(Image.FLIP_LEFT_RIGHT)


	#print('finished loading images')

	#ear=ear.resize((100,100))
	#ear_mask=ear_mask.resize((100,100))
	#tail = Image.open('tail.png', 'r')

	body_w, body_h = body.size
	head_w, head_h = head.size
	ear_w, ear_h = right_ear.size
	tail_w, tail_h = tail.size
	leg_w, leg_h = front_leg.size

	background = Image.new('RGBA', (1000, 1000), (255, 255, 255, 255))
	bg_w, bg_h = background.size

	#draw body
	offset_center_w, offset_center_h = ((bg_w - body_w) // 2, (bg_h - body_h) // 2)
	offset_center=(offset_center_w, offset_center_h)
	background.paste(body, offset_center)

	#draw neck and head
	neck=int(100+70*(neck_l**1/2)) #neck length
	offset = (bg_w//2 + neck - head_w//2, bg_h//2 - neck - head_h//2)
	background.paste(head, offset, mask=head_mask)
	draw = ImageDraw.Draw(background) 
	draw.line((bg_w//2, bg_h//2, bg_w//2 + neck, bg_h//2 - neck), fill=(0,0,0,255), width=10)


	#draw ears
	offset = (bg_w //2 + neck , bg_h // 2 - neck  - ear_h )
	background.paste(right_ear, offset, mask=right_ear_mask)
	offset = (bg_w //2 + neck - ear_w , bg_h // 2 - neck - ear_h )
	background.paste(left_ear, offset, mask=left_ear_mask)

	#draw legs

	offset = (bg_w //2 , bg_h // 2 + body_h//2 - 10)
	background.paste(front_leg, offset, mask=front_leg_mask)
	offset = (bg_w //2 - leg_w, bg_h // 2 + body_h//2 - 10)
	background.paste(back_leg, offset, mask=back_leg_mask)

	#draw tail
	offset = (bg_w//2 - tail_w - body_w//2, bg_h//2 - tail_h//2)
	background.paste(tail, offset, mask=tail_mask)

	background=background.resize((256,256))
	
	return background


def simple_set():
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
	return df	

def xor_set():

	df=pd.DataFrame().astype(int)
	values = range(1,5)

	for l in range(1,3):
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					name='random'
					if i>1:
						name='random'
					if i<2:
						name='longneck'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	for l in range(3,5):
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					name='random'
					if i>1:
						name='longneck'
					if i<2:
						name='random'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	float_col = df.select_dtypes(include=['float64']) 
	for col in float_col.columns.values:
		df[col] = df[col].astype('int64')

	first_column = df.pop('category')
	df.insert(0, 'category', first_column)

	return df
	#df.category.value_counts()

	#df=df.groupby('category').apply(lambda s: s.sample(100))

def hierarchical_set():
	df=pd.DataFrame().astype(int)
	values = range(1,5)

	for l in range(1,3):
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					name='random'
					if i>1:
						name='random'
					if i<2:
						name='shortneck'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	for l in range(3,5):
		for i in range(0,4):
			for j in range(0,4):
				for k in range(0,4):
					name='random'
					if i>1:
						name='longneck'
					if i<2:
						name='random'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	float_col = df.select_dtypes(include=['float64']) 
	for col in float_col.columns.values:
		df[col] = df[col].astype('int64')

	first_column = df.pop('category')
	df.insert(0, 'category', first_column)
	return df

def interdependent_set():
	df=pd.DataFrame().astype(int)
	values = range(1,5)

	for l in range(1,3):
		for i in range(0,4):
			for j in range(0,2):
				for k in range(0,4):
					name='random'
					if (i<2)&(k<2):
						name='longneck'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
			for j in range(2,4):
				for k in range(0,4):
					if (i<2)&(k>1):
						name='longneck'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	for l in range(3,5):
		for i in range(0,4):
			for j in range(0,2):
				for k in range(0,4):
					name='random'
					if (i>1)&(k<2):
						name='longneck'
					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
			for j in range(2,4):
				for k in range(0,4):
					name='random'
					if (i>1)&(k>1):
						name='longneck'

					new_row = {'category': name, 'neck': l, 'ears': values[i], 'tail': values[j], 'legs': values [k]}
					df = df.append(new_row, ignore_index=True)
		print("done for l=" +str(l))


	float_col = df.select_dtypes(include=['float64']) 
	for col in float_col.columns.values:
		df[col] = df[col].astype('int64')

	first_column = df.pop('category')
	df.insert(0, 'category', first_column)
	return df


def sliding_average(a, span):
	c=np.random.rand(256,256)
	for i in range(0, len(a), span):
		for j in range(0, len(a), span):
			b=a[i:i+span, j:j+span]
			m=np.mean(b)
			c[i:i+span, j:j+span]=m
	return c




'''
a=np.random.rand(256,256)

avg_a_2=sliding_average(a, 2)
avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 16)
avg_a_16=sliding_average(a, 16)
'''
#df=xor_set()
df=simple_set()

data_numerical = df[[ "ears", "neck", "legs", "tail"]].values

target=df['category'].tolist()

target_numerical=[t=='longneck' for t in target]
#pil_data=[draw_animal(d[0],d[1],d[2],d[3]).convert('L') for d in data_numerical]
data=[np.asarray(draw_animal(d[0],d[1],d[2],d[3]).convert('L')) for d in data_numerical]

data_2=[sliding_average(x, 8) for x in data]
data_2_flat=[x.flatten() for x in data_2]
'''avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 16)
avg_a_16=sliding_average(a, 16)'''
#a=draw_animal(data[0][0],data[0][1],data[0][2],data[0][3])

current_data=data_2_flat

reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
reducer.fit(current_data)




embedding = reducer.transform(current_data)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
print(embedding.shape)

plt.scatter(embedding[:, 0], embedding[:, 1], c=target_numerical, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()


'''

scaled_data = StandardScaler().fit_transform(data)

embedding = reducer.fit_transform(scaled_data)
#embedding.shape


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

green_patch = mpatches.Patch(color='green', label='long neck and ears or short neck and ears')
#blue_patch = mpatches.Patch(color='blue', label='short neck and ears')
orange_patch = mpatches.Patch(color='orange', label='random')



colours=['green', 'blue', 'orange']
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[colours[x] for x in df.category.map({"longneck":0, "shortneck":1, "random":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.legend(handles=[green_patch, orange_patch])

plt.title('UMAP projection of the dataset', fontsize=24);
'''

