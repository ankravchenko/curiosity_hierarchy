import numpy as np

from PIL import Image 
import os

import math
from random import randint, random

from tifffile import imsave


import matplotlib.pyplot as plt

#code sequences as lists of int: [1, 2 , 3, 4]

#creates a row of coloured squares from a sequence, returns image
def create_row(sq):
	dir = "squares"
	images = []
	# Loop through the images in the directory and append them to the list
	for file in os.listdir(dir):
	  img = Image.open(os.path.join(dir, file))
	  images.append(img)

	#dimensions of the new image
	width = sum(img.size[0] for img in images)
	height = max(img.size[1] for img in images)
	result = Image.new("RGB", (width, height))

	# Loop through the images and paste them into the resulting image
	x_offset = 0
	for s in sq:
	  img=images[s]
	  result.paste(img, (x_offset, 0))
	  x_offset += img.size[0]

	# Save the resulting image
	return result


#for squares
def sliding_average(a, span):
	c=np.random.rand(a.shape[0],a.shape[1])
	for i in range(0, len(a), span):
		for j in range(0, len(a), span):
			b=a[i:i+span, j:j+span]
			m=np.mean(b)
			c[i:i+span, j:j+span]=m
	return c

def sliding_average_string(a, span):
	c=np.random.rand(len(a))
	for i in range(0, len(a), span):
		if (i+span) < len(a):
			b=a[i:i+span]
			m=np.mean(b)
			c[i:i+span]=m
	return c
'''
def normalize(matrix):
    min_val = 0
    max_val = 4
    
    # Find the minimum and maximum values in the matrix
    for row in matrix:
        for val in row:
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
    
    # Normalize the matrix
    for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            matrix[i,j] = val / 4 * 2 - 1
    return matrix
'''

'''
def normalize(m):
	min_val=np.amin(m)
	max_val=np.amax(m)
	m_new=np.ones((m.shape[0],m.shape[1]))
	for i in range(0, m.shape[0]):
		for j in range(0, m.shape[1]):
			m_new[i,j] = (m[i,j] - min_val) / (max_val - min_val) * 2 - 1
			
	#print('diff: ', np.amax(m) - np.amin(m))
	return m_new 
'''
def normalize(m):
	min_val=np.amin(m)
	max_val=np.amax(m)
	m_new=np.ones(m.shape[0])
	for i in range(0, m.shape[0]):
			m_new[i] = (m[i] - (max_val-min_val)/2) / (max_val - min_val) * 2 - 1
			
	#print('diff: ', np.amax(m) - np.amin(m))
	return m_new 
'''
def overlap(m1, m2):
	m0=m1*m1
	m=m1*m2
	return (m0.sum(axis=(0,1))-m.sum(axis=(0,1))) / (m0.shape[0]*m0.shape[1])
'''

def overlap(m1, m2):
	m0=m1*m1
	m=m1*m2
	r= (m0.sum(axis=(0))-m.sum(axis=(0))) / (m0.shape[0])

	return r

def generate_strings_array(length: int):
    if length == 0:
        yield []
    else:
        for ch in [0, 1, 2, 3]:
            for string in generate_strings_array(length - 1):
                yield [ch] + string


def generate_mirror_strings_array(length: int):
    if length == 0:
        yield []
    else:
        for ch in [0, 1, 2, 3]:
            for mirror_string in generate_mirror_strings_array(length - 1):
                yield [ch] + mirror_string + [ch]

#for grammar subset
def generate_random_string(length: int):
    s=[]	
    for i in range(0,length):
        s.append(randint(0,4))
    return s



def calculate_total_complexity(cf_string):

    avg_a_2=sliding_average_string(cf_string, 2)
    avg_a_4=sliding_average_string(cf_string, 4)
    avg_a_8=sliding_average_string(cf_string, 8)
    avg_a_16=sliding_average_string(cf_string, 16)
    avg_a_32=sliding_average_string(cf_string, 32)
    avg_a_64=sliding_average_string(cf_string, 64)
    avg_a_128=sliding_average_string(cf_string, 128)
    avg_a_256=sliding_average_string(cf_string, 256)

    '''
    print("0-2: ", overlap(normalize(cf_string),normalize(avg_a_2)))
    print("2-4: ", overlap(normalize(avg_a_2),normalize(avg_a_4)))
    print("4-8: ", overlap(normalize(avg_a_4),normalize(avg_a_8)))
    print("8-16: ", overlap(normalize(avg_a_8),normalize(avg_a_16)))
    print("16-32: ", overlap(normalize(avg_a_16),normalize(avg_a_32)))
    print("32-64: ", overlap(normalize(avg_a_32),normalize(avg_a_64)))
    print("64-128: ", overlap(normalize(avg_a_64),normalize(avg_a_128)))
    print("128-256: ", overlap(normalize(avg_a_128),normalize(avg_a_256)))
    '''
    sizes=[cf_string, avg_a_2, avg_a_4, avg_a_8, avg_a_16]#, avg_a_32, avg_a_64, avg_a_128]
    total_complexity=0
    for i in range(0, len(sizes)-1):
        total_complexity=total_complexity+overlap(normalize(sizes[i]),normalize(sizes[i+1]))

    return [overlap(normalize(sizes[0]),normalize(sizes[1])), overlap(normalize(sizes[1]),normalize(sizes[2])), overlap(normalize(sizes[2]),normalize(sizes[3])), overlap(normalize(sizes[3]),normalize(sizes[4])), total_complexity]


##########function end###############

file1 = open('complexity_mirror_copy_4_100.txt', 'w')

#len of 100 for total complexity, 2-4, 4-8, 8-16, 

cf_all=[]
cs_all=[]
random_all=[]
type0_all=[]


for l in range(0,20):#20
	print('iteration running: ', l)
	cf_total=[[0]*98 for _ in range(5)]
	cs_total=[[0]*98 for _ in range(5)]
	random_total=[[0]*98 for _ in range(5)]
	type0_total=[[0]*98 for _ in range(5)]

	for n in range(2,100):#100

		#print('n = ',n*2)

		mirror_strings=[]
		copy_strings=[]
		random_strings=[]
		type0_strings=[]

		'''
		#generates full grammar
		for half_string in generate_strings_array(n):
		    copy_string=half_string+half_string
		    hs=half_string.copy()
		    hs.reverse()
		    mirror_string=half_string+hs
		    copy_strings.append(copy_string)
		    mirror_strings.append(mirror_string)
		'''

		#generates l sentences from a grammar
		l=100
		for i in range(1,l):
		    half_string = generate_random_string(n)
		    hs=half_string.copy()
		    hs = [x+1 for x in hs]
		    copy_string=half_string+hs
		    hs.reverse()
		    mirror_string=half_string+hs
		    r_string = generate_random_string(n*2)
		    hr_string = generate_random_string(n)
		    hs.reverse()
		    hs1=half_string.copy()
		    hs1.reverse()
		    type0_string=half_string.copy()+hr_string+hs1+hs
		    copy_strings.extend(copy_string)
		    mirror_strings.extend(mirror_string)
		    random_strings.extend(r_string)	
		    type0_strings.extend(type0_string)	
		#print('grammars generated')

		cf=np.asarray(mirror_strings)
		cs=np.asarray(copy_strings)
		cr=np.asarray(random_strings)
		c0=np.asarray(type0_strings)

	
		#cf_normalized =cf / cf.sum(axis=0)
		#cs_normalized =cs / cs.sum(axis=0)
		#string

		cf_string=cf
		cs_string=cs
		cr_string=cr
		c0_string=c0


		#print("context-free grammar:")
		[o_0_2, o_2_4, o_4_8, o_8_16, total_complexity] = calculate_total_complexity(cf_string)
		#print("total_complexity: ", total_complexity) 

		#file1.write(str(o_4_8))
		#file1.write('\t')

		cf_total[0][n-2]=total_complexity
		cf_total[1][n-2]=o_0_2
		cf_total[2][n-2]=o_2_4
		cf_total[3][n-2]=o_4_8
		cf_total[4][n-2]=o_8_16
		
		#print("context-senstitive grammar:")
		[o_0_2, o_2_4, o_4_8, o_8_16, total_complexity] = calculate_total_complexity(cs_string)
		#print("total_complexity: ", total_complexity) 

		#file1.write(str(o_4_8))
		#file1.write('\n')

		cs_total[0][n-2]=total_complexity
		cs_total[1][n-2]=o_0_2
		cs_total[2][n-2]=o_2_4
		cs_total[3][n-2]=o_4_8
		cs_total[4][n-2]=o_8_16


		[o_0_2, o_2_4, o_4_8, o_8_16, total_complexity] = calculate_total_complexity(cr_string)
		#print("total_complexity: ", total_complexity) 

		#file1.write(str(o_4_8))
		#file1.write('\n')

		random_total[0][n-2]=total_complexity
		random_total[1][n-2]=o_0_2
		random_total[2][n-2]=o_2_4
		random_total[3][n-2]=o_4_8
		random_total[4][n-2]=o_8_16

		[o_0_2, o_2_4, o_4_8, o_8_16, total_complexity] = calculate_total_complexity(c0_string)
		#print("total_complexity: ", total_complexity) 

		#file1.write(str(o_4_8))
		#file1.write('\n')

		type0_total[0][n-2]=total_complexity
		type0_total[1][n-2]=o_0_2
		type0_total[2][n-2]=o_2_4
		type0_total[3][n-2]=o_4_8
		type0_total[4][n-2]=o_8_16
		'''
		ca_4=sliding_average_string(cs_string, 4)


		ca_8=sliding_average_string(cs_string, 8)

		ca_16=sliding_average_string(cs_string, 16)

		ca_32=sliding_average_string(cs_string, 32)
		ca_64=sliding_average_string(cs_string, 64)
		'''
	cf_all.append(cf_total)
	cs_all.append(cs_total)
	random_all.append(random_total)
	type0_all.append(type0_total)

cf_all_np=np.asarray(cf_all)
cf_all_np_mean = np.mean(cf_all_np, axis=0)

cs_all_np=np.asarray(cs_all)
cs_all_np_mean = np.mean(cs_all_np, axis=0)

random_all_np=np.asarray(random_all)
random_all_np_mean = np.mean(random_all_np, axis=0)


type0_all_np=np.asarray(type0_all)
type0_all_np_mean = np.mean(type0_all_np, axis=0)



# settings
h, w = 10, 10        # for raster image
nrows, ncols = 5, 4  # array of sub-plots
figsize = [10, 10]     # figure size, inches


# create figure (fig), and array of axes (ax)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)


x = np.arange(2, 100)


row_names=['total complexity', '0-2', '2-4', '4-8', '8-16']
column_names=['Context-free', 'Context senstitive', 'Random string', 'Type 0']
# plot simple raster image on each sub-plot
for i, axi in enumerate(ax.flat):
    # i runs from 0 to (nrows*ncols-1)
    # axi is equivalent with ax[rowid][colid]

    # get indices of row/column
    rowid = row_names [i // ncols]
    colid = column_names[i % ncols]
    if (colid == 'Context-free'):
	   # print('cf, picking row '+str(i // ncols))
	    y = cf_all_np_mean[i // ncols] 
    if (colid == 'Context senstitive'):
	    y = cs_all_np_mean[i // ncols] 
	    #print('cs, picking row '+ str(i // ncols))	 
    if (colid == 'Random string'):
	    y = random_all_np_mean[i // ncols]    
    if (colid == 'Type 0'):
	    y = type0_all_np_mean[i // ncols] 
    axi.set_ylim([0, 1])
    axi.plot(x, y, color ="red")
    #print(str(y[0]))
    # write row/col indices as axes' title for identification
    axi.set_title(str(colid)+": "+str(rowid))

# one can access the axes by ax[row_id][col_id]

plt.tight_layout(True)
plt.show()


################DEBUG#########################
'''
rand = np.random.choice([x for x in [-1,1]],l*n*2)
print("random string:")
total_complexity = calculate_total_complexity(rand)
print("total_complexity: ", total_complexity) 
'''

file1.close()

'''
>>> stats.ttest_ind(cf_all_np_mean[0], cs_all_np_mean[0])
Ttest_indResult(statistic=-0.6140039012041903, pvalue=0.5399320824647673)
>>> stats.ttest_ind(cf_all_np_mean[1], cs_all_np_mean[])
  File "<stdin>", line 1
    stats.ttest_ind(cf_all_np_mean[1], cs_all_np_mean[])
                                                      ^
SyntaxError: invalid syntax
>>> stats.ttest_ind(cf_all_np_mean[1], cs_all_np_mean[1])
Ttest_indResult(statistic=0.20083144789090812, pvalue=0.8410407665140255)
>>> stats.ttest_ind(cf_all_np_mean[2], cs_all_np_mean[2])
Ttest_indResult(statistic=-1.0037418281451356, pvalue=0.31675338141963116)
>>> stats.ttest_ind(cf_all_np_mean[3], cs_all_np_mean[3])
Ttest_indResult(statistic=0.25581732978073785, pvalue=0.7983629309749216)
>>> 

'''

'''
>>> stats.ttest_ind(cs_all_np_mean[1], type0_all_np_mean[1])
Ttest_indResult(statistic=4.2234336828859576, pvalue=5.559096402700161e-05)
>>> stats.ttest_ind(cs_all_np_mean[0], type0_all_np_mean[0])
Ttest_indResult(statistic=1.6087811615773864, pvalue=0.11101779364226988)
>>> stats.ttest_ind(cs_all_np_mean[2], type0_all_np_mean[2])
Ttest_indResult(statistic=2.0062767807048933, pvalue=0.04770026090012435)
>>> stats.ttest_ind(cs_all_np_mean[3], type0_all_np_mean[3])
Ttest_indResult(statistic=0.22824669246664026, pvalue=0.8199502472951979)
>>> stats.ttest_ind(cs_all_np_mean[4], type0_all_np_mean[4])
Ttest_indResult(statistic=0.2188156655172901, pvalue=0.8272680375323211)
>>> 


'''


