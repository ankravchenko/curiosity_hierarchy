import numpy as np

from PIL import Image 
import os

import math
from random import randint, random

from tifffile import imsave

import itertools

import matplotlib.pyplot as plt

import scipy.stats as stats

from numpy import linalg as LA


SENTENCE_N = 100
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
        for ch in [0, 1, 2, 3, 4]:
            for string in generate_strings_array(length - 1):
                yield [ch] + string


def generate_mirror_strings_array(length: int):
    if length == 0:
        yield []
    else:
        for ch in [0, 1, 2, 3, 4]:
            for mirror_string in generate_mirror_strings_array(length - 1):
                yield [ch] + mirror_string + [ch]

#for grammar subset
def generate_random_string(length: int):
    s=[]	
    for i in range(0,length):
        s.append(randint(0,5))
    return s

def all_pair_correlations(sentence_length, all_sentences):
	n=sentence_length #well, actually sentence_length/2
	#alph length is always 5 here
	all_letter_pairs=list(itertools.combinations(range(5), 2))
	letter_pair_positional_cors={} #(0,1): [cor_ij for letters 0 and 1 for all i and j position]
	positionals=list(itertools.combinations(range(n*2), 2))
	
	for pair in all_letter_pairs:
			cor_matrix=np.zeros((n*2,n*2))

			for position_pair in positionals:	
				cnt=0	
		        	#calculate probability this letter pair will end up in this position: how many times it happened to be there/total times. 
				for j in range(0,SENTENCE_N): #there has to be a way to map instead of iterating. also make this a separate function for clarity FIXME
					if (all_sentences[j][position_pair[0]],all_sentences[j][position_pair[1]]) == pair:
						cnt=cnt+1
					elif (all_sentences[j][position_pair[1]],all_sentences[j][position_pair[0]]) == pair:
						cnt=cnt+1
				cor_matrix[position_pair[0]][position_pair[1]]=cnt/SENTENCE_N
				cor_matrix[position_pair[1]][position_pair[0]]=cnt/SENTENCE_N	
			letter_pair_positional_cors[pair]=cor_matrix
	return letter_pair_positional_cors	

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


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
##########function end###############

def extract_above_diagonal(matrix):
    n = matrix.shape[0]
    # create a mask of the upper triangle of the matrix
    mask = np.triu(np.ones((n, n)), k=0)
    # extract the elements above the diagonal using the mask
    above_diag = matrix[mask == 1]
    return above_diag

# test the function
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(extract_above_diagonal(matrix))


file1 = open('complexity_mirror_copy_4_100.txt', 'w')

#len of 100 for total complexity, 2-4, 4-8, 8-16, 

cf_all=[]
cs_all=[]
random_all=[]
type0_all=[]

all_letter_pairs=list(itertools.combinations(range(5), 2))
letter_pair_positional_cors={} #(0,1): [cor_ij for letters 0 and 1 for all i and j position]

for l in range(0,1):#20
	print('iteration running: ', l)
	'''cf_total=[[0]*98 for _ in range(5)]
	cs_total=[[0]*98 for _ in range(5)]
	random_total=[[0]*98 for _ in range(5)]
	type0_total=[[0]*98 for _ in range(5)]'''

	for n in range(10,11):#100

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
		f=SENTENCE_N
		for i in range(0,f):
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
		    type0_string=half_string.copy()+generate_random_string(n)
		    copy_strings.append(copy_string)
		    mirror_strings.append(mirror_string)
		    random_strings.append(r_string)	
		    type0_strings.append(type0_string)	
		#print('grammars generated')
		copy_strings_cors=all_pair_correlations(n, copy_strings)
		mirror_strings_cors=all_pair_correlations(n, mirror_strings)
		random_strings_cors=all_pair_correlations(n, random_strings)
		type0_strings_cors=all_pair_correlations(n, type0_strings)

		print('!!!!')
		'''
		positionals=list(itertools.combinations(range(n*2), 2))
		for pair in all_letter_pairs:
			cor_matrix=np.zeros((n*2,n*2))

			for position_pair in positionals:	
				cnt=0	
		        	#calculate probability this letter pair will end up in this position: how many times it happened to be there/total times. 
				for j in range(0,100): #there has to be a way to map instead of iterating. also make this a separate function for clarity FIXME
					if (copy_strings[j][position_pair[0]],copy_strings[j][position_pair[1]]) == pair:
						cnt=cnt+1
					elif (copy_strings[j][position_pair[1]],copy_strings[j][position_pair[0]]) == pair:
						cnt=cnt+1
				cor_matrix[position_pair[0]][position_pair[1]]=cnt/100
				cor_matrix[position_pair[1]][position_pair[0]]=cnt/100	
			letter_pair_positional_cors[pair]=cor_matrix	
		'''
		#fixme make this into a dictonary so that only grammar
		copy_cor_mean=np.zeros((n*2,n*2))	
		for val in copy_strings_cors.values():
			copy_cor_mean += val
		copy_cor_mean = copy_cor_mean / len(copy_strings_cors)
		mean_y = np.mean(copy_cor_mean)
		copy_cor_abs_mean=np.full(copy_cor_mean.shape, mean_y)	


		copy_total_cor=np.zeros((n*2,n*2))	
		for val in copy_strings_cors.values():
			copy_total_cor = copy_total_cor + val - copy_cor_abs_mean
		#copy_total_cor = copy_total_cor / len(copy_strings_cors)
		


		mirror_cor_mean=np.zeros((n*2,n*2))	
		for val in mirror_strings_cors.values():
			mirror_cor_mean += val
		mirror_cor_mean = mirror_cor_mean / len(mirror_strings_cors)
		mean_y = np.mean(mirror_cor_mean)
		mirror_cor_abs_mean=np.full(mirror_cor_mean.shape, mean_y)	

		
		mirror_total_cor=np.zeros((n*2,n*2))	
		for val in mirror_strings_cors.values():
			mirror_total_cor = mirror_total_cor + val - mirror_cor_abs_mean
		#mirror_total_cor = mirror_total_cor #/ len(mirror_strings_cors)


		random_cor_mean=np.zeros((n*2,n*2))	
		for val in random_strings_cors.values():
			random_cor_mean += val
		random_cor_mean = random_cor_mean / len(random_strings_cors)
		mean_y = np.mean(random_cor_mean)
		random_cor_abs_mean=np.full(random_cor_mean.shape, mean_y)	

		random_total_cor=np.zeros((n*2,n*2))	
		for val in random_strings_cors.values():
			random_total_cor = random_total_cor + val - random_cor_abs_mean


		type0_cor_mean=np.zeros((n*2,n*2))	
		for val in type0_strings_cors.values():
			type0_cor_mean += val
		type0_cor_mean = type0_cor_mean / len(type0_strings_cors)
		mean_y = np.mean(type0_cor_mean)
		type0_cor_abs_mean=np.full(type0_cor_mean.shape, mean_y)	

		type0_total_cor=np.zeros((n*2,n*2))	
		for val in type0_strings_cors.values():
			type0_total_cor = type0_total_cor + val - type0_cor_abs_mean
		#random_total_cor = random_total_cor# / len(mirror_strings_cors)

'''
#normalize vectors so that the sum is one for it to be a probability distribution
random_total_cor  = random_total_cor  / sum(random_total_cor )
mirror_total_cor  = mirror_total_cor  / sum(mirror_total_cor )
copy_total_cor  = copy_total_cor  / sum(copy_total_cor )
'''


print('>>>')
plt.clf()


random_total_cor_flat  = random_total_cor.flatten()
mirror_total_cor_flat  = mirror_total_cor.flatten()
copy_total_cor_flat  = copy_total_cor.flatten()
diff=mirror_total_cor - copy_total_cor


'''
print('>>>')
plt.clf()
a = plt.hist(x=random_total_cor, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('random strings')
plt.savefig('hist_random.png')

plt.clf()
print('>>>')

a = plt.hist(x=mirror_total_cor, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('mirror strings')
plt.savefig('hist_mirror.png')
plt.clf()

a = plt.hist(x=copy_total_cor, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('copy strings')
plt.savefig('hist_copy.png')

plt.clf()


a = plt.hist(x=diff, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.title('diff')
plt.savefig('hist_diff.png')
plt.clf()
'''

#all_letter_pairs=list(itertools.combinations(range(5), 2))


all_pos_pairs = list(itertools.combinations_with_replacement(range(0,20), 2))
ttt=len(all_pos_pairs)
x=range(0,ttt)

x_ticks=all_pos_pairs
x_ticks=[str(a) for a in all_pos_pairs]
y = extract_above_diagonal(mirror_total_cor)

plt.bar(x,y,align='center') # A bar chart
plt.xlabel('pairs')
plt.ylabel('l1')
#ax.set_xticklabels(all_pos_pairs)
#for i in range(len(y)):
#    plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
plt.savefig('hist_mirror.png')
plt.clf()



y = extract_above_diagonal(copy_total_cor)

plt.bar(x,y,align='center') # A bar chart
plt.xlabel('pairs')
plt.ylabel('l1')
#plt.xticklabels(all_pos_pairs)
#for i in range(len(y)):
#    plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
plt.savefig('hist_copy.png')
plt.clf()



y = extract_above_diagonal(random_total_cor)
plt.bar(x,y,align='center') # A bar chart
plt.xlabel('pairs')
plt.ylabel('l1')
#plt.xticklabels(all_pos_pairs)
#for i in range(len(y)):
#    plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
plt.savefig('hist_random.png')
plt.clf()

diff=extract_above_diagonal(copy_total_cor)-extract_above_diagonal(mirror_total_cor)
y=diff
plt.bar(x,y,align='center') # A bar chart
plt.xlabel('pairs')
plt.ylabel('l1')
#plt.xticklabels(all_pos_pairs)
#for i in range(len(y)):
#    plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
plt.savefig('hist_diff.png')
plt.clf()

y=extract_above_diagonal(type0_total_cor)
plt.bar(x,y,align='center') # A bar chart
plt.xlabel('pairs')
plt.ylabel('l1')
#plt.xticklabels(all_pos_pairs)
#for i in range(len(y)):
#    plt.hlines(y[i],0,x[i]) # Here you are drawing the horizontal lines
plt.savefig('hist_type0.png')
plt.clf()

'''
#Scipy's entropy function will calculate KL divergence if feed two vectors p and q, each representing a probability distribution. If the two vectors aren't pdfs, it will normalize then first.
kl_copy=stats.entropy(random_total_cor, copy_total_cor, axis=(0,1))
kl_mirror=stats.entropy(random_total_cor, mirror_total_cor, axis=(0,1))
kl_random=stats.entropy(random_total_cor, random_total_cor, axis=(0,1))'''

