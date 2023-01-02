import numpy as np

from PIL import Image 
import os

import math
from random import randint, random

from tifffile import imsave

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
        s.append(randint(0,3))
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


    print("0-2: ", overlap(normalize(cf_string),normalize(avg_a_2)))
    print("2-4: ", overlap(normalize(avg_a_2),normalize(avg_a_4)))
    print("4-8: ", overlap(normalize(avg_a_4),normalize(avg_a_8)))
    print("8-16: ", overlap(normalize(avg_a_8),normalize(avg_a_16)))
    print("16-32: ", overlap(normalize(avg_a_16),normalize(avg_a_32)))
    print("32-64: ", overlap(normalize(avg_a_32),normalize(avg_a_64)))
    print("64-128: ", overlap(normalize(avg_a_64),normalize(avg_a_128)))
    print("128-256: ", overlap(normalize(avg_a_128),normalize(avg_a_256)))

    sizes=[cf_string, avg_a_2, avg_a_4, avg_a_8, avg_a_16]#, avg_a_32, avg_a_64, avg_a_128]
    total_complexity=0
    for i in range(0, len(sizes)-1):
        total_complexity=total_complexity+overlap(normalize(sizes[i]),normalize(sizes[i+1]))

    return total_complexity


##########function end###############

file1 = open('complexity_mirror_copy_4_100.txt', 'w')
n=20

print('n = ',n*2)

mirror_strings=[]
copy_strings=[]


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
    copy_string=half_string+half_string
    hs=half_string.copy()
    hs.reverse()
    mirror_string=half_string+hs
    copy_strings.extend(copy_string)
    mirror_strings.extend(mirror_string)
	
print('grammars generated')

cf=np.asarray(mirror_strings)

cs=np.asarray(copy_strings)


#cf_normalized =cf / cf.sum(axis=0)
#cs_normalized =cs / cs.sum(axis=0)
#string

cf_string=cf
cs_string=cs



print("context-free grammar:")
total_complexity = calculate_total_complexity(cf_string)
print("total_complexity: ", total_complexity) 

file1.write(total_complexity)
file1.write('\t')
  
'''
a_4=sliding_average_string(cf_string, 4)
a_8=sliding_average_string(cf_string, 8)
a_16=sliding_average_string(cf_string, 16)
a_32=sliding_average_string(cf_string, 32)
a_64=sliding_average_string(cf_string, 64)
'''

print("context-senstitive grammar:")
total_complexity = calculate_total_complexity(cs_string)
print("total_complexity: ", total_complexity) 

file1.write(total_complexity)
file1.write('\n')

'''
ca_4=sliding_average_string(cs_string, 4)


ca_8=sliding_average_string(cs_string, 8)

ca_16=sliding_average_string(cs_string, 16)

ca_32=sliding_average_string(cs_string, 32)
ca_64=sliding_average_string(cs_string, 64)
'''

################DEBUG#########################
'''
rand = np.random.choice([x for x in [-1,1]],l*n*2)
print("random string:")
total_complexity = calculate_total_complexity(rand)
print("total_complexity: ", total_complexity) 
'''

file1.close()
