import numpy as np

from PIL import Image 
import os

import math

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


def normalize(m):
	min_val=np.amin(m)
	max_val=np.amax(m)
	m_new=np.ones((m.shape[0],m.shape[1]))
	for i in range(0, m.shape[0]):
		for j in range(0, m.shape[1]):
			m_new[i,j] = (m[i,j] - min_val) / (max_val - min_val) * 2 - 1
			
	#print('diff: ', np.amax(m) - np.amin(m))
	return m_new 

def overlap(m1, m2):
	m0=m1*m1
	m=m1*m2
	return (m0.sum(axis=(0,1))-m.sum(axis=(0,1))) / (m0.shape[0]*m0.shape[1])

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




n=4

print('n = ',n*2)

mirror_strings=[]
copy_strings=[]
for half_string in generate_strings_array(n):
    copy_string=half_string+half_string
    hs=half_string.copy()
    hs.reverse()
    mirror_string=half_string+hs
    copy_strings.append(copy_string)
    mirror_strings.append(mirror_string)


print('grammars generated')

cf=np.asarray(mirror_strings)

cs=np.asarray(copy_strings)



a=cf

r=cf.shape[0]*cf.shape[1]
b=cf.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_original.png')

cf_normalized =cf / cf.sum(axis=(0,1))



avg_a_2=sliding_average(cf, 2)
b=avg_a_2.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg2.png')


avg_a_4=sliding_average(cf, 4)
b=avg_a_4.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg4.png')

avg_a_8=sliding_average(cf, 8)
b=avg_a_8.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg8.png')

avg_a_16=sliding_average(cf, 16)
b=avg_a_16.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg16.png')

avg_a_32=sliding_average(cf, 32)
b=avg_a_32.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg32.png')

avg_a_64=sliding_average(cf, 64)
b=avg_a_64.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg64.png')

avg_a_128=sliding_average(cf, 128)
b=avg_a_128.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg128.png')

avg_a_256=sliding_average(cf, 256)
b=avg_a_128.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg256.png')


print("context-free grammar:")
print("0-2: ", overlap(normalize(cf),normalize(avg_a_2)))
print("2-4: ", overlap(normalize(avg_a_2),normalize(avg_a_4)))
print("4-8: ", overlap(normalize(avg_a_4),normalize(avg_a_8)))
print("8-16: ", overlap(normalize(avg_a_8),normalize(avg_a_16)))
print("16-32: ", overlap(normalize(avg_a_16),normalize(avg_a_32)))
print("32-64: ", overlap(normalize(avg_a_32),normalize(avg_a_64)))
print("64-128: ", overlap(normalize(avg_a_64),normalize(avg_a_128)))
print("128-256: ", overlap(normalize(avg_a_128),normalize(avg_a_256)))

'''
#non-normalized
print("0-2: ", overlap(cf,avg_a_2))
print("2-4: ", overlap(avg_a_2,avg_a_4))
print("4-8: ", overlap(avg_a_4,avg_a_8))
print("8-16: ", overlap(avg_a_8,avg_a_16))
print("16-32: ", overlap(avg_a_16,avg_a_32))
print("32-64: ", overlap(avg_a_32,avg_a_64))
print("64-128: ", overlap(avg_a_64,avg_a_128))
print("128-256: ", overlap(avg_a_128,avg_a_256))
'''
a=cs

cs_normalized =cs / cs.sum(axis=(0,1))

r=cs.shape[0]*cs.shape[1]
b=cs.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_original.png')


avg_a_2=sliding_average(cs, 2)
b=avg_a_2.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_avg2.png')

avg_a_4=sliding_average(cs, 4)
b=avg_a_4.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_avg4.png')

avg_a_8=sliding_average(cs, 8)
b=avg_a_8.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_avg8.png')

avg_a_16=sliding_average(cs, 16)
b=avg_a_16.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_avg16.png')

avg_a_32=sliding_average(cs, 32)
b=avg_a_32.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-sensitive_avg32.png')

avg_a_64=sliding_average(cs, 64)
b=avg_a_64.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg64.png')

avg_a_128=sliding_average(cs, 128)
b=avg_a_128.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg128.png')

avg_a_256=sliding_average(cs, 256)
b=avg_a_128.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('context-free_avg256.png')

print("context-sensitive grammar:")

print("0-2: ", overlap(normalize(cs),normalize(avg_a_2)))
print("2-4: ", overlap(normalize(avg_a_2),normalize(avg_a_4)))
print("4-8: ", overlap(normalize(avg_a_4),normalize(avg_a_8)))
print("8-16: ", overlap(normalize(avg_a_8),normalize(avg_a_16)))
print("16-32: ", overlap(normalize(avg_a_16),normalize(avg_a_32)))
print("32-64: ", overlap(normalize(avg_a_32),normalize(avg_a_64)))
print("64-128: ", overlap(normalize(avg_a_64),normalize(avg_a_128)))
print("128-256: ", overlap(normalize(avg_a_128),normalize(avg_a_256)))



################DEBUG#########################
rand = np.random.choice([x for x in [-1,1]],(256,256))

'''
print('!!!!', r)
print(rand.shape)
b=rand.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_original.png')

'''

avg_a_2=sliding_average(rand, 2)

#b=avg_a_2.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
#data = Image.fromarray((b * 255).astype(np.uint8))
#data.save('random_avg2.png')


avg_a_4=sliding_average(rand, 4)
'''b=avg_a_4.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg4.png')'''

avg_a_8=sliding_average(rand, 8)
'''b=avg_a_8.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg8.png')'''


avg_a_16=sliding_average(rand, 16)
'''
b=avg_a_16.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg16.png')'''


avg_a_32=sliding_average(rand, 32)
'''b=avg_a_32.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg32.png')'''


avg_a_64=sliding_average(rand, 64)
'''b=avg_a_64.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg64.png')'''


avg_a_128=sliding_average(rand, 128)
'''b=avg_a_128.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg128.png')'''


avg_a_256=sliding_average(rand, 256)
'''b=avg_a_256.reshape(r,)[0:int(math.sqrt(r))**2].reshape(int(math.sqrt(r)),int(math.sqrt(r)))
data = Image.fromarray((b * 255).astype(np.uint8))
data.save('random_avg256.png')'''


print("test random matrix:")

print("0-2: ", overlap(normalize(rand),normalize(avg_a_2)))
print("2-4: ", overlap(normalize(avg_a_2),normalize(avg_a_4)))
print("4-8: ", overlap(normalize(avg_a_4),normalize(avg_a_8)))
print("8-16: ", overlap(normalize(avg_a_8),normalize(avg_a_16)))
print("16-32: ", overlap(normalize(avg_a_16),normalize(avg_a_32)))
print("32-64: ", overlap(normalize(avg_a_32),normalize(avg_a_64)))
print("64-128: ", overlap(normalize(avg_a_64),normalize(avg_a_128)))
print("128-256: ", overlap(normalize(avg_a_128),normalize(avg_a_256)))

'''
print("0-2: ", overlap(cf,avg_a_2))
print("2-4: ", overlap(avg_a_2,avg_a_4))
print("4-8: ", overlap(avg_a_4,avg_a_8))
print("8-16: ", overlap(avg_a_8,avg_a_16))
print("16-32: ", overlap(avg_a_16,avg_a_32))
print("32-64: ", overlap(avg_a_32,avg_a_64))
print("64-128: ", overlap(avg_a_64,avg_a_128))
print("128-256: ", overlap(avg_a_128,avg_a_256))
'''

'''
print("context-sensitive grammar:")
print("0-2: ", overlap(cs_normalized,cs_normalized)-overlap(cs_normalized,avg_a_2 / avg_a_2.sum(axis=(0,1))))
print("2-4: ", overlap(avg_a_2 / avg_a_2.sum(axis=(0,1)),avg_a_2 / avg_a_2.sum(axis=(0,1)))-overlap(avg_a_2 / avg_a_2.sum(axis=(0,1)),avg_a_2 / avg_a_4.sum(axis=(0,1))))
print("4-8: ", overlap(avg_a_4 / avg_a_4.sum(axis=(0,1)),avg_a_4 / avg_a_4.sum(axis=(0,1)))-overlap(avg_a_4 / avg_a_4.sum(axis=(0,1)),avg_a_8 / avg_a_8.sum(axis=(0,1))))
print("8-16: ", overlap(avg_a_8 / avg_a_8.sum(axis=(0,1)),avg_a_8 / avg_a_8.sum(axis=(0,1)))-overlap(avg_a_8 / avg_a_8.sum(axis=(0,1)),avg_a_16 / avg_a_16.sum(axis=(0,1))))
print("16-32: ", overlap(avg_a_16 / avg_a_16.sum(axis=(0,1)),avg_a_16 / avg_a_16.sum(axis=(0,1)))-overlap(avg_a_16 / avg_a_16.sum(axis=(0,1)),avg_a_32 / avg_a_32.sum(axis=(0,1))))
'''

'''

#a=np.random.rand(256,256)


#create sequences

#context free vs context senstitive

'''

'''
#mirror language
cf=np.asarray([[0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 2, 3, 3, 2, 1, 0],
[4, 1, 1, 1, 1, 1, 1, 4],
[2, 2, 3, 1, 1, 3, 2, 2],
[0, 3, 2, 0, 0, 3, 2, 0],
[2, 2, 3, 3, 3, 3, 2, 2],
[0, 0, 0, 0, 0, 0, 0, 0],
[3, 2, 1, 1, 1, 1, 2, 3]])



#copy language
cs=np.asarray([[0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 2, 3, 0, 1, 2, 3],
[4, 1, 1, 1, 4, 1, 1, 1],
[2, 2, 3, 1, 2, 2, 3, 1],
[0, 3, 2, 0, 0, 3, 2, 0],
[2, 2, 3, 3, 2, 2, 3, 3],
[0, 0, 0, 0, 0, 0, 0, 0],
[3, 2, 1, 1, 3, 2, 1, 1]])


r=np.asarray([[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 2, 2, 2],
[0, 1, 1, 1, 1, 1, 2, 2],
[0, 0, 1, 2, 2, 2, 2, 3],
[1, 1, 1, 1, 1, 1, 1, 3],
[2, 2, 3, 3, 3, 3, 3, 3],
[0, 2, 2, 2, 3, 3, 3, 3],
[0, 1, 2, 3, 3, 3, 3, 3]])

r=r / np.linalg.norm(r)

#normalize first image (all image weight is 1)

#100x100 squares
#10
'''
'''
a=r

avg_a_2=sliding_average(a, 2)
avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 16)
avg_a_16=sliding_average(a, 16)

print("regular grammar:")
print("0-2: ", str(overlap(a,avg_a_2)))
print("2-4: ", str(overlap(avg_a_2,avg_a_4)))
print("4-8: ", str(overlap(avg_a_4,avg_a_8)))
print("8-16: ", str(overlap(avg_a_8,avg_a_16)))

'''

'''
context-free mirror language	
(i.e. the set of strings xy over a given Σ such that y is the mirror image of x)	

context-senstitive
copy language(i.e. the set of strings xx over a given Σ such that x is an arbitrary string of symbols from Σ)


context-free:
a^nb^mc^md^n

context-sensitive
a^nb^mc^nd^m

regular
a^nb^m
non-regular
a^nb^n

'''
