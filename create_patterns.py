import numpy as np

from PIL import Image
import os

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


def overlap(m1, m2):
	m=m1*m2
	return m.sum(axis=(0,1))
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


	

n=6

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


cf=cf / np.linalg.norm(cf)
cs=cs / np.linalg.norm(cs)


a=cf

avg_a_2=sliding_average(a, 2)
avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 8)
avg_a_16=sliding_average(a, 16)
avg_a_32=sliding_average(a, 32)

print("context-free grammar:")
print("0-2: ", overlap(a,a)-overlap(a,avg_a_2))
print("2-4: ", overlap(avg_a_2,avg_a_2)-overlap(avg_a_2,avg_a_4))
print("4-8: ", overlap(avg_a_4,avg_a_4)-overlap(avg_a_4,avg_a_8))
print("8-16: ", overlap(avg_a_8,avg_a_8)-overlap(avg_a_8,avg_a_16))
print("16-32: ", overlap(avg_a_16,avg_a_16)-overlap(avg_a_16,avg_a_32))

a=cs

avg_a_2=sliding_average(a, 2)
avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 16)
avg_a_16=sliding_average(a, 16)

print("context-sensitive grammar:")
print("0-2: ", overlap(a,a)-overlap(a,avg_a_2))
print("2-4: ", overlap(avg_a_2,avg_a_2)-overlap(avg_a_2,avg_a_4))
print("4-8: ", overlap(avg_a_4,avg_a_4)-overlap(avg_a_4,avg_a_8))
print("8-16: ", overlap(avg_a_8,avg_a_8)-overlap(avg_a_8,avg_a_16))
print("16-32: ", overlap(avg_a_16,avg_a_16)-overlap(avg_a_16,avg_a_32))



#a=np.random.rand(256,256)


#create sequences

#context free vs context senstitive

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

