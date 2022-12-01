import numpy as np


def sliding_average(a, span):
	c=np.random.rand(256,256)
	for i in range(0, len(a), span):
		for j in range(0, len(a), span):
			b=a[i:i+span, j:j+span]
			m=np.mean(b)
			c[i:i+span, j:j+span]=m
	return c




a=np.random.rand(256,256)

avg_a_2=sliding_average(a, 2)
avg_a_4=sliding_average(a, 4)
avg_a_8=sliding_average(a, 16)
avg_a_16=sliding_average(a, 16)



