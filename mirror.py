'''def generate_mirror_strings(length: int):
    if length == 0:
        yield ''
    else:
        for ch in '0123':
            for mirror_string in generate_mirror_strings(length - 1):
                yield ch + mirror_string + ch


def generate_strings(length: int):
    if length == 0:
        yield []
    else:
        for ch in '0123':
            for string in generate_strings(length - 1):
                yield ch + string
'''
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

mirror_strings=[]
copy_strings=[]
for half_string in generate_strings_array(n):
    copy_string=half_string+half_string
    hs=half_string.copy()
    hs.reverse()
    mirror_string=half_string+hs
    copy_strings.append(copy_string)
    mirror_strings.append(mirror_string)
    #print(str(half_string))
    #print(str(copy_string))
    #print(str(mirror_string))



