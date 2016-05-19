import re
import numpy as np


def get_input():
    print('Input string for parsing (press enter to exit):')
    s = '\n'.join(iter(input, ''))

    if s == '':
        return None

    prog = re.compile('[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?')
    final_list = [float(x) for x in re.findall(prog, s)]
    print(final_list)
    print()
    return final_list

master_list = []
while True:
    res = get_input()
    if res is None:
        break
    else:
        master_list.append(res)

a = np.array(master_list)
output = np.mean(a, axis=0)
print(output.tolist())
