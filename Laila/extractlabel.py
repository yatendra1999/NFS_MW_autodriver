import re
import numpy as np
def extract(name):
    pattern1='\d+\['
    pattern2='\].+'
    replace=''
    new_name=re.sub(pattern1,replace,name)
    new_name=re.sub(pattern2,replace,new_name)
    new_name=list(map(int, new_name.split(',')))
    return np.asarray(new_name)