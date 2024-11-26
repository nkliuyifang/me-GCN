# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:46:16 2022

@author: nkliu
"""

import glob
import re
import shutil


drop_file = 'statistics/NTU_RGBD120_samples_with_missing_skeletons.txt'
f = open(drop_file)
drop = f.readlines()
f.close()
action_ind = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

files = glob.glob('nturgb+d_skeletons/*.skeleton')

for name in files:
    name = name.replace('\\', '/')
    
    if name.split('/')[1].replace('.skeleton', '') + '\n' in drop:
        continue
    
    info = re.findall(r"\d+\.?\d*", name)
    action = int(info[4][0:-1])
    if action in action_ind:
        old_path = name
        new_path = name.replace('nturgb+d_skeletons/', 'ntu120_interaction/')
        shutil.move(old_path, new_path)
   