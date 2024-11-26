# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import re
import shutil

files = glob.glob('ntu60_interaction/*.skeleton')


for ind, name in enumerate(files):
    name = name.replace('\\', '/')
    name = name.split('/')[1].replace('.skeleton', '')
    info = re.findall(r"\d+\.?\d*", name)
    
    setup = str(int(info[0]))
    camera = str(int(info[1]))
    performer = str(int(info[2]))
    replication = str(int(info[3]))
    label = str(int(info[4]))
    
    
    with open('statistics/setup.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(setup)
        else:
            f.write(setup + '\n')
        
    with open('statistics/camera.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(camera)
        else:
            f.write(camera + '\n')
        
    with open('statistics/performer.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(performer)
        else:
            f.write(performer + '\n')
        
    with open('statistics/replication.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(replication)
        else:
            f.write(replication + '\n')
        
    with open('statistics/label.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(label)
        else:
            f.write(label + '\n')
        
            
    with open('statistics/skes_available_name.txt', 'a') as f:
        if ind == len(files) - 1: 
            f.write(name)
        else:
            f.write(name + '\n')