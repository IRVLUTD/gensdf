#!/usr/bin/env python
import sys, os
from pathlib import Path


if __name__ == '__main__':

    root = '/home/yuxiang/Datasets/acronym/'
    shapenet = '/home/yuxiang/Datasets/ShapeNetSem/models'    
    
    grasp_path = Path(os.path.join(root, 'grasps'))
    print(grasp_path)
    grasp_files = sorted(list(grasp_path.glob('*.h5')))
    print(len(grasp_files))
    
    for name in grasp_files:
        strings = os.path.basename(name).split('_')
        cls = strings[0]
        base_name = strings[1]
        print(cls)
        
        folder = os.path.join(root, 'models', cls, base_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # copy model files
        src_file = os.path.join(shapenet, base_name + '.obj')
        print(src_file)
        
        dst_file = os.path.join(folder, 'model.obj')
                        
        command = 'cp ' + src_file + ' ' + dst_file
        print(command)
        os.system(command)
