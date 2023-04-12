#!/usr/bin/env python
import sys, os
from pathlib import Path


if __name__ == '__main__':

    root = '/home/yuxiang/Datasets/acronym/'
    exe = '/home/yuxiang/Projects/Manifold/build'
    
    model_path = Path(os.path.join(root, 'models'))
    folders = sorted(os.listdir(model_path))
        
    for folder in folders:
    
        meshes = sorted(os.listdir(os.path.join(model_path, folder)))
        
        for mesh in meshes:
            src_file = os.path.join(model_path, folder, mesh, 'model.obj')
            dst_file = os.path.join(model_path, folder, mesh, 'temp.watertight.obj')
            print(src_file)
          
            command = exe + '/manifold ' + src_file + ' ' + dst_file + ' -s'
            print(command)
            os.system(command)

            src_simple_file = os.path.join(model_path, folder, mesh, 'model_simplified.obj')
            command =  exe + '/simplify -i ' + dst_file + ' -o ' + src_simple_file + ' -m -r 0.02'
            print(command)
            os.system(command)
            
            sys.exit(1)
