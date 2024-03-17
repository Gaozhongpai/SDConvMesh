from tqdm import tqdm
import numpy as np
import os, argparse
from psbody.mesh import Mesh
import torch
import pickle

mesh = Mesh(filename="/media/pai/Disk/data/monoData/template.obj")
mean = torch.load('/media/pai/Disk/data/monoData/Processed/sliced/mean.tch')
mesh.v = mean.numpy()
mesh.write_obj('weights/mean.obj')
# with open(os.path.join('/media/pai/Disk/data/monoData/COMA_downsample','downsampling_matrices.pkl'), 'rb') as fp:
#     #downsampling_matrices = pickle.load(fp,encoding = 'latin1')
#     downsampling_matrices = pickle.load(fp)
# M_verts_faces = downsampling_matrices['M_verts_faces']
# for i in range(len(M_verts_faces)):
#     mesh = Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1])
#     mesh.write_obj('weights/template{}.obj'.format(i))
