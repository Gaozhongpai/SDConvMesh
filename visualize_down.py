from tqdm import tqdm
import numpy as np
import os, argparse
from psbody.mesh import Mesh
import torch
import pickle

mesh = Mesh(filename="/media/pai/Disk/data/monoData/template.obj")
with open(os.path.join('/media/pai/Disk/data/dfaustData/COMA_downsample','downsampling_matrices.pkl'), 'rb') as fp:
    #downsampling_matrices = pickle.load(fp,encoding = 'latin1')
    downsampling_matrices = pickle.load(fp)
M_verts_faces = downsampling_matrices['M_verts_faces']
name = ['hdown', 'hdown_origin']
i = 0 # if 'ndown' else 1

v0 = torch.from_numpy(mesh.v).float()
c0 = torch.from_numpy(mesh.vc).float()
# v0 = torch.load('/media/pai/Disk/data/Neural3DMMdata/Processed/sliced/mean.tch')
# mesh = Mesh(v=v0.numpy(), f=M_verts_faces[0][1])
# mesh.write_obj('weights/{}0.obj'.format(name[i]))

D0 = torch.load('weights/{}0.tch'.format(name[i]))[:-1, :-1]
v1 = torch.matmul(D0, v0)
c1 = torch.matmul(D0, c0)
mesh = Mesh(v=v1.numpy(), vc=c1.numpy(), f=M_verts_faces[1][1])
# mesh.write_obj('weights/{}1.obj'.format(name[i]))
mesh.write_ply('weights/{}1.ply'.format(name[i]))

D1 = torch.load('weights/{}1.tch'.format(name[i]))[:-1, :-1]
v2 = torch.matmul(D1, v1)
c2 = torch.matmul(D1, c1)
mesh = Mesh(v=v2.numpy(), vc=c2.numpy(), f=M_verts_faces[2][1])
mesh.write_ply('weights/{}2.ply'.format(name[i]))

D2 = torch.load('weights/{}2.tch'.format(name[i]))[:-1, :-1]
v3 = torch.matmul(D2, v2)
c3 = torch.matmul(D2, c2)
mesh = Mesh(v=v3.numpy(), vc=c3.numpy(), f=M_verts_faces[3][1])
mesh.write_ply('weights/{}3.ply'.format(name[i]))

D3 = torch.load('weights/{}3.tch'.format(name[i]))[:-1, :-1]
v4 = torch.matmul(D3, v3)
c4 = torch.matmul(D3, c3)
mesh = Mesh(v=v4.numpy(), vc=c4.numpy(), f=M_verts_faces[4][1])
mesh.write_ply('weights/{}4.ply'.format(name[i]))