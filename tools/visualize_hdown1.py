from tqdm import tqdm
import numpy as np
import os, argparse
from psbody.mesh import Mesh
import torch
import pickle
from sklearn.manifold import TSNE
from matplotlib import cm
import trimesh

def _cmap2rgb(cmap, step):
    return getattr(cm, cmap)(step, bytes=True)

name = ['hdown', 'hdown_origin']
inx = 1 # if 'ndown' else 1

D0 = torch.load('weights/{}1.tch'.format(name[inx]))[:-1, :-1]

# X_embedded = TSNE(n_components=1, learning_rate='auto', init='random').fit_transform(torch.transpose(D0, 0, 1).numpy())
# X_embedded = (X_embedded - X_embedded.min())/(X_embedded.max() - X_embedded.min())

X_embedded = (D0 != 0)
X_embedded = X_embedded * torch.tensor(range(X_embedded.shape[0])).unsqueeze(1)

# index = [939, 1530, 664, 76, 102, 1430, 1360, 1392, 755, 519, 485, 566,
#          1575, 890, 139, 1311, 442, 1720, 298, 1665, 822] # 1
index = [245, 381, 18, 28, 365, 346, 265, 334, 135, 129, 145, 
         262, 107, 45, 394, 119, 309, 226, 415, 212] # 2


template = trimesh.load_mesh("/home/pai/code/PaiConvMesh/weights/hdown_origin2.ply", process=False)
if not os.path.exists(os.path.join("./weights","hdown2_refer.ply")):
    for j in range(X_embedded.shape[0]):
        if j not in index:
            template.visual.vertex_colors[j] = np.asarray([128, 128, 128, 255])
    template.export(os.path.join("./weights","hdown2_refer.ply"), 'ply')

X_Zeros = torch.zeros_like(X_embedded)

X_Zeros[index] = X_embedded[index]

# template_next = trimesh.load_mesh("/media/pai/Disk/data/dfaustData/template_color.ply", process=False)
template_next = trimesh.load_mesh("/home/pai/code/PaiConvMesh/weights/hdown_origin1.ply", process=False)
for j in range(X_Zeros.shape[1]):
    if X_Zeros[:, j].sum() != 0:
        color = np.zeros(4)
        k = 0
        for i in range(X_Zeros.shape[0]):
            if X_Zeros[i, j] !=0:
                k = k + 1
                color = color + template.visual.vertex_colors[X_Zeros[i, j]]
        color = color / k
        template_next.visual.vertex_colors[j] = color
    else:
        template_next.visual.vertex_colors[j] = np.asarray([128, 128, 128, 255])
template_next.export(os.path.join("./weights",name[inx]+"_1.ply"), 'ply')
