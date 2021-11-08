# from sklearn.decomposition import PCA
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import os
from matplotlib import cm
import trimesh

def _cmap2rgb(cmap, step):
    return getattr(cm, cmap)(step, bytes=True)

# template = Mesh(filename="/home/user/3dfaceRe/coma3/Neural3DMMdata/template.obj")dfaustData
template = trimesh.load_mesh("/media/pai/data/Neural3DMMdata/template.obj", process=False)

root_dir = '/media/pai/data/Neural3DMMdata/Processed/sliced'
path = np.load(os.path.join(root_dir, "paths_test.npy"))
std = torch.load(os.path.join(root_dir, "std.tch"))
mean = torch.load(os.path.join(root_dir, "mean.tch"))
# index = 455
# template.v = torch.load(os.path.join(root_dir, "points_test", path[index]+".tch")).numpy()
# template.show()
# predition = torch.load("coma_predictions_"+"coma"+".tch")[index].numpy()
# template.show()
# predition = torch.load("coma_predictions_"+"pai"+".tch")[index].numpy()
# template.show()
# predition = torch.load("coma_predictions_"+"pca"+".tch")[index].numpy()
# template.show()
# predition = torch.load("coma_predictions_"+"spiral"+".tch")[index].numpy()
# template.show()

if not os.path.exists("COMA_X_test.tch"):
    X_test = []
    for i, x_name in enumerate(path):
        x = torch.load(os.path.join(root_dir, "points_test", x_name+".tch"))
        X_test.append(x)
        if i % 100 == 0:
            print("we are at {}".format(i))
    X_test = torch.stack(X_test)*1000
    torch.save(X_test, "COMA_X_test.tch")
else: 
    X_test = torch.load("COMA_X_test.tch")

value = _cmap2rgb("jet", 128)
errors_methods = {}
maxerror = 0
methods = ["COMA", "Spiral", "LSA-small", "SDConv", "HSDConv"]
if not os.path.exists("COMA_errors_methods.tch"):
    for method in methods:
        preditions = torch.load("COMA_AUTO_"+method+".tch")
        if preditions.shape[1] != X_test.shape[1]:
            preditions = preditions[:, :-1]
        if method is not "pca":
            preditions = (preditions*std + mean)*1000
        Errors = torch.sqrt(torch.sum((preditions - X_test)**2,dim=2))
        maxerror = torch.max(Errors) if torch.max(Errors) > maxerror else maxerror
        errors_methods[method] = Errors
    torch.save(errors_methods, "COMA_errors_methods.tch")
else:
    errors_methods = torch.load("COMA_errors_methods.tch")

_, index = torch.topk(torch.mean(errors_methods['COMA'] - errors_methods['HSDConv'], dim=1), 20)
maxerror = 2
# method = "SDConvL"
# index = index[10:]
X_test = X_test.numpy()
for key in errors_methods:
    if not os.path.exists(os.path.join("./", key)):
        os.mkdir(os.path.join("./", key))
    for i in index:
        template.vertices = X_test[i]
        for j in range(errors_methods[key].shape[1]):
            color = _cmap2rgb("jet", int(errors_methods[key][i, j]/maxerror*255))
            template.visual.vertex_colors[j] = np.asarray(color)
        print("We are at {}".format(i))
        template.export(os.path.join("./", key, path[i]+".ply"), 'ply')



