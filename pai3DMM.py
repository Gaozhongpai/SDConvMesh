# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import json
import os
import copy
import pickle

import mesh_sampling
import trimesh
#from psbody.mesh import Mesh
from shape_data import ShapeData

from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader

from utils import get_adj, sparse_mx_to_torch_sparse_tensor
from models import PaiAutoencoder
from train_funcs import train_autoencoder_dataloader
from test_funcs import test_autoencoder_dataloader
import scipy.sparse as sp
from device import device
import torch
from tensorboardX import SummaryWriter
from utils import IOStream

from sklearn.metrics.pairwise import euclidean_distances
meshpackage = 'trimesh' # 'mpi-mesh', trimesh'

root_dir = 'dataset/COMA-dataset'   ## 'COMA-dataset' or 'DFAUST-dataset' or 'MANO-dataset''
is_hierarchical = True              ## 'True' or 'False' for learnable up/down sampling
is_same_param = False                   ## '0', '1', '2' where '1' for increaes channel and '2' for increase base 
is_old_filter = False               ## 'False' or 'True' to use different spectral filter
mode = 'test'                       ## 'test' or 'train' to train or test the models

generative_model = 'SDConvFinal' # method name
if not is_old_filter:
    generative_model = generative_model + '-x'
if is_hierarchical:
    generative_model = 'H' + generative_model
if is_same_param:
    generative_model = generative_model + '-param'
    
name = 'sliced'
GPU = True
device_idx = 0
torch.cuda.get_device_name(device_idx)


#%%
args = {}


downsample_method = 'COMA_downsample' # choose'COMA_downsample' or 'meshlab_downsample'
# below are the arguments for the DFAUST run
reference_mesh_file = os.path.join(root_dir, 'template.obj')
downsample_directory = os.path.join(root_dir, downsample_method)
ds_factors = [4, 4, 4, 4]
kernal_size = [9, 9, 9, 9, 9]
step_sizes = [2, 2, 1, 1, 1]

## # monoData
# filter_sizes_enc = [3, 16, 32, 64]
# filter_sizes_dec = [64, 32, 32, 16, 3]

filter_sizes_enc = [3, 16, 32, 64, 128]
filter_sizes_dec = [128, 64, 32, 32, 16, 3]
if is_same_param: 
    if "COMA" in root_dir: ## COMA
        filter_sizes_enc = [3, 32, 45, 64, 128]
        filter_sizes_dec = [128, 80, 48, 32, 32, 3]
    elif "DFAUST" in root_dir: ## dfaustData
        filter_sizes_enc = [3, 32, 42, 80, 128]
        filter_sizes_dec = [128, 80, 64, 40, 32, 3]

base_size = 32
# if is_same_param == 2: 
#     if "COMA" in root_dir: ## COMA
#         base_size = 78
#     elif "DFAUST" in root_dir: ## dfaustData
#         base_size = 112

args = {'generative_model': generative_model,
        'name': name, 'data': os.path.join(root_dir, 'Processed',name),
        'results_folder':  os.path.join(root_dir,'results/'+ generative_model),
        'reference_mesh_file':reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed':2, 'loss':'l1',
        'batch_size':32, 'num_epochs':300, 'eval_frequency':200, 'num_workers': 8,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz':32,
        'ds_factors': ds_factors, 'step_sizes' : step_sizes,
        
        'lr':1e-3, 'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99,'decay_steps':1,
        'resume': False,

        'mode': mode, 'shuffle': True, 'nVal': 100, 'normalization': True}

args['results_folder'] = os.path.join(args['results_folder'],'latent_'+str(args['nz']))

if not os.path.exists(os.path.join(args['results_folder'])):
    os.makedirs(os.path.join(args['results_folder']))

summary_path = os.path.join(args['results_folder'],'summaries',args['name'])
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

checkpoint_path = os.path.join(args['results_folder'],'checkpoints', args['name'])
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

samples_path = os.path.join(args['results_folder'],'samples', args['name'])
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

prediction_path = os.path.join(args['results_folder'],'predictions', args['name'])
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)

if not os.path.exists(downsample_directory):
    os.makedirs(downsample_directory)


#%%
np.random.seed(args['seed'])
print("Loading data .. ")
if not os.path.exists(args['data']+'/mean.tch') or not os.path.exists(args['data']+'/std.tch'):
    shapedata =  ShapeData(nVal=args['nVal'],
                          train_file=args['data']+'/train.npy',
                          test_file=args['data']+'/test.npy',
                          reference_mesh_file=args['reference_mesh_file'],
                          normalization = args['normalization'],
                          meshpackage = meshpackage, load_flag = True)
    torch.save(args['data']+'/mean.tch', shapedata.mean)
    torch.save(args['data']+'/std.tch', shapedata.std)
else:
    shapedata = ShapeData(nVal=args['nVal'],
                         train_file=args['data']+'/train.npy',
                         test_file=args['data']+'/test.npy',
                         reference_mesh_file=args['reference_mesh_file'],
                         normalization = args['normalization'],
                         meshpackage = meshpackage, load_flag = False)
    shapedata.mean = torch.load(args['data']+'/mean.tch')
    shapedata.std = torch.load(args['data']+'/std.tch')
    shapedata.n_vertex = shapedata.mean.shape[0]
    shapedata.n_features = shapedata.mean.shape[1]

if not os.path.exists(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl')):
    if shapedata.meshpackage == 'trimesh':
        raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
    print("Generating Transform Matrices ..")
    if downsample_method == 'COMA_downsample':
        M,A,D,U,F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, args['ds_factors'])
    with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'wb') as fp:
        M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
        pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)
    
else:
    print("Loading Transform Matrices ..")
    with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'rb') as fp:
        #downsampling_matrices = pickle.load(fp,encoding = 'latin1')
        downsampling_matrices = pickle.load(fp)

    M_verts_faces = downsampling_matrices['M_verts_faces']
    if shapedata.meshpackage == 'mpi-mesh':
        M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
    elif shapedata.meshpackage == 'trimesh':
        M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process = False) for i in range(len(M_verts_faces))]
    A = downsampling_matrices['A']
    D = downsampling_matrices['D']
    U = downsampling_matrices['U']
    F = downsampling_matrices['F']

vertices = [torch.cat([torch.tensor(M_verts_faces[i][0], dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) for i in range(len(M_verts_faces))]

#%%
if shapedata.meshpackage == 'mpi-mesh':
    sizes = [x.v.shape[0] for x in M]
elif shapedata.meshpackage == 'trimesh':
    sizes = [x.vertices.shape[0] for x in M]
if not os.path.exists(os.path.join(args['downsample_directory'],'pai_matrices.pkl')):
    Adj = get_adj(A)
    bU = []
    bD = []
    for i in range(len(D)):
        d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
        u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
        d[0,:-1,:-1] = D[i].todense()
        u[0,:-1,:-1] = U[i].todense()
        d[0,-1,-1] = 1
        u[0,-1,-1] = 1
        bD.append(d)
        bU.append(u)
    bD = [sp.csr_matrix(s[0, ...]) for s in bD]
    bU = [sp.csr_matrix(s[0, ...]) for s in bU] 
    with open(os.path.join(args['downsample_directory'],'pai_matrices.pkl'), 'wb') as fp:
        pickle.dump([Adj, sizes, bD, bU], fp)
else: 
    print("Loading adj Matrices ..")
    with open(os.path.join(args['downsample_directory'],'pai_matrices.pkl'), 'rb') as fp:
        [Adj, sizes, bD, bU] = pickle.load(fp)

tD = [sparse_mx_to_torch_sparse_tensor(s) for s in bD]
tU = [sparse_mx_to_torch_sparse_tensor(s) for s in bU]

#%%
torch.manual_seed(args['seed'])
print(device)
io = IOStream(os.path.join(args['results_folder']) + '/run.log')
io.cprint(str(args))
#%%
# Building model, optimizer, and loss function

if args['mode'] == 'train': 
    dataset_train = autoencoder_dataset(
        root_dir=args['data'],
        points_dataset='train',
        shapedata=shapedata,
        normalization=args['normalization'])

    dataloader_train = DataLoader(
        dataset_train, batch_size=args['batch_size'],
        shuffle=args['shuffle'],
        num_workers = args['num_workers']
        )

    dataset_val = autoencoder_dataset(
        root_dir=args['data'],
        points_dataset='val',
        shapedata=shapedata,
        normalization=args['normalization'])

    dataloader_val = DataLoader(
        dataset_val, batch_size=args['batch_size'],
        shuffle=False,
        num_workers = args['num_workers'])
else:
    dataset_test = autoencoder_dataset(
        root_dir=args['data'],
        points_dataset='test',
        shapedata=shapedata,
        normalization=args['normalization'])

    dataloader_test = DataLoader(
        dataset_test, batch_size=args['batch_size'],
        shuffle=False,
        #num_workers = args['num_workers']
        )

model = PaiAutoencoder(filters_enc = args['filter_sizes_enc'],
                            filters_dec = args['filter_sizes_dec'],
                            latent_size=args['nz'],
                            sizes=sizes,
                            t_vertices=vertices,
                            num_neighbors=kernal_size,
                            x_neighbors=Adj,
                            D=tD, U=tU, 
                            is_hierarchical=is_hierarchical,
                            is_old_filter=is_old_filter,
                            base_size=base_size).to(device)
# model = torch.nn.DataParallel(model)

trainables_wo_index = [param for name, param in model.named_parameters()
                if param.requires_grad and 'adjweight' not in name] # and 'key' not in name and 'weight_prior' not in name
trainables_wt_index = [param for name, param in model.named_parameters()
                if param.requires_grad and 'adjweight' in name] # or 'key' in name or 'weight_prior' in name
optim = torch.optim.Adam([{'params': trainables_wo_index, 'weight_decay': args['regularization']},
                          {'params': trainables_wt_index}],
                          lr=args['lr'])
#optim = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['regularization'])
if args['scheduler']:
    scheduler=torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'],gamma=args['decay_rate'])
else:
    scheduler = None

if args['loss']=='l1':
    def loss_l1(outputs, targets):
        L = torch.abs(outputs - targets).mean()
        return L
    loss_fn = loss_l1


#%%
print(model)
params = sum(param.numel() for name, param in model.named_parameters() \
            if param.requires_grad and 't_vertices' not in name and 'attpool' not in name)
print("Total number of parameters is: {}".format(params))
io.cprint("Total number of parameters is: {}".format(params))
#%%
if args['mode'] == 'train':
    writer = SummaryWriter(summary_path)
    with open(os.path.join(args['results_folder'],'checkpoints', args['name'] +'_params.json'),'w') as fp:
        saveparams = copy.deepcopy(args)
        json.dump(saveparams, fp)

    if args['resume']:
        print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file'])))
        checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
        start_epoch = checkpoint_dict['epoch'] + 1 
        model_dict = model.state_dict()
        pretrained_dict = checkpoint_dict['autoencoder_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(pretrained_dict, strict=False)
        #model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
        optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        print('Resuming from epoch %s'%(str(start_epoch)))
    else:
        start_epoch = 0

    train_autoencoder_dataloader(dataloader_train, dataloader_val,
                        device, model, optim, loss_fn, io,
                        bsize = args['batch_size'],
                        start_epoch = start_epoch,
                        n_epochs = args['num_epochs'],
                        eval_freq = args['eval_frequency'],
                        scheduler = scheduler,
                        writer = writer,
                        save_recons=True,
                        shapedata=shapedata,
                        metadata_dir=checkpoint_path, samples_dir=samples_path,
                        checkpoint_path = args['checkpoint_file'])


#%%
if args['mode'] == 'test':
    print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar')))
    checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
    
    print('Current Epoch is {}.'.format(checkpoint_dict['epoch']))
    model_dict = model.state_dict()
    pretrained_dict = checkpoint_dict['autoencoder_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "U." not in k and "D." not in k}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(pretrained_dict, strict=False)
    #model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

    predictions, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test,
                                                                     shapedata, mm_constant = 1000)
    torch.save(predictions, os.path.join(prediction_path,'predictions.tch'))
    torch.save({'norm_l1_loss':norm_l1_loss, 'l2_loss':l2_loss}, os.path.join(prediction_path,'loss.tch'))

    io.cprint('autoencoder: normalized loss={}'.format(norm_l1_loss.item()))
    io.cprint('autoencoder: euclidean distance in mm={}'.format(l2_loss.item()))
