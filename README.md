

# Learning Spectral Dictionary for Local Representation of Mesh
![PaiNeural3DMM architecture](images/architecture.png "PaiNeural3DMM architecture")
This repository is the official implementation of my paper: "Learning Spectral Dictionary for Local Representation of Mesh"
# Project Abstract 
Learning mesh representation is important for many 3D tasks. Conventional convolution for regular data (i.e., images) cannot directly be applied to meshes since each vertex's neighbors are unordered. Previous methods use isotropic filters or predefined local coordinate systems or learning weighting matrices for each template vertex to overcome the irregularity. Learning weighting matrices to resample the vertex's neighbors into an implicit canonical order is the most effective way to capture the local structure of each vertex. However, learning weighting matrices for each vertex increases the model size linearly with the vertex number. Thus, large parameters are required for high-resolution 3D shapes, which is not favorable for many applications. In this paper, we learn spectral dictionary (i.e., bases) for the weighting matrices such that the model size is independent of the resolution of 3D shapes. The coefficients of the weighting matrix bases are learned from the spectral features of the template and its hierarchical levels in a weight-sharing manner. Furthermore, we introduce an adaptive sampling method that learns the hierarchical mapping matrices directly to improve the performance without increasing the model size at the inference stage. Comprehensive experiments demonstrate that our model produces state-of-the-art results with much smaller model size.

[IJCAI link](https://www.ijcai.org/proceedings/2021/95)

![Pai-Conv](images/pai-gcn.png "Pai-Conv operation")

![Results](images/complexity1.png "Results")

# Repository Requirements

This code was written in Pytorch 1.4. We use tensorboardX for the visualisation of the training metrics. We recommend setting up a virtual environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html). To install Pytorch in a conda environment, simply run 

```
$ conda install pytorch torchvision -c pytorch
```

Then the rest of the requirements can be installed with 

```
$ pip install -r requirements.txt
```


# Data Organization

Here are the pre-processed datasets on Google Drive: [DFAUST-dataset.zip](https://drive.google.com/file/d/14UZq9JkDqjLLBiqHkSoIBZpkW6PQ_Xbi/view?usp=sharing) and [COMA-dataset.zip](https://drive.google.com/file/d/1LNhYee-h5_m1RVzguZvT2oPUsJAK28ac/view?usp=sharing). 

Here are the trained models on Google Drive [DFAUST-Models.zip] (https://drive.google.com/file/d/1Eq93ZX0uewJZBHuPdNeFmgCm5dl7WjLm/view?usp=sharing) and [COMA-Models.zip](https://drive.google.com/file/d/185hIebXxBDvDezteXDzXfZCdQRODo_Ck/view?usp=sharing). Please put the models in the folder as the structure below. 

### Preprocessing for your custom dataset:

In order to use a pytorch dataloader for training and testing, we split the data into seperate files by:

```
$ python data_generation.py --root_dir=/path/to/data_root_dir --dataset=DFAUST --num_valid=100
```

The following is the organization of the dataset directories expected by the code:


* data **root_dir**/
  * **dataset** name/ (eg DFAUST-dataset)
    * COMA_downsample
      * downsampling_matrices.pkl (created by the code the first time you run it)
      * pai_matrices.pkl
    * Processed/
      * sliced
        * train.npy (number_meshes, number_vertices, 3) (no Faces because they all share topology)
        * test.npy 
        * points_train/ (created by data_generation.py)
        * points_val/ (created by data_generation.py)
        * points_test/ (created by data_generation.py)
        * paths_train.npy (created by data_generation.py)
        * paths_val.npy (created by data_generation.py)
        * paths_test.npy (created by data_generation.py)
        * mean.tch
        * std.tch
    * results
      * HSDConvFinal-x (hierarchical SDConv result)
        * checkpoints
        * run.log
      * SDConvFinal-x (SDConv result)
        * checkpoints
        * run.log
    * template.obj




#### Training and Testing

```
args['mode'] = 'train' or 'test'

python pai3DMM.py
```

#### Some important notes:
* The code has compatibility with both _mpi-mesh_ and _trimesh_ packages (it can be chosen by setting the _meshpackage_ variable pai3DMM.py).




#### Acknowlegements:

The structure of this codebase is borrowed from [Neural3DMM](https://github.com/gbouritsas/Neural3DMM).

# Cite

Please consider citing our work if you find it useful:

```
@inproceedings{ijcai2021-95,
  title     = {Learning Spectral Dictionary for Local Representation of Mesh},
  author    = {Gao, Zhongpai and Yan, Junchi and Zhai, Guangtao and Yang, Xiaokang},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {685--692},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/95},
  url       = {https://doi.org/10.24963/ijcai.2021/95},
}
```