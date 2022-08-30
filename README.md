
## Few-shot learning for echocardiogram

This is the implementation of my thesis. Implemented on Python 3.7 and Pytorch 1.5.1.

## Requirements

- Python 3.7
- PyTorch 1.5.1
- cuda 10.1
- tensorboard 1.14

Conda environment settings:
```bash
conda create -n segmentation python=3.7
conda activate segmentation

conda install pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow
pip install tensorboardX
```

Create a directory '../Datasets_HSN' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                  # current (project) directory
    ├── common/             # (dir.) helper functions
    ├── data_loader/        # (dir.) dataloaders for dataset
    ├── model/              # (dir.) implementation of proposed model 
    ├── README.md           # intstruction for reproduction
    ├── train.py            # code for training 
    └── test.py             # code for testing
    


## Training
> ### 1. PASCAL-5<sup>i</sup>
> ```bash
> python train.py --backbone {vgg16, resnet50, resnet101} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark pascal
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 2 days until convergence (trained with four 2080 Ti GPUs).


> ### 2. COCO-20<sup>i</sup>
> ```bash
> python train.py --backbone {resnet50, resnet101} 
>                 --fold {0, 1, 2, 3} 
>                 --benchmark coco 
>                 --lr 1e-3
>                 --bsz 40
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 1 week until convergence (trained four Titan RTX GPUs).

> ### 3. FSS-1000
> ```bash
> python train.py --backbone {vgg16, resnet50, resnet101} 
>                 --benchmark fss 
>                 --lr 1e-3
>                 --bsz 20
>                 --logpath "your_experiment_name"
> ```
> * Training takes approx. 3 days until convergence (trained with four 2080 Ti GPUs).

> ### Data preparation
> ``` bash
> python train.py




> ### Babysitting training:
> Use tensorboard to babysit training progress:
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 



## Testing

> ### 1. PASCAL-5<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1z4KgjgOu--k6YuIj3qWrGg264GRcMis2?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50, resnet101} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```


> ### 2. COCO-20<sup>i</sup>
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1WpwmCQzxTWhJD5aLQhsgJASaoxxqmFUk?usp=sharing)].
> ```bash
> python test.py --backbone {resnet50, resnet101} 
>                --fold {0, 1, 2, 3} 
>                --benchmark coco 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```

> ### 3. FSS-1000
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/1JOaaJknGwsrSEPoLF3x6_lDiy4XfAe99?usp=sharing)].
> ```bash
> python test.py --backbone {vgg16, resnet50, resnet101} 
>                --benchmark fss 
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```

> ### 4. Evaluation *without support feature masking* on PASCAL-5<sup>i</sup>
> * To reproduce the results in Tab.1 of our main paper, **COMMENT OUT line 51 in hsnet.py**: support_feats = self.mask_feature(support_feats, support_mask.clone())
> 
> Pretrained models with tensorboard logs are available on our [[Google Drive](https://drive.google.com/drive/folders/18YWMCePIrza194pZvVMqQBuYqhwBmJwd?usp=sharing)].
> ```bash
> python test.py --backbone resnet101 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```


## Visualization

* To visualize mask predictions, add command line argument **--visualize**:
  (prediction results will be saved under vis/ directory)
```bash 
  python test.py '...other arguments...' --visualize  
```

#### Example qualitative results (1-shot):

<p align="middle">
    <img src="data/assets/qualitative_results.png">
</p>
   
## BibTeX
