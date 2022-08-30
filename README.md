
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
> ### Training process
> ``` bash
> python train.py --nshot {1, 5}

> * Training takes approx (trained with four 2080 Ti GPUs).


> ### Babysitting training:
> Use tensorboard to babysit training progress:
> - For each experiment, a directory that logs training progress will be automatically generated under logs/ directory. 
> - From terminal, run 'tensorboard --logdir logs/' to monitor the training progress.
> - Choose the best model when the validation (mIoU) curve starts to saturate. 



## Testing
> ```bash
> python test.py --nshot {1, 5} --load "path_to_trained_model/best_model.pt"
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
