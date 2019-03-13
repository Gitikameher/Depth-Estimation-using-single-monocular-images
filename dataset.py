import torch
import h5py
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tfms):
        super(NYUDataset, self).__init__()
        self.data_dir = data_dir
        self.tfms = tfms
        
        #self.ds_v_1 = h5py.File(self.data_dir+'nyu_depth_data_labeled.mat')
        self.ds_v_2 = h5py.File(self.data_dir+'nyu_depth_v2_labeled.mat')
        
        #self.len = len(self.ds_v_1["images"]) + len(self.ds_v_2["images"])
        self.len = len(self.ds_v_2["images"])

           
    def __getitem__(self, index):
        if(index<len(self.ds_v_2["images"])):
            ds = self.ds_v_2    
            i = index
        else:    
            ds = self.ds_v_2
            i = index - len(self.ds_v_2["images"])

        img = np.transpose(ds["images"][i], axes=[2,1,0])
        img = img.astype(np.uint8)

        depth = np.transpose(ds["depths"][i], axes=[1,0])
        
        depth = (depth/depth.max())*255
        depth = depth.astype(np.uint8)
        
        if self.tfms:
            tfmd_sample = self.tfms({"image":img, "depth":depth})
            img, depth = tfmd_sample["image"], tfmd_sample["depth"]
        
        depth = (depth - torch.min(depth))/(torch.max(depth) - torch.min(depth)) 
        return (img, depth)
    
    def __len__(self):
        return self.len 
    
    def create_split_loaders(self, batch_size, seed, transform,
                         p_val=0.1, p_test=0.2, shuffle=True, 
                         show_sample=False):
        """ Creates the DataLoader objects for the training, validation, and test sets. 
        Params:
        -------
        - batch_size: (int) mini-batch size to load at a time
        - seed: (int) Seed for random generator (use for testing/reproducibility)
        - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
        - p_val: (float) Percent (as decimal) of dataset to use for validation
        - p_test: (float) Percent (as decimal) of the dataset to split for testing
        - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
        - show_sample: (bool) Plot a mini-example as a grid of the dataset
        - extras: (dict) 
            If CUDA/GPU computing is supported, contains:
            - num_workers: (int) Number of subprocesses to use while loading the dataset
            - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
            Otherwise, extras is an empty dict.
        Returns:
        --------
        - train_loader: (DataLoader) The iterator for the training set
        - val_loader: (DataLoader) The iterator for the validation set
        - test_loader: (DataLoader) The iterator for the test set
         """

        # Get create a ChestXrayDataset object
        dataset = NYUDataset('data/',self.tfms)

        # Dimensions and indices of training set
        dataset_size = len(dataset)
        all_indices = list(range(dataset_size))

        # Shuffle dataset before dividing into training & test sets
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(all_indices)
    
        # Create the validation split from the full dataset
        val_split = int(np.floor(p_val * dataset_size))
        train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
        # Separate a test split from the training dataset
        test_split = int(np.floor(p_test * len(train_ind)))
        train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    
        # Use the SubsetRandomSampler as the iterator for each subset
        sample_train = SubsetRandomSampler(train_ind)
        sample_test = SubsetRandomSampler(test_ind)
        sample_val = SubsetRandomSampler(val_ind)

        
        # Define the training, test, & validation DataLoaders
        train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=sample_train)

        test_loader = DataLoader(dataset, batch_size=batch_size, 
                             sampler=sample_test)

        val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val)

    
    # Return the training, validation, test DataLoader objects
        return (train_loader, val_loader, test_loader)
