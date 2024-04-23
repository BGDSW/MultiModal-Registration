import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import cv2
import numpy as np

def save_tensor_img(tensor_img: torch.Tensor, name='./123.jpg'):
    while(tensor_img.dim()>2):
        tensor_img = tensor_img.squeeze(0)
    img = (tensor_img.cpu().numpy() + 1) / 2 * 255
    print(img.shape)
    cv2.imwrite(name, img.astype(np.uint8))

def show_tensor_img(tensor_img: torch.Tensor, name='123'):
    img = (tensor_img.cpu().squeeze(0).numpy() + 1) / 2
    # img = img.transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(0)


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        if (not opt.unsupervised):
            self.dir_segA = os.path.join(opt.dataroot, 'segA')
            self.dir_segB = os.path.join(opt.dataroot, 'segB')
            self.segA_paths = sorted(
                make_dataset(self.dir_segA, opt.max_dataset_size))  # load images from '/path/to/data/segA'
            self.segB_paths = sorted(
                make_dataset(self.dir_segB, opt.max_dataset_size))  # load images from '/path/to/data/segB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        print('size')
        print(self.A_size)
        print(self.B_size)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        # print(input_nc)
        # input('-----------')
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_seg = get_transform(self.opt, grayscale=True)
        self.phase = phase
        self.unsupervised = opt.unsupervised

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # print(index)

        if index >= self.__len__():
            raise IndexError
        # print('hihihi')
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        if (self.unsupervised):
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        elif (self.phase == 'test'):
            segA_path = self.segA_paths[index % self.A_size]
            segB_path = self.segB_paths[index % self.A_size]
            seg_A = torch.zeros_like(A)
            seg_B = seg_A
        else:
            segA_path = self.segA_paths[index % self.A_size]
            segB_path = self.segB_paths[index % self.A_size]
            segA_img = Image.open(segA_path)
            segB_img = Image.open(segB_path)
            seg_A = self.transform_seg(segA_img)
            seg_B = self.transform_seg(segB_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'seg_A': seg_A, 'seg_B': seg_B,
                'segA_path': segA_path, 'segB_path': segB_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


class UnalignedDataset2(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, phase='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        if (not opt.unsupervised):
            self.dir_segA = os.path.join(opt.dataroot, 'segA')
            self.dir_segB = os.path.join(opt.dataroot, 'segB')
            self.segA_paths = sorted(
                make_dataset(self.dir_segA, opt.max_dataset_size))  # load images from '/path/to/data/segA'
            self.segB_paths = sorted(
                make_dataset(self.dir_segB, opt.max_dataset_size))  # load images from '/path/to/data/segB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        print('size')
        print(self.A_size)
        print(self.B_size)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=True)
        self.transform_B = get_transform(self.opt, grayscale=True)
        self.transform_seg = get_transform(self.opt, grayscale=True)
        self.phase = phase
        self.unsupervised = opt.unsupervised
        self.mask = torch.from_numpy(cv2.imread(opt.dataroot+'/mask256.png', 0)).to(torch.float32).unsqueeze(0) / 255 * 2 - 1



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # print(index)

        if index >= self.__len__():
            raise IndexError
        # print('hihihi')
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        if(self.unsupervised):
            A = self.transform_A(A_img)
        else:
            # A = -self.transform_A(A_img)
            A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        A = torch.where(self.mask > 0.9, A, self.mask)
        B = torch.where(self.mask > 0.9, B, self.mask)


        if (self.unsupervised):
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        elif (self.phase == 'test'):
            seg_A = torch.zeros_like(A)
            seg_B = seg_A
            segA_path = 'test no path'
            segB_path = 'test no path'
        else:
            segA_path = self.segA_paths[index % self.A_size]
            segB_path = self.segB_paths[index % self.A_size]
            segA_img = Image.open(segA_path)
            segB_img = Image.open(segB_path)
            seg_A = self.transform_seg(segA_img)
            seg_B = self.transform_seg(segB_img)
            seg_A = torch.where(self.mask == 1, seg_A, self.mask)
            seg_B = torch.where(self.mask == 1, seg_B, self.mask)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'seg_A': seg_A, 'seg_B': seg_B,
                'segA_path': segA_path, 'segB_path': segB_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


class MyUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        self.dir_segA = os.path.join(opt.dataroot, 'segA')
        self.dir_segB = os.path.join(opt.dataroot, 'segB')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.segA_paths = sorted(
            make_dataset(self.dir_segA, opt.max_dataset_size))  # load images from '/path/to/data/segA'
        self.segB_paths = sorted(
            make_dataset(self.dir_segB, opt.max_dataset_size))  # load images from '/path/to/data/segB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_seg = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if index >= self.__len__():
            raise IndexError

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        segA_path = self.segA_paths[index % self.A_size]
        segB_path = self.segB_paths[index % self.A_size]
        index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        A = torch.load(A_path).cuda()
        B = torch.load(B_path).cuda()
        segA_img = Image.open(segA_path)
        segB_img = Image.open(segB_path)
        # apply image transformation
        seg_A = self.transform_seg(segA_img)
        seg_B = self.transform_seg(segB_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'seg_A': seg_A, 'seg_B': seg_B,
                'segA_path': segA_path, 'segB_path': segB_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
