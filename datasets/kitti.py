from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch.utils.data as data
from glob import glob

from torchvision import transforms as vision_transforms

from datasets import transforms
from datasets import common


class Kitti(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 resize_targets=[-1,-1],
                 num_examples=-1,
                 dstype="train"):

        self._args = args
        self._resize_targets = resize_targets

        # ----------------------------------------------------------
        # Determine image and flow paths
        # ----------------------------------------------------------
        if dstype == "train":
            dir = "training"
        elif dstype == "valid":
            dir = "testing"
        else:
            raise ValueError("FlyingChairs: dstype '%s' unknown!", dstype)

        img_path = os.path.join(root, dir, "colored_0")      # KITTI 2012
        if not os.path.isdir(img_path):
            img_path = os.path.join(root, dir, "image_2")    # KITTI 2015

        flo_path = os.path.join(root, dir, "flow_occ")       # KITTI 2012/2015

        # ----------------------------------------------------------
        # Save list of filenames for inputs and flows
        # ----------------------------------------------------------
        self._image_list = []
        self._flow_list = []
        filenames = sorted(glob(os.path.join(img_path, "*_10.png")))
        for fname in filenames:
            ind = os.path.basename(fname)[:6]
            flo = os.path.join(flo_path, f"{ind}_10.png")
            im1 = os.path.join(img_path, f"{ind}_10.png")
            im2 = os.path.join(img_path, f"{ind}_11.png")
            self._image_list += [ [ im1, im2 ] ]
            if dstype == "train":
                self._flow_list += [ flo ]
        self._size = len(self._image_list)

        # ----------------------------------------------------------
        # photometric_augmentations
        # ----------------------------------------------------------
        if photometric_augmentations:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> PIL
                vision_transforms.ToPILImage(),
                # PIL -> PIL : random hsv and contrast
                vision_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                # PIL -> FloatTensor
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True),
            ], from_numpy=True, to_numpy=False)

        else:
            self._photometric_transform = transforms.ConcatTransformSplitChainer([
                # uint8 -> FloatTensor
                vision_transforms.transforms.ToTensor(),
            ], from_numpy=True, to_numpy=False)

    def __getitem__(self, index):
        index = index % self._size
        im1_filename = self._image_list[index][0]
        im2_filename = self._image_list[index][1]

        # read float32 images
        im1_np0 = common.read_image_as_byte(im1_filename)
        im2_np0 = common.read_image_as_byte(im2_filename)

        # possibly apply photometric transformations
        im1, im2 = self._photometric_transform(im1_np0, im2_np0)

        # example filename
        basename = os.path.basename(im1_filename)[:6]

        example_dict = {
            "input1": im1,
            "input2": im2,
            "index": index,
            "basename": basename
        }

        # If flow is available add it to the example
        if len(self._flow_list) > 0:
            flo_filename = self._flow_list[index]
            flo_np0 = common.read_flo_as_float32(flo_filename)
            flo = common.numpy2torch(flo_np0)
            example_dict.update({"target1": flo})

        # import numpy as np
        # from matplotlib import pyplot as plt
        # import numpy as np
        # plt.figure()
        # im1_np = im1.numpy().transpose([1,2,0])
        # im2_np = im2.numpy().transpose([1,2,0])
        # plt.imshow(np.concatenate((im1_np0.astype(np.float32)/255.0, im2_np0.astype(np.float32)/255.0, im1_np, im2_np), 1))
        # plt.show(block=True)

        return example_dict

    def __len__(self):
        return self._size


class KittiTrain(Kitti):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=True,
                 num_examples=-1):
        super(KittiTrain, self).__init__(
            args,
            root=root,
            photometric_augmentations=photometric_augmentations,
            dstype="train",
            num_examples=num_examples)


class KittiValid(Kitti):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=False,
                 num_examples=-1):
        super(KittiValid, self).__init__(
            args,
            root=root,
            photometric_augmentations=photometric_augmentations,
            dstype="valid",
            num_examples=num_examples)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    dataset = KittiTrain({}, root="/home/deu/Datasets/KITTI_2012", photometric_augmentations=False, num_examples=-1)
    loader = DataLoader(dataset, batch_size=1)
    for sample in loader:
        img1 = sample['input1'][0].numpy().transpose(1, 2, 0)
        img2 = sample['input2'][0].numpy().transpose(1, 2, 0)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        plt.show()
