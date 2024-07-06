'''import os
import cv2
import json

class ImageNet100:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_folders = sorted([folder for folder in os.listdir(dataset_dir) if folder.startswith('train')])
        self.val_folder = os.path.join(dataset_dir, 'val.X')
        self.labels_file = os.path.join(dataset_dir, 'Labels.json')
        self.labels = self._load_labels()

    def _load_labels(self):
        with open(self.labels_file, 'r') as f:
            labels = json.load(f)
        return labels

    def get_train_data(self):
        train_data = []
        for train_folder in self.train_folders:
            class_folders = sorted(os.listdir(os.path.join(self.dataset_dir, train_folder)))
            for class_folder in class_folders:
                class_path = os.path.join(self.dataset_dir, train_folder, class_folder)
                images = os.listdir(class_path)
                for image_filename in images:
                    image_path = os.path.join(class_path, image_filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    label = self.labels[class_folder]
                    train_data.append((image, label))
        return train_data

    def get_validation_data(self):
        val_data = []
        class_folders = sorted(os.listdir(self.val_folder))
        for class_folder in class_folders:
            class_path = os.path.join(self.val_folder, class_folder)
            images = os.listdir(class_path)
            for image_filename in images:
                image_path = os.path.join(class_path, image_filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                label = self.labels[class_folder]
                val_data.append((image, label))
        return val_data

    def __getitem__(self, idx):
        if idx < len(self.train_folders):
            train_data = self.get_train_data()
            return train_data[idx]
        else:
            val_data = self.get_validation_data()
            return val_data[idx - len(self.train_folders)]

    def __len__(self):
        return len(self.get_train_data()) + len(self.get_validation_data())

'''
import os
import json
import cv2
from torchvision import datasets
import torchvision.transforms as transforms

'''class ImageNet100(datasets.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        self.loader = cv2_loader

    def _find_classes(self):
        classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    images.append(item)
        return images

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.samples)

def cv2_loader(path):
    # Using OpenCV to load image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img'''
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

'''class ImageNet100(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 初始化时加载数据集
        self.reload_dataset()
    
    def reload_dataset(self):
        if self.train:
            dataset_dir = os.path.join(self.root, 'train.X1')
        else:
            dataset_dir = os.path.join(self.root, 'val.X')
        
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        return self.image_folder[idx]'''
    

import os
import cv2
import json
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import random
import copy

'''class ImageNet100(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_dir = '/home/yuanweijie/workspace/backdoorattack/BackdoorBox/core/attacks/ImageNet100'
        self.train_folders = sorted([folder for folder in os.listdir(self.dataset_dir) if folder.startswith('train')])
        self.val_folder = os.path.join(self.dataset_dir, 'val.X')
        self.labels_file = os.path.join(self.dataset_dir, 'Labels.json')
        self.labels = self._load_labels()

    def _load_labels(self):
        with open(self.labels_file, 'r') as f:
            labels = json.load(f)
        return labels

    def get_train_data(self):
        train_data = []
        for train_folder in self.train_folders:
            class_folders = sorted(os.listdir(os.path.join(self.dataset_dir, train_folder)))
            for class_folder in class_folders:
                class_path = os.path.join(self.dataset_dir, train_folder, class_folder)
                images = os.listdir(class_path)
                for image_filename in images:
                    image_path = os.path.join(class_path, image_filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    label = self.labels[class_folder]
                    train_data.append((image, label))
        return train_data

    def get_validation_data(self):
        val_data = []
        class_folders = sorted(os.listdir(self.val_folder))
        for class_folder in class_folders:
            class_path = os.path.join(self.val_folder, class_folder)
            images = os.listdir(class_path)
            for image_filename in images:
                image_path = os.path.join(class_path, image_filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                label = self.labels[class_folder]
                val_data.append((image, label))
        return val_data

    def __len__(self):
        if self.train:
            return len(self.get_train_data())
        else:
            return len(self.get_validation_data())

    def __getitem__(self, idx):
        if self.train:
            data = self.get_train_data()
        else:
            data = self.get_validation_data()

        img, target = data[idx]

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target'''

'''class ImageNet100(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 初始化时加载数据集
        self.reload_dataset()
        self.data = self.get_data()
        self.targets = self.get_targets()
    
    def reload_dataset(self):
        # 根据 self.train 设置加载训练集或验证集
        if self.train:
            dataset_dir = os.path.join(self.root, 'train.X1')
        else:
            dataset_dir = os.path.join(self.root, 'val.X')

        # 定义要应用于每个图像的转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 将 PIL 图像转换为张量
            ])

        # 加载数据集
        self.image_folder = ImageFolder(root=dataset_dir,
                                        transform=self.transform,
                                        loader=cv2_loader)  # 使用自定义loader加载图像

        # 获取类别到索引的映射
        self.class_to_idx = self.image_folder.class_to_idx
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        return self.image_folder[idx]

    def get_data(self):
        # 通过 self.image_folder 获取所有数据
        data = []
        for index in range(len(self.image_folder)):
            img, target = self.image_folder[index]
            data.append((img))
        return data
    
    def get_targets(self):
        # 通过 self.image_folder 获取所有数据
        targets = []
        for index in range(len(self.image_folder)):
            img, target = self.image_folder[index]
            targets.append((target))
        return targets
    

    
def cv2_loader(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(image)
    return image'''
import os
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class ImageNet100(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform       
        # 初始化时加载数据集
        self.reload_dataset()
        
        # 获取所有数据和目标
        self.data, self.targets = self.get_data_and_targets()
    
    def reload_dataset(self):
        # 根据 self.train 设置加载训练集或验证集
        if self.train:
            dataset_dir = os.path.join(self.root, 'train.X1')
        else:
            dataset_dir = os.path.join(self.root, 'val.X')



        # 加载数据集
        self.image_folder = ImageFolder(root=dataset_dir,
                                        transform=self.transform,
                                        loader=cv2_loader)  # 使用自定义loader加载图像

        # 获取类别到索引的映射
        self.class_to_idx = self.image_folder.class_to_idx
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        return self.image_folder[idx]

    def get_data_and_targets(self):
        # 获取所有数据和目标
        data = []
        targets = []
        for index in range(len(self.image_folder)):
            img, target = self.image_folder[index]
            data.append(img)
            targets.append(target)
        return data, targets

# 自定义 loader 函数，用于加载图像
def cv2_loader(path):
    import cv2
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(image)
    return image



