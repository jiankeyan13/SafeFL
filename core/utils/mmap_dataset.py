import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import json

class MemoryMappedDataset(Dataset):
    """
    基于 Numpy Memmap 的高性能数据集。
    使用 Raw Binary 模式，配合 meta.json 存储元数据。
    """
    def __init__(self, original_dataset, cache_path, transform=None):
        self.transform = transform
        
        # 这里的路径是前缀，我们生成 .bin (数据) 和 .json (元数据)
        self.data_path = cache_path + "_data.bin"
        self.target_path = cache_path + "_targets.bin"
        self.meta_path = cache_path + "_meta.json"
        
        # 1. 检查并创建缓存
        if not (os.path.exists(self.data_path) and 
                os.path.exists(self.target_path) and 
                os.path.exists(self.meta_path)):
            self._create_cache(original_dataset)
            
        # 2. 加载元数据
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
            
        # 3. 映射内存 (使用 shape 和 dtype)
        # 图像数据
        self.data = np.memmap(
            self.data_path, 
            dtype=self.meta['data_dtype'], 
            mode='r', 
            shape=tuple(self.meta['data_shape'])
        )
        # 标签数据
        self.targets = np.memmap(
            self.target_path, 
            dtype=self.meta['target_dtype'], 
            mode='r', 
            shape=tuple(self.meta['target_shape'])
        )
        
        print(f"Loaded MemoryMappedDataset from {cache_path} {tuple(self.meta['data_shape'])}")

    def _create_cache(self, dataset):
        print(f"Creating raw binary cache at {self.data_path}...")
        
        # 1. 预读取以确定 Shape
        sample_img, _ = dataset[0]
        if hasattr(sample_img, 'numpy'):
            sample_np = sample_img.numpy()
        else:
            sample_np = np.array(sample_img)
            
        N = len(dataset)
        shape = (N,) + sample_np.shape
        dtype = sample_np.dtype
        
        # 2. 创建 memmap 文件 (写模式)
        fp_data = np.memmap(self.data_path, dtype=dtype, mode='w+', shape=shape)
        fp_targets = np.memmap(self.target_path, dtype=np.int64, mode='w+', shape=(N,))
        
        # 3. 写入数据
        for i in tqdm(range(N), desc="Caching"):
            img, target = dataset[i]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            fp_data[i] = img
            fp_targets[i] = target
            
        # 4. 刷入磁盘 (这一步很重要，确保数据写完)
        fp_data.flush()
        fp_targets.flush()
        
        # 5. 显式删除引用以关闭文件句柄 (Windows下尤为重要)
        del fp_data
        del fp_targets
        
        # 6. 保存元数据 (Shape 和 Dtype)
        meta = {
            'data_shape': shape,
            'data_dtype': np.dtype(dtype).name,
            'target_shape': (N,),
            'target_dtype': 'int64'
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f)
            
        print("Cache created successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        
        # 注意：np.memmap 返回的是只读视图，如果不 copy，转 PIL 可能会报错
        img = Image.fromarray(img) # Image.fromarray 会自动 copy 数据
        
        if self.transform:
            img = self.transform(img)
            
        return img, target