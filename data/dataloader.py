"""
WebDataset-based dataloader using pre-mined JPEG pairs.
Modified for Contrastive Learning: Extracts Geography (Lat/Lon) from meta.json
"""

import torch
import json
from torchvision.transforms import v2 as T
from torch.utils.data import IterableDataset, DataLoader
import webdataset as wds
import os

class GeoPairDataset(IterableDataset):
    def __init__(
        self,
        dataset: wds.WebDataset,
        config: dict,
    ):
        super().__init__()
        self.dataset = dataset
        
        train_cfg = config["training"]
        aug_scale = train_cfg.get("resized_crop_scale", [0.7, 1.0])
        cj_cfg = train_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05})

        # GENTLE AUGMENTATION: Essential to preserve "Baukultur" architectural signatures
        self.augment_pipeline = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomResizedCrop(size=(518, 518), scale=tuple(aug_scale), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(**cj_cfg)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.count = 0

    def __iter__(self):
        """
        Yields (img1, img2, lat, lon) by decoding the .tar shards.
        """
        stream = self.dataset.decode("torchrgb8").to_tuple("img1.jpg", "img2.jpg", "meta.json")
        
        for img1, img2, raw_meta in stream:
            try:
                # 1. Parse Metadata
                if isinstance(raw_meta, (bytes, bytearray)):
                    meta = json.loads(raw_meta.decode("utf-8"))
                elif isinstance(raw_meta, str):
                    meta = json.loads(raw_meta)
                elif isinstance(raw_meta, dict):
                    meta = raw_meta
                else:
                    meta = {}
                    
                lat_val = meta.get("lat", 0.0)
                lon_val = meta.get("lon", 0.0)
                
                lat = torch.tensor(lat_val, dtype=torch.float32)
                lon = torch.tensor(lon_val, dtype=torch.float32)

                # 2. Augment Views
                view1 = self.augment_pipeline(img1)
                view2 = self.augment_pipeline(img2)

                self.count += 1
                if self.count % 100 == 0:
                    print(f"[Dataloader] Yielded {self.count} samples...", flush=True)

                yield view1, view2, lat, lon
            except Exception as e:
                if self.count < 5:
                    print(f"[Dataloader Error] {e}", flush=True)
                continue


def get_dataloader(
    shards_pattern: list or str,
    config: dict,
    batch_size: int = 32,
    num_workers: int = 8,
    **kwargs
) -> DataLoader:
    if isinstance(shards_pattern, list):
        pattern = shards_pattern
    else:
        shards_dir = os.path.dirname(shards_pattern)
        pattern = os.path.join(shards_dir, "*.tar")
    
    train_cfg = config.get("training", {})
    shuffle_buf = train_cfg.get("shuffle_buffer", 20000)
    
    # DECORRELATION: Explicit shuffle buffer before resampling
    dataset = wds.WebDataset(pattern, shardshuffle=True, resampled=True)
    dataset = dataset.shuffle(shuffle_buf)
    ds = GeoPairDataset(dataset, config)

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
