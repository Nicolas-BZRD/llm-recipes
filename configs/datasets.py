# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
 
@dataclass
class custom_dataset:
    file: str = None
    train_split: str = "train"
    test_split: str = "validation"