import os
import glob
import ffmpeg
from natsort import natsorted
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from utils import util
from utils.utils2 import from_frames_to_clips, load_rgb_batch
from utils.utils2 import split_clip_indices_into_batches



class UCF_CRIME(torch.utils.data.Dataset):

    def __init__(self, temp_path, root_path, split, chunk_size, frequency, batch_size):
        super(UCF_CRIME, self).__init__()

        self.temp_path = temp_path
        self.root = root_path
        self.split = split
        self.chunk_size = chunk_size
        self.frequency = frequency
        self.batch_size = batch_size

        self.clip_transform = util.clip_transform(self.split)
        self.videos = [str(f) for f in self.root.glob('**/*.mp4')]

    def __getitem__(self, index):

        video = self.videos[index]
        return video

    def __len__(self):
        return len(self.data)

