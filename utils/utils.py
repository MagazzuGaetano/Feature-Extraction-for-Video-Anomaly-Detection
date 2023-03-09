import numpy as np
import torch
import os
import cv2
from opencv_transforms import transforms as T
from concurrent.futures import ThreadPoolExecutor


def from_frames_to_clips(frames, step=16):
    frame_cnt = len(frames)
    stack_depth = 16

    frame_indices = []  # Frames to chunks
    frame_ticks = range(1, frame_cnt + 1, step)
    for tick in frame_ticks:  # for clip in frames
        # frame indices for a clip
        frame_idx = [min(frame_cnt, tick + offset) for offset in range(stack_depth)]
        frame_indices.append(frame_idx)
    return np.asarray(frame_indices, dtype=object) - 1  # indices start from 0


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_clip_indices_into_batches(frame_indices, batch_size):
    frame_indices = list(divide_chunks(list(frame_indices), batch_size))

    y = len(frame_indices[-1])  # number of clips inside the last batch
    pad = batch_size - y  # how many clip to fill

    # pad the last batch with a number of (x - y) clips
    # by repeating the last clip inside the last batch
    last_batch = [clip for clip in frame_indices[-1]] + [frame_indices[-1][-1]] * pad
    frame_indices[-1] = last_batch

    # print("Exact size:", (len(frame_indices) - 1) * batch_size + y)

    return np.asarray(frame_indices, dtype=np.uint32), y


# Normalize the video frames with mean and standard deviation calculated across all images
# ResNet50 was pretrained on ImageNet Than trained on Kinetics same for the C3D model trained on the Sports-1M
def imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


def transform_frame(frame_path):
    mean, std = imagenet_mean_std()
    crop_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    transform = T.Compose([
        # T.Resize(256),
        T.TenCrop(224),  # 112 for C3D or 224 for I3D
        T.Lambda(lambda crops: torch.stack([crop_transform(crop) for crop in crops])),
    ])

    # read a single frame
    frame = cv2.imread(frame_path)

    # apply data augmentation
    frame = transform(frame)
    return frame


def load_rgb_batch(frames_dir, rgb_files, frame_indices, device):
    # B=16 x T=16 x CROP=10 x CH=3 x H=224 x W=224
    out = torch.zeros(frame_indices.shape + (10, 3, 224, 224), device=device)

    # num of clips
    n_clips = frame_indices.shape[0]
    for i in range(n_clips):

        # num of frames inside a clip
        n_frames = frame_indices.shape[1]
        frames_path = [os.path.join(frames_dir, rgb_files[frame_indices[i][j]]) for j in range(n_frames)]

        frames = None
        with ThreadPoolExecutor(max_workers=8) as executor:
            frames = list(executor.map(transform_frame, frames_path))

        for j in range(len(frames)):
            out[i, j, :, :, :, :] = frames[j]

    return out
