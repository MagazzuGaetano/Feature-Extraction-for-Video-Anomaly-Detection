import numpy as np
import torch
import os
import cv2
from opencv_transforms import transforms as cv_t
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


def extract_frames_from_video(video_name, output_dir):
    cap = cv2.VideoCapture(video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0:
        exit("Video file not found")

    batch_size = 2 ** 12  # how many frames to save in parallel
    batches = list(divide_chunks([0 for _ in range(frame_count)], batch_size))

    k = 0
    for i in range(len(batches)):
        frames_to_save = []
        frames_paths = []
        for j in range(len(batches[i])):
            frames_to_save.append(cap.read()[1])
            frames_paths.append(os.path.join(output_dir, 'image%d.jpg' % k))
            k = k + 1

        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            f = lambda x: cv2.imwrite(x[0], x[1])
            futures = list(map(lambda x: executor.submit(f, x), zip(frames_paths, frames_to_save)))
            wait(futures, return_when=ALL_COMPLETED)

    cap.release()
    print('Frames Extracted From Video')


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

    return np.asarray(frame_indices, dtype=np.uint32), y


# Normalize the video frames with mean and standard deviation calculated across all images
# The C3D model were trained from scratch on the Sports-1M
def sports1M_mean_std():
    mean = (0.4823, 0.4588, 0.4078)
    std = (0.229, 0.224, 0.225)
    return mean, std


# Normalize the video frames with mean and standard deviation calculated across all images
# ResNet50 and Inception-V1 were pretrained on ImageNet than trained on Kinetics
def imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


def transform_frame(frame_path, size, mean_std, n_crops=10):
    # read a single frame
    frame = cv2.imread(frame_path)

    if len(frame.shape) <= 2:
        frame = cv2.merge([frame, frame, frame])

    mean, std = mean_std
    crop_transform = cv_t.Compose([
        cv_t.ToTensor(),
        cv_t.Normalize(mean, std),
    ])

    transform = cv_t.Compose([
        # T.Resize(256),
        cv_t.TenCrop(size) if n_crops == 10 else cv_t.FiveCrop(size),
        cv_t.Lambda(lambda crops: torch.stack([crop_transform(crop) for crop in crops])),
    ])
    frame = transform(frame)
    return frame


def load_rgb_batch(frames_dir, rgb_files, frame_indices, feature, device, n_crops):
    # B=16 x T=16 x CROP=10 x CH=3 x H x W
    image_size = 224 if feature == "I3D" else 112
    image_normalization = imagenet_mean_std() if feature == "I3D" else sports1M_mean_std()
    out = torch.zeros(frame_indices.shape + (n_crops, 3, image_size, image_size), device=device)

    # num of clips
    n_clips = frame_indices.shape[0]
    for i in range(n_clips):

        # num of frames inside a clip
        n_frames = frame_indices.shape[1]
        frames_path = [os.path.join(frames_dir, rgb_files[frame_indices[i][j]]) for j in range(n_frames)]

        frames = None
        with ThreadPoolExecutor(max_workers=8) as executor:
            if feature == "I3D":
                frames = list(executor.map(
                    lambda x: transform_frame(x, image_size, image_normalization, n_crops), frames_path))
            else:
                frames = list(executor.map(
                    lambda x: transform_frame(x, image_size, image_normalization, n_crops), frames_path))

        for j in range(len(frames)):
            out[i, j, :, :, :, :] = frames[j]

    return out
