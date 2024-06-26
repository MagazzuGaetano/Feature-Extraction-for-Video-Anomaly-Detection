import numpy as np
import torch
import os
import cv2

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from src.utils.transforms import (
    transform_clip_from_paths_c3d,
    transform_clip_from_paths_i3d,
)


def extract_frames_from_video(video_name, output_dir):
    cap = cv2.VideoCapture(video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_res = np.asarray(cap.read()[1]).shape

    if frame_count == 0:
        exit("Video file not found")

    batch_size = 2**12  # how many frames to save in parallel
    batches = list(divide_chunks([0 for _ in range(frame_count)], batch_size))

    k = 0
    for i in range(len(batches)):
        frames_to_save = []
        frames_paths = []
        for j in range(len(batches[i])):
            frames_to_save.append(cap.read()[1])
            frames_paths.append(os.path.join(output_dir, "image%d.jpg" % k))
            k = k + 1

        max_workers = 8
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            f = lambda x: cv2.imwrite(x[0], x[1])
            futures = list(
                map(lambda x: executor.submit(f, x), zip(frames_paths, frames_to_save))
            )
            wait(futures, return_when=ALL_COMPLETED)

    cap.release()
    return frame_count, frame_res


def read_video(video_path, use_rgb=False, transform_frame=None):
    # Empty List declared to store video frames
    frames_list = []

    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:
        # Reading a frame from the video file
        success, frame = video_reader.read()

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Grayscale frames are converted in RGB frames by repeating the same channel
        if len(frame.shape) <= 2:
            frame = cv2.merge([frame, frame, frame])

        # NOTE: opencv read images in BGR format
        if use_rgb:
            frame = frame[:, :, [2, 1, 0]]  # from BGR to RGB

        # preprocess the frame
        if transform_frame:
            frame = transform_frame(frame)

        # Appending the preprocessed frame into the frames list
        frames_list.append(frame)

    # Closing the VideoCapture object and releasing all resources.
    video_reader.release()

    # returning the frames list
    return frames_list


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
        yield l[i : i + n]


def split_clip_indices_into_batches(frame_indices, batch_size):
    frame_indices = list(divide_chunks(list(frame_indices), batch_size))

    y = len(frame_indices[-1])  # number of clips inside the last batch
    pad = batch_size - y  # how many clip to fill

    # pad the last batch with a number of (x - y) clips
    # by repeating the last clip inside the last batch
    last_batch = [clip for clip in frame_indices[-1]] + [frame_indices[-1][-1]] * pad
    frame_indices[-1] = last_batch

    return np.asarray(frame_indices, dtype=np.uint32), y


def load_rgb_batch(frames_dir, rgb_files, frame_indices, feature, device, n_crops):
    # B=16 x T=16 x CROP=10 x CH=3 x H x W
    patch_size = 224 if feature == "I3D" else 112

    out = torch.zeros(
        frame_indices.shape + (n_crops, 3, patch_size, patch_size), device=device
    )

    n_clips = frame_indices.shape[0]  # num of clips
    n_frames = frame_indices.shape[1]  # num of frames inside a clip

    for i in range(n_clips):
        frames_path = [
            os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
            for j in range(n_frames)
        ]

        if feature == "I3D":
            processed_frames = transform_clip_from_paths_i3d(
                frames_path, patch_size, True, n_crops
            )
        else:
            processed_frames = transform_clip_from_paths_c3d(
                frames_path, patch_size, n_crops
            )

        out[i] = processed_frames

    return out
