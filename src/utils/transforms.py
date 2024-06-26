import numpy as np
import torch
import cv2

from opencv_transforms import transforms as cv_t
from concurrent.futures import ThreadPoolExecutor
from deprecation import deprecated

sports1m_mean = np.load("data/c3d_mean.npy")
sports1m_mean = sports1m_mean.squeeze().transpose(1, 2, 3, 0).astype(np.float32)


class CropAugmentationFrame(cv_t.Compose):
    def __init__(self, size, n_crops=10, crop_transform=None):
        self.size = size
        self.n_crops = n_crops
        self.crop_transform = crop_transform

        self.transforms = [
            cv_t.TenCrop(size) if n_crops == 10 else cv_t.FiveCrop(size),
            cv_t.Lambda(
                lambda crops: torch.stack([crop_transform(crop) for crop in crops])
            ),
        ]


@deprecated(
    details="This is an approximation. You should subtract the mean tensor by using the 'transform_clip_c3d' function."
)
def c3d_normalization(tensor):
    """Given a tensor of frames return the normalized tensor, by applaying mean-std normalization.
    The C3D model were trained from scratch on the Sports-1M"""

    # This is an approximation: (given a clip 16x3x128x171 you should subtract the mean_tensor 16x3x128x171)
    # BGR_MEAN = (90.25164677347574, 97.65700354418385, 101.4083346052324)
    # BGR_MEAN / 255 = (0.35392802656265, 0.38296864134974057, 0.39767974354993096)

    mean = [0.35392802656265, 0.38296864134974057, 0.39767974354993096]
    std = [1.0, 1.0, 1.0]
    return cv_t.Normalize(mean, std)(tensor)


def i3d_normalization(tensor):
    """Given a tensor of frames return the normalized tensor, following the original preprocessing.
    The I3D model is based on Inception-V1 pretrained on ImageNet than trained on Kinetics-400"""

    # NOTE: Not the original normalization!!!
    # Normalize the video frames with ImageNet mean and standard deviation
    # (RGB) mean = [0.485, 0.456, 0.406]
    # (RGB) std = [0.229, 0.224, 0.225]
    # return cv_t.Normalize(mean, std)(tensor)

    # NOTE: Found in the Kinetics official repository
    # Pixel values are then rescaled between -1 and 1
    tensor = tensor * 2 - 1
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0
    return tensor


def read_frame(frame_path, use_rgb=False):
    # read a single frame image
    frame = cv2.imread(frame_path)

    # Grayscale frames are converted in RGB frames by repeating the same channel
    if len(frame.shape) <= 2:
        frame = cv2.merge([frame, frame, frame])

    # NOTE: opencv read images in BGR format
    if use_rgb:
        frame = frame[:, :, [2, 1, 0]]  # from BGR to RGB

    return frame


def transform_frame_i3d(frame, patch_size, n_crops=10):
    """Given a single frame return the preprocessed frame, in case of 10/5 crop
    data augmentation more frames are returned"""

    # NOTE: in the original preprocessing of I3D all frames are resized with the min_side to 256
    # but i never used resize because I get better results without, i only ensure the resolution for cropping

    # NOTE: to apply 10/5 crop data augmentations frames should be larger enough in resolution
    # for I3D patches of 224 x 224 and for C3D patches of 112 x 112
    # if H < 226 or W < 226 then resize by keeping aspect ratio with min_side = 226
    if frame.shape[0] < 226 or frame.shape[1] < 226:
        frame = cv_t.Resize(226)(frame)

    # transforms applied to single crops
    crop_transform = cv_t.Compose(
        [
            cv_t.ToTensor(),
            i3d_normalization,
        ]
    )
    # 10/5 crop data augmentation
    transform = CropAugmentationFrame(
        patch_size, n_crops, crop_transform=crop_transform
    )
    frame = transform(frame)
    return frame


def transform_clip_from_frames_i3d(frames, patch_size, n_crops):
    n_frames = frames.shape[0]  # T x H x W x C
    out_frames = torch.zeros((n_frames, n_crops, 3, patch_size, patch_size))

    # transform each single frame in the clip
    for j in range(n_frames):
        out_frames[j] = transform_frame_i3d(frames[j], patch_size, n_crops)

    return out_frames


def transform_clip_from_frames_c3d(frames, patch_size, n_crops):
    n_frames = frames.shape[0] # T x H x W x C
    H = 128
    W = 171
    CH = 3

    # resize all 16 j-frames of the clip-i
    resized_frames = np.zeros((n_frames, H, W, CH), dtype=np.float32)
    for j in range(n_frames):
        resized_frames[j] = cv_t.Resize((H, W))(frames[j])

    # mean-subtraction normalization sports-1m
    resized_frames = resized_frames - sports1m_mean

    processed_frames = torch.zeros((n_frames, n_crops, CH, patch_size, patch_size))
    transform = CropAugmentationFrame(
        patch_size, n_crops, crop_transform=cv_t.ToTensor()
    )

    # crop patches of 112x112 for all 16 j-frames of the clip-i
    for j in range(n_frames):
        processed_frames[j] = transform(resized_frames[j])

    return processed_frames


def transform_clip_from_paths_c3d(frames_path, patch_size, n_crops):
    n_frames = len(frames_path)
    frames = np.zeros((n_frames, 128, 171, 3))

    # read and resize all 16 j-frames of the clip-i
    for j in range(n_frames):
        frame = read_frame(frames_path[j])
        frames[j] = cv_t.Resize((128, 171))(frame)

    # mean-subtraction normalization sports-1m
    frames = np.asarray(frames, dtype=np.float32)
    frames = frames - sports1m_mean

    processed_frames = torch.zeros((n_frames, n_crops, 3, patch_size, patch_size))
    transform = CropAugmentationFrame(
        patch_size, n_crops, crop_transform=cv_t.ToTensor()
    )

    # crop patches of 112x112 for all 16 j-frames of the clip-i
    for j in range(n_frames):
        processed_frames[j] = transform(frames[j])

    return processed_frames


def transform_clip_from_paths_i3d(frames_path, patch_size, use_rgb, n_crops):
    def process_frame(frame_path):
        frame = read_frame(frame_path, use_rgb)

        return transform_frame_i3d(frame, patch_size, n_crops)

    with ThreadPoolExecutor(max_workers=8) as executor:
        processed_frames = list(executor.map(process_frame, frames_path))

    return torch.stack(processed_frames)
