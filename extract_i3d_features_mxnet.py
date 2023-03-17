import os
from pathlib import Path
import argparse

from natsort import natsorted
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from utils.utils import from_frames_to_clips, split_clip_indices_into_batches, extract_frames_from_video

import mxnet
from mxnet import is_np_array
from mxnet.gluon import HybridBlock
from mxnet.gluon.data.vision import transforms as gluon_t
from gluoncv.data.transforms import image as image_t
from gluoncv.model_zoo import i3d_inceptionv3_kinetics400 as i3d_model


class TenCrop(HybridBlock):
    def __init__(self, size):
        super(TenCrop, self).__init__()
        self._size = size

    def hybrid_forward(self, F, x):
        if is_np_array():
            F = F.npx
        return image_t.ten_crop(x, self._size)


# Normalize the video frames with mean and standard deviation calculated across all images
# ResNet50 and Inception-V1 were pretrained on ImageNet than trained on Kinetics
def imagenet_mean_std():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


def transform_frame_mxnet(frame_path):
    mean, std = imagenet_mean_std()
    crop_transform = gluon_t.Compose([
        gluon_t.ToTensor(),
        gluon_t.Normalize(mean, std),
    ])
    transform = gluon_t.Compose([
        TenCrop((224, 224)),
        lambda crops: mxnet.nd.stack(
            mxnet.nd.array([crop_transform(crops[i, :, :, :]) for i in range(10)]), ctx=mxnet.gpu(0)
        ),
    ])

    # read a single frame
    frame = cv2.imread(frame_path)

    # from numpy array to mxnet array
    frame = mxnet.nd.array(frame, ctx=mxnet.gpu(0))

    # apply data augmentation
    frame = transform(frame)

    return frame


def load_rgb_batch_mxnet(frames_dir, rgb_files, frame_indices, device):
    # B=16 x T=16 x CROP=10 x CH=3 x H=224 x W=224
    out = mxnet.nd.zeros(frame_indices.shape + (10, 3, 224, 224), ctx=device)

    # num of clips
    n_clips = frame_indices.shape[0]
    for i in range(n_clips):

        # num of frames inside a clip
        n_frames = frame_indices.shape[1]
        frames_path = [os.path.join(frames_dir, rgb_files[frame_indices[i][j]]) for j in range(n_frames)]

        frames = None
        with ThreadPoolExecutor(max_workers=8) as executor:
            frames = list(executor.map(transform_frame_mxnet, frames_path))

        for j in range(len(frames)):
            out[i, j, :, :, :, :] = frames[j]

    return out


def extract_features(model, device, clip_length, temp_path, batch_size):
    # rgb files
    rgb_files = natsorted([i for i in os.listdir(temp_path)])

    # creating clips
    frame_indices = from_frames_to_clips(rgb_files, clip_length)

    # divide clips into batches (for memory optimization)
    # k = true values in the last batch
    frame_indices, k = split_clip_indices_into_batches(frame_indices, batch_size)

    n_batches = frame_indices.shape[0]
    n_clips_inside_a_batch = frame_indices.shape[1]

    # iterate 10-crop augmentation
    full_features = mxnet.nd.zeros((n_batches, 10, n_clips_inside_a_batch, 2048, 1, 1, 1), ctx=device)
    for batch_id in range(n_batches):

        # B=16 x T=16 x CROP=10 x CH=3 x H=224 x W=224
        batch_data = load_rgb_batch_mxnet(temp_path, rgb_files, frame_indices[batch_id], device)

        # iterate 10-crop augmentation
        n_crops = batch_data.shape[2]
        for i in range(n_crops):
            b_data = batch_data[:, :, i, :, :, :]  # select the i-crop
            b_data = b_data.transpose(1, 2)  # B=16 x CH=3 x T=16 x H=224 x W=224

            features = model(b_data)

            v = features  # single clip features extracted
            v = v / mxnet.np.linalg.norm(v)  # single clip features normalized (L2)
            full_features[batch_id, i, :, :, :, :, :] = v

    full_features = full_features.asnumpy()

    # from fixed tensor to variable length list
    out = [[] for i in range(n_crops)]
    for batch_id in range(n_batches):
        for i in range(n_crops):
            if batch_id == n_batches - 1:
                # ignore padded values in the last batch
                out[i].append(full_features[batch_id, i, :k, :, :, :, :])
            else:
                out[i].append(full_features[batch_id, i, :, :, :, :, :])

    out = [np.concatenate(i, axis=0) for i in out]
    out = [np.expand_dims(i, axis=0) for i in out]
    out = np.concatenate(out, axis=0)
    out = out[:, :, :, 0, 0, 0]
    out = np.array(out).transpose([1, 0, 2])

    return out


def generate(dataset_path, output_path, clip_size, batch_size):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    temp_path = output_path + "/temp/"
    root_dir = Path(dataset_path)
    # videos = [str(f) for f in root_dir.glob('**/*.mp4')]
    videos = [str(f) for f in root_dir.glob('**/*.avi')]

    # set up the model
    mxnet.random.seed(42)  # set seed for repeatability
    device = mxnet.gpu(0)
    model = i3d_model(nclass=400, pretrained=True, feat_ext=True, ctx=device)

    # create temp folder
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    for video in videos:
        # remove files into temp folder
        for f in Path(temp_path).glob('**/*'):
            os.remove(f)

        # splitting the video into frames and saving them to an output folder
        extract_frames_from_video(video, temp_path)

        # extract features
        features = extract_features(model, device, clip_size, temp_path, batch_size)

        # save features
        video_name = video.split("/")[-1].split(".")[0]
        np.save(output_path + "/" + video_name, features)

        print('features of dimension {} extracted from {}'.format(features.shape, video))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/media/ubuntu/EFB1-B23D/ShanghaiTech_new_split/test")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    generate(args.dataset_path, str(args.output_path), args.clip_length, args.batch_size)

# python3 extract_c3d_features.py --dataset_path=/home/ubuntu/Downloads/UCF-Crime/train --output_path=output
# python3 extract_c3d_features.py --dataset_path=ShanghaiTech_new_split/train/ --output_path=output
