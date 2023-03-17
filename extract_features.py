import os
from pathlib import Path
import argparse

from natsort import natsorted
import numpy as np
import torch

from models.i3d_inception import i3d_model
from models.c3d import c3d_model
from utils.utils import from_frames_to_clips, load_rgb_batch, split_clip_indices_into_batches, \
    extract_frames_from_video


def extract_features(model, feature, device, clip_length, temp_path, batch_size):
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
    feature_dim = 1024 if feature == "I3D" else 4096
    full_features = torch.zeros((n_batches, 10, n_clips_inside_a_batch, feature_dim, 1, 1, 1), device=device)
    for batch_id in range(n_batches):

        # B=16 x T=16 x CROP=10 x CH=3 x H x W
        batch_data = load_rgb_batch(temp_path, rgb_files, frame_indices[batch_id], feature, device)

        # iterate 10-crop augmentation
        n_crops = batch_data.shape[2]
        for i in range(n_crops):
            b_data = batch_data[:, :, i, :, :, :]  # select the i-crop
            b_data = b_data.transpose(1, 2)  # B=16 x CH=3 x T=16 x H x W

            with torch.no_grad():
                inp = {'frames': b_data}
                features = model(inp)

            v = features  # single clip features extracted
            v = v / torch.linalg.norm(v)  # single clip features normalized (L2)
            full_features[batch_id, i, :, :, :, :, :] = v

    full_features = full_features.cpu().numpy()

    # from fixed tensor to variable length list
    out = [[] for i in range(n_crops)]
    for batch_id in range(n_batches):
        for i in range(n_crops):
            if batch_id == n_batches - 1:
                # ignore padded values in the last batch
                out[i].append(full_features[batch_id, i, :k, :])
            else:
                out[i].append(full_features[batch_id, i, :, :])

    out = [np.concatenate(i, axis=0) for i in out]
    out = [np.expand_dims(i, axis=0) for i in out]
    out = np.concatenate(out, axis=0)
    out = out[:, :, :, 0, 0, 0]
    out = np.array(out).transpose([1, 0, 2])
    return out


def generate(dataset_path, output_path, feature, clip_size, batch_size):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    temp_path = output_path + "/temp/"
    root_dir = Path(dataset_path)
    # videos = [str(f) for f in root_dir.glob('**/*.mp4')]
    videos = [str(f) for f in root_dir.glob('**/*.avi')]

    # set up the model
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature == "I3D":
        model = i3d_model(nb_classes=400, pretrainedpath='pretrained/rgb_imagenet.pt')
    else:
        model = c3d_model(nb_classes=487, pretrainedpath='pretrained/c3d.pickle', feature_layer=6)

    model.to(device)  # Put model to GPU
    model.train(False)  # Set model to evaluate mode

    # create temp folder
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    for video in videos:
        # remove files into temp folder
        for f in Path(temp_path).glob('**/*'):
            os.remove(f)

        # splitting the video into frames and saving them to an output folder
        extract_frames_from_video(video, temp_path)

        # extract features
        features = extract_features(model, feature, device, clip_size, temp_path, batch_size)

        # save features
        video_name = video.split("/")[-1].split(".")[0]
        np.save(output_path + "/" + video_name, features)

        print('features of dimension {} extracted from {}'.format(features.shape, video))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/media/ubuntu/TrekStor/TESI_ANOMALY_DETECTION/Datasets/videos/ShanghaiTech_new_split/test")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--feature', type=str, default="I3D")
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    generate(args.dataset_path, str(args.output_path), args.feature, args.clip_length, args.batch_size)

# "/media/ubuntu/TrekStor/TESI_ANOMALY_DETECTION/Datasets/videos/ShanghaiTech_new_split/test"
# python3 extract_features.py --dataset_path=/home/ubuntu/Downloads/UCF-Crime/train --output_path=output
# python3 extract_features.py --dataset_path=ShanghaiTech_new_split/train/ --output_path=output
