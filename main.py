import os
from pathlib import Path
import argparse
from natsort import natsorted
import numpy as np
import ffmpeg
import torch
from models.resnet import i3_res50
from utils.utils import from_frames_to_clips, load_rgb_batch, split_clip_indices_into_batches


def extract_features(i3d, device, clip_length, temp_path, batch_size):
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
    full_features = torch.zeros((n_batches, 10, n_clips_inside_a_batch, 2048, 1, 1, 1), device=device)
    for batch_id in range(n_batches):

        # B=16 x T=16 x CROP=10 x CH=3 x H=224 x W=224
        batch_data = load_rgb_batch(temp_path, rgb_files, frame_indices[batch_id], device)

        # iterate 10-crop augmentation
        n_crops = batch_data.shape[2]
        for i in range(n_crops):
            b_data = batch_data[:, :, i, :, :, :]  # select the i-crop
            b_data = b_data.transpose(1, 2)  # B=16 x CH=3 x T=16 x H=224 x W=224

            with torch.no_grad():
                inp = {'frames': b_data}
                features = i3d(inp)

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
                out[i].append(full_features[batch_id, i, :k, :, :, :, :])
            else:
                out[i].append(full_features[batch_id, i, :, :, :, :, :])

    out = [np.concatenate(i, axis=0) for i in out]
    out = [np.expand_dims(i, axis=0) for i in out]
    out = np.concatenate(out, axis=0)
    out = out[:, :, :, 0, 0, 0]
    out = np.array(out).transpose([1, 0, 2])

    return out


def generate(dataset_path, output_path, pretrained_path, clip_size, batch_size):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    temp_path = output_path + "/temp/"
    root_dir = Path(dataset_path)
    # videos = [str(f) for f in root_dir.glob('**/*.mp4')]
    videos = [str(f) for f in root_dir.glob('**/*.avi')][:3]

    # set up the model
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i3d = i3_res50(400, pretrained_path)
    i3d.to(device)
    i3d.train(False)  # Set model to evaluate mode

    for video in videos:
        frames_path = video.split(dataset_path)[1].split('.')[0]
        frames_path = os.path.join('/home/ubuntu/Desktop/tmp', frames_path)

        # remove files into temp folder
        # for f in Path(temp_path).glob('**/*'):
        #    os.remove(f)

        # create temp folder
        # Path(temp_path).mkdir(parents=True, exist_ok=True)

        # splitting the video into frames
        # ffmpeg.input(video).output('{}%d.jpg'.format(temp_path), start_number=0).global_args('-loglevel', 'quiet').run()

        # extract features
        features = extract_features(i3d, device, clip_size, frames_path, batch_size)

        # save features
        video_name = video.split("/")[-1].split(".")[0]
        np.save(output_path + "/" + video_name, features)

        print('features of dimension {} extracted from {}'.format(features.shape, video))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="ShanghaiTech_new_split/test/")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--pretrained_path', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--clip_length', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate(args.dataset_path, str(args.output_path), args.pretrained_path, args.clip_length, args.batch_size)

# python3 main.py --dataset_path=UCF-Crime-Test/ --output_path=output
# python3 main.py --dataset_path=ShanghaiTech_new_split/test/ --output_path=output
