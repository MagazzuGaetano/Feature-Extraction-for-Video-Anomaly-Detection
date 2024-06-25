import os
from pathlib import Path
import argparse

from natsort import natsorted
import numpy as np
import torch

from src.models.i3d_inception import i3d_model
from src.models.c3d import c3d_model
from src.utils.video import (
    from_frames_to_clips,
    load_rgb_batch,
    split_clip_indices_into_batches,
    extract_frames_from_video,
)
from time import perf_counter


def extract_features(
    model, feature, device, clip_length, temp_path, batch_size, n_crops
):
    # rgb files
    rgb_files = natsorted(
        [i for i in os.listdir(temp_path)]
    )  # [:480000] # only for ucf-crime long videos

    # creating clips
    frame_indices = from_frames_to_clips(rgb_files, clip_length)

    # divide clips into batches (for memory optimization)
    # k = true values in the last batch
    frame_indices, k = split_clip_indices_into_batches(frame_indices, batch_size)

    # iterate N-crop augmentation
    n_batches = frame_indices.shape[0]
    n_clips_inside_a_batch = frame_indices.shape[1]

    # 1024 if inception-v1 or 2048 if resnet50 and 4096 if C3D
    feat_vec_dim = 1024 if feature == "I3D" else 4096
    full_features = torch.zeros(
        (n_batches, n_crops, n_clips_inside_a_batch, feat_vec_dim), device=device
    )

    for batch_id in range(n_batches):
        # B=16 x T=16 x CROP=10 x CH=3 x H x W
        batch_data = load_rgb_batch(
            temp_path, rgb_files, frame_indices[batch_id], feature, device, n_crops
        )

        # iterate 10-crop augmentation
        n_crops = batch_data.shape[2]
        for i in range(n_crops):
            b_data = batch_data[:, :, i, :, :, :]  # select the i-crop
            b_data = b_data.transpose(1, 2)  # B=16 x CH=3 x T=16 x H x W

            with torch.no_grad():
                inp = {"frames": b_data}
                features = model(inp).squeeze()

            full_features[batch_id, i, :, :] = features

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

    out = np.array(out).transpose([1, 0, 2])
    return out


def generate(
    dataset_path,
    output_path,
    feature,
    clip_size,
    batch_size,
    video_format_type,
    n_crops,
    save_single_crops,
):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    temp_path = output_path + "/temp/"
    root_dir = Path(dataset_path)
    videos = [str(f) for f in root_dir.glob("**/*{}".format(video_format_type))]

    # set up the model
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if feature == "I3D":
        model = i3d_model(nb_classes=400, pretrainedpath="models/rgb_imagenet.pt")
    else:
        model = c3d_model(
            nb_classes=487, pretrainedpath="models/c3d.pickle", feature_layer=6
        )

    model.to(device)  # Put model to GPU
    model.train(False)  # Set model to evaluate mode

    # create temp folder
    Path(temp_path).mkdir(parents=True, exist_ok=True)

    # list of corrupted files to discard
    discard_list = ["v=8cTqh9tMz_I__#1_label_A"]

    avg_frames_extraction_time = []  # list for estimate avg frames extraction time
    avg_features_extraction_time = []  # list for estimate avg features extraction time
    avg_frame_res = []  # list for estimate avg frame resolution

    k = 1
    for video in videos:
        # remove files into temp folder
        for f in Path(temp_path).glob("**/*"):
            os.remove(f)

        # discard files mentioned in the discard_list
        if video.split("/")[-1].split(video_format_type)[0] in discard_list:
            continue

        print(f"current video: {video}")

        # start the stopwatch to mesure the frames extraction time
        t1_start = perf_counter()

        # splitting the video into frames and saving them as images to an output folder
        frame_count, frame_res = extract_frames_from_video(video, temp_path)

        t1_elapsed = perf_counter() - t1_start
        avg_frames_extraction_time.append(t1_elapsed)
        avg_frame_res.append(frame_res)
        print(
            f"{frame_count} frames extracted with shape {frame_res}, time elapsed: {t1_elapsed:.2f} s"
        )

        # start the stopwatch to mesure the features extraction time
        t2_start = perf_counter()

        # extract features from the extracted frames of a video
        features = extract_features(
            model, feature, device, clip_size, temp_path, batch_size, n_crops
        )

        t2_elapsed = perf_counter() - t2_start
        avg_features_extraction_time.append(t2_elapsed)
        print(
            f"features extracted with shape {features.shape}, time elapsed: {t2_elapsed:.2f} s"
        )

        # save features
        video_name = video.split("/")[-1].split(video_format_type)[0]
        if save_single_crops:
            for i in range(n_crops):
                feature_filename = output_path + "/" + video_name + "__" + str(i)
                np.save(feature_filename, features[:, i, :])
                print("features saved as {}".format(feature_filename + ".npy"))
        else:
            feature_filename = output_path + "/" + video_name
            np.save(feature_filename, features)
            print("features saved as {}".format(feature_filename + ".npy"))

        total_elapsed = t1_elapsed + t2_elapsed
        print(
            f"videos processed: {k} / {len(videos)}, time elapsed: {total_elapsed:.2f} s \n"
        )
        k += 1

    avg_frame_res = np.asarray(avg_frame_res).mean(0)
    avg_frames_extraction_time = np.asarray(avg_frames_extraction_time).mean()
    avg_features_extraction_time = np.asarray(avg_features_extraction_time).mean()

    print(f"avg frame resolution: {avg_frame_res} s")
    print(f"avg frames extraction time: {avg_frames_extraction_time} s")
    print(f"avg features extraction time: {avg_features_extraction_time} s")


# dataset-path:
# /media/ubuntu/Volume/datasets/XD-Violence/train
# /media/ubuntu/Volume/datasets/UCF-Crime/train
# /media/ubuntu/Volume/datasets/ShanghaiTech_new_split/train

# output-path: /media/ubuntu/A020C22220C1FEF2/prova/output
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/ubuntu/Volume/datasets/XD-Violence/test",
    )
    parser.add_argument(
        "--output_path", type=str, default="/home/ubuntu/Desktop/output_fake"
    )
    parser.add_argument("--feature", type=str, default="C3D")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--video_format_type", type=str, default=".mp4")
    parser.add_argument("--n_crops", type=int, default=10)  # 10, 5
    parser.add_argument("--save_single_crops", type=bool, default=False)  # False, True
    args = parser.parse_args()
    generate(
        args.dataset_path,
        str(args.output_path),
        args.feature,
        args.clip_length,
        args.batch_size,
        args.video_format_type,
        args.n_crops,
        args.save_single_crops,
    )
