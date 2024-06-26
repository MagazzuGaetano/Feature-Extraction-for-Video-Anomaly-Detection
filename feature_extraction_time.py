import argparse
import torch
import numpy as np

from src.models.c3d import c3d_model
from src.models.i3d_inception import i3d_model

from src.utils.transforms import (
    transform_clip_from_frames_c3d,
    transform_clip_from_frames_i3d,
)
from src.utils.video import from_frames_to_clips, read_video

# USAGE: python3 feature_extraction_time.py --video_path "path/to/test_video.mp4"
parser = argparse.ArgumentParser()
parser.add_argument("--feature_type", type=str, default="I3D")
parser.add_argument("--patch_size", type=int, default=224)
parser.add_argument("--n_crops", type=int, default=10)
parser.add_argument("--clip_step", type=int, default=16)
parser.add_argument("--video_path", type=str, required=True)
args = parser.parse_args()

FEATURE_TYPE = args.feature_type
PATCH_SIZE = args.patch_size
N_CROPS = args.n_crops
CLIP_STEP = args.clip_step
VIDEO_PATH = args.video_path

# extract frames from the video and split into clips
frames = read_video(
    VIDEO_PATH, use_rgb=True if FEATURE_TYPE == "I3D" else False, transform_frame=None
)
frames = np.stack(frames)
clip_indices = from_frames_to_clips(frames, step=CLIP_STEP)
n_clips = len(clip_indices)

# set up the model
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
if FEATURE_TYPE == "I3D":
    model = i3d_model(nb_classes=400, pretrainedpath="models/rgb_imagenet.pt")
else:
    model = c3d_model(
        nb_classes=487, pretrainedpath="models/c3d.pickle", feature_layer=6
    )

model.to(device)  # Put model to GPU
model.train(False)  # Set model to evaluate mode


# INIT LOGGERS
starter, ender = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
timings = np.zeros((len(clip_indices), 1))

# GPU-WARM-UP
dummy_input = torch.randn(
    1, 3, CLIP_STEP, PATCH_SIZE, PATCH_SIZE, dtype=torch.float
).to(device)
for _ in range(10):
    _ = model({"frames": dummy_input})

# MEASURE PERFORMANCE
with torch.no_grad():
    # iterate over each clip
    for i in range(n_clips):
        curr_clip = frames[clip_indices[i].tolist()]

        if FEATURE_TYPE == "I3D":
            processed_clip = transform_clip_from_frames_i3d(
                curr_clip, PATCH_SIZE, N_CROPS
            )
        else:
            processed_clip = transform_clip_from_frames_c3d(
                curr_clip, PATCH_SIZE, N_CROPS
            )

        starter.record()

        processed_clip = processed_clip.unsqueeze(0)
        processed_clip = processed_clip.to(device)

        # extract clip-level features
        # I3D shape with ten crops: ([1, 16, 10, 3, 224, 224])
        # I3D shape just one crop: (1, 3, 16, 224, 224)
        for j in range(processed_clip.shape[2]):
            input = processed_clip[:, :, j, :, :, :]
            input = input.transpose(1, 2)
            features = model({"frames": input}).squeeze()

        ender.record()

        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

mean_syn = np.mean(timings)
std_syn = np.std(timings)
print("Feature Extraction Time: {:.2f} ± {:.2f} s".format(mean_syn, std_syn))
