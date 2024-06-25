import torch
import numpy as np

from src.models.c3d import c3d_model
# from src.models.i3d_inception import i3d_model


from src.utils.video import from_frames_to_clips, read_video


# extract frames from the video and split into clips
video_path = "/media/ubuntu/Volume/datasets/ShanghaiTech_new_split/test/01_001.avi"
frames = np.stack(read_video(video_path, None))
clip_indices = from_frames_to_clips(frames)
n_clips = len(clip_indices)

print(len(frames), n_clips)

# set up the model
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = i3d_model(nb_classes=400, pretrainedpath='models/rgb_imagenet.pt')
model = c3d_model(nb_classes=487, pretrainedpath="models/c3d.pickle", feature_layer=6)
model.to(device)  # Put model to GPU
model.train(False)  # Set model to evaluate mode

# INIT LOGGERS
starter, ender = (
    torch.cuda.Event(enable_timing=True),
    torch.cuda.Event(enable_timing=True),
)
timings = np.zeros((len(clip_indices), 1))

# GPU-WARM-UP
dummy_input = torch.randn(1, 3, 16, 112, 112, dtype=torch.float).to(device)
for _ in range(10):
    _ = model({"frames": dummy_input})

# MEASURE PERFORMANCE
with torch.no_grad():
    # scorro le clip
    for i in range(n_clips):
        # seleziono la clip corrente shape = (16, 10, 3, 112, 112)
        curr_clip = frames[clip_indices[i].tolist()]

        # clip preprocessing
        #processed_clip = 

        starter.record()

        # estraggo le feature per una clip (1, 3, 16, 112, 112)
        for j in range(10):
            input = torch.from_numpy(curr_clip[:, j, :, :, :])  # (16, 3, 112, 112)
            input = input.transpose(0, 1)  # (3, 16, 112, 112)
            input = input.unsqueeze(0).to(device)  # (1, 3, 16, 112, 112)
            features = model({"frames": input}).squeeze()

        ender.record()

        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

mean_syn = np.mean(timings)
std_syn = np.std(timings)
print(mean_syn, std_syn)
