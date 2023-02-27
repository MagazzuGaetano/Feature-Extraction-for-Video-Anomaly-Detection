import os
from pathlib import Path
import shutil
import argparse
from natsort import natsorted
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from models.resnet import i3_res50
from utils.utils import from_frames_to_clips, load_rgb_batch, split_clip_indices_into_batches



def extract_features(i3d, clip_size, frequency, temppath, batch_size):

    # rgb files
    rgb_files = natsorted([i for i in os.listdir(temppath)])

    # creating clips
    frame_indices = from_frames_to_clips(rgb_files)

    # divide clips into batches (for memory optimization)
    # a batch of dimension 16 it's equal to a 16-frames clip
    frame_indices = split_clip_indices_into_batches(frame_indices, batch_size)

    # iterate 10-crop augmentation
    full_features = [[] for i in range(10)]
    number_of_batches = frame_indices.shape[0]

    for batch_id in range(number_of_batches):

        # B=16 x T=16 x CROP=10 x CH=3 x H=224 x W=224
        batch_data = load_rgb_batch(temppath, rgb_files, frame_indices[batch_id]) 

        # iterate 10-crop augmentation
        for i in range(batch_data.shape[2]):
            b_data = batch_data[:,:,i,:,:,:] # select the i-crop
            b_data = torch.from_numpy(b_data)
            b_data = b_data.transpose(1, 2) # B=16 x CH=3 x T=16 x H=224 x W=224

            with torch.no_grad():
                b_data = Variable(b_data.cuda()).float()
                inp = {'frames': b_data}
                features = i3d(inp)

            v = features.cpu().numpy() # single clip features extracted
            v = v / np.sqrt(np.sum(v**2)) # single clip features normalized (L2)
            full_features[i].append(v)

    full_features = [np.concatenate(i, axis=0) for i in full_features]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)
    full_features = full_features[:,:,:,0,0,0]
    full_features = np.array(full_features).transpose([1,0,2])

    return full_features


def generate(datasetpath, outputpath, pretrainedpath, frequency, clip_size, batch_size):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = outputpath+ "/temp/"
    rootdir = Path(datasetpath)
    #videos = [str(f) for f in rootdir.glob('**/*.mp4')]
    videos = [str(f) for f in rootdir.glob('**/*.avi')]

    # setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    for video in videos:
        # create temp folder
        Path(temppath).mkdir(parents=True, exist_ok=True)

        # splitting the video into frames
        ffmpeg.input(video).output('{}%d.jpg'.format(temppath),start_number=0).global_args('-loglevel', 'quiet').run()

        # extract features
        features = extract_features(i3d, clip_size, frequency, temppath, batch_size)

        # save features
        videoname = video.split("/")[-1].split(".")[0]
        np.save(outputpath + "/" + videoname, features)



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="samplevideos/")
    parser.add_argument('--outputpath', type=str, default="output")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--clip_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.clip_size, args.batch_size)
