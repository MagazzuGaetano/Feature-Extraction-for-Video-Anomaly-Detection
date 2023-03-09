# Feature-Extraction-for-Video-Anomaly-Detection
This repo contains code to extract the C3D / I3D features for the task of Weakly Video Anomaly Detection given a folder of videos

## Credits
The main resnet code and others is collected from the following repository.
* [c3d-pytorch](https://github.com/DavideA/c3d-pytorch)
* [I3D_Feature_Extraction_resnet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet/edit/main/README.md)

## Overview
This code takes a folder of videos as input and for each video it saves ```C3D``` features the numpy file have dimension ```int(n/16) + 1 * 10 * 4096``` where n is the no.of frames in the video, for each frame 10 crops are taken and 4096 is the fc6 dimension.

For ```I3D``` feature numpy file of dimension ```int(n/16) + 1 * 10 * 2048``` where 2048 is the mix_5c dimension.


### C3D Setup
* Download the pretrained weights (Sports1M) from [here](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle).

### I3D Setup
Download pretrained weights for I3D from the nonlocal repo
```bash
wget https://dl.fbaipublicfiles.com/video-nonlocal/i3d_baseline_32x2_IN_pretrain_400k.pkl -P pretrained/
```
Convert these weights from caffe2 to pytorch. This is just a simple renaming of the blobs to match the pytorch model.
```bash
python -m utils.convert_weights pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl pretrained/i3d_r50_kinetics.pth
```

### Parameters
<pre>
--datasetpath:       folder of input videos (contains videos or subdirectories of videos)
--outputpath:        folder of extracted features
--feature:           C3D or I3D
--clip_length        number of frames in a clip
--batch_size:        batch size for clips
</pre>

### Run
```bash
python main.py --datasetpath=samplevideos/ --outputpath=output
```
