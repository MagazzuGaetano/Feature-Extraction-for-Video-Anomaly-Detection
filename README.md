# Activity-Recognition-Feature-Extraction-for-W-VAD
This repo contains code to extract the I3D / C3D features for the task of Weakly Video Anomaly Detection with resnet50 backbone given a folder of videos

## Credits
The main resnet code and others is collected from the following repository.
* [I3D_Feature_Extraction_resnet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet/edit/main/README.md)

## Overview
This code takes a folder of videos as input and for each video it saves ```I3D``` feature numpy file of dimension ```1*n/16*2048``` where n is the no.of frames in the video

## Usage
### Setup
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
--frequency:         how many frames between adjacent snippet
--batch_size:        batch size for snippets
</pre>

### Run
```bash
python main.py --datasetpath=samplevideos/ --outputpath=output
```
