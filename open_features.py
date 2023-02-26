import numpy as np
import matplotlib.pyplot as plt

file1 = np.load('/home/ubuntu/Desktop/I3D_Feature_Extraction_resnet/output/Explosion036_x264.npy')
file1 = file1.astype('float32')
print(np.asarray(file1).shape)


file2 = np.load('/home/ubuntu/Desktop/RTFM/features/UCF_Test_ten_crop_i3d(corrupted)/Explosion036_x264_i3d.npy')
file2 = file2.astype('float32')
print(np.asarray(file2).shape)


print(file1[0, 0, :50])

print(file2[0, 0, :50])

