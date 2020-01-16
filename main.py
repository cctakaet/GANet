from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.GANet_deep import GANet
from dataloader.data import get_test_set
import numpy as np
import cv2

import time
from util import writePFM

# Training settings
crop_height = 528
crop_width = 384
max_disp = 12 * 5
resume = './checkpoint/kitti2015_final.pth'
model = 'GANet_deep'

if model == 'GANet11':
    from models.GANet11 import GANet
elif model == 'GANet_deep':
    from models.GANet_deep import GANet
else:
    raise Exception("No suitable model found ...")

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def histEqual(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def load_data(leftname, rightname):
    #left = Image.open(leftname)
    #right = Image.open(rightname)
    
    left = cv2.imread(leftname)
    right = cv2.imread(rightname)
    

    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def check_file(left, right):
    surf = cv2.xfeatures2d.SURF_create(1000)
    bf = cv2.BFMatcher()

    left_kp, left_des = surf.detectAndCompute(left,None)
    right_kp, right_des = surf.detectAndCompute(right,None)
    matches = bf.knnMatch(left_des, right_des, k=2)

    goodmatches = []
    for (m, n) in matches:
        if m.distance < 0.75 * n.distance:
            goodmatches.append(m)

    dis = []
    for matches in goodmatches:
        left_pt = left_kp[matches.queryIdx].pt
        right_pt = right_kp[matches.trainIdx].pt
        dis.append(left_pt[0] - right_pt[0])
    dis = np.sort(np.array(dis))
    min_disp, max_disp = 0, 50
    for i in range(len(dis)):
        if dis[i] > -60 and dis[i+1] - dis[i] < 2 and dis[i+2] - dis[i] < 2:
            min_disp = dis[i]
            break

    for i in range(len(dis)):
        if dis[-1 - i] < 60 and dis[-1 - i] - dis[-1 - (i + 1)] < 2 and dis[-1 - i] - dis[-1 - (i + 2)] < 2:
            max_disp = dis[-1 - i]
            break
    output = int(max(abs(min_disp), abs(max_disp)))
    output += 1
    return output

def test(leftname, rightname, cuda):
  #  count=0
    
    input1, input2, height, width = test_transform(load_data(leftname, rightname), crop_height, crop_width)
    
    img_left = cv2.imread(leftname)
    img_right = cv2.imread(rightname)
    max_disp = check_file(img_left, img_right)
    if max_disp%12 != 0:
        max_disp = max_disp + 12 - max_disp%12
    if max_disp >= 30:
        max_disp = 192
    print('start')

    print('===> Building model')
    model = GANet(max_disp)

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --cuda=False")

    if cuda:
        model = torch.nn.DataParallel(model).cuda()

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(resume))
    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)
     
    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= crop_height and width <= crop_width:
        temp = temp[0, crop_height - height: crop_height, crop_width - width: crop_width]
    else:
        temp = temp[0, :, :]
    
    synth = max_disp
    if synth == 192:
        temp = cv2.ximgproc.weightedMedianFilter(joint=img_left, src=temp, r=5)

    disp = temp
    max_disp = np.nanmax(disp[disp != np.inf])
    min_disp = np.nanmin(disp[disp != np.inf])
    disp_normalized = (disp - min_disp) / (max_disp - min_disp)
    PFM = disp_normalized

    # Jet color mapping
    disp_normalized = (disp_normalized * 255.0).astype(np.uint8)
    disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)

    cv2.imwrite('C_'+leftname[-5:], disp_normalized)
    return PFM

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./result/Synthetic/TL0.pfm', type=str, help='left disparity map')
parser.add_argument('--cuda', default=True, type=str, help='left disparity map')

if __name__ == "__main__":

    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    left = args.input_left
    right = args.input_right
    cuda = args.cuda

    tic = time.time()
    disp = test(left, right, cuda)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))
