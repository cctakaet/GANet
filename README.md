# GANet

[GA-Net: Guided Aggregation Net for End-to-end Stereo Matching](https://arxiv.org/pdf/1904.06587.pdf)

# Building Environment
gcc: >= 5.3

GPU memory: >= 6.5G (for testing)

Pytorch: >= 1.0

**Cuda: >=9.2** 

My platform/settings: ubuntu 18.04 + cuda 10.1 + python 3.6

Requirements
* pip install torch torchvision
* pip install scikit-image
* pip install opencv-python\==3.4.2.17 opencv-contrib-python==3.4.2.17

# Compile

    sh compile.sh
Download pre-trained weight and compile

# Run
With GPU:

python main.py --input-left="./data/Synthetic/TL0.png" --input-right="./data/Synthetic/TR0.png" --output="./result/Synthetic/TL0.pfm"

Without GPU:

python main.py --input-left="./data/Synthetic/TL0.png" --input-right="./data/Synthetic/TR0.png" --output="./result/Synthetic/TL0.pfm" --cuda False

* -\-input-left "path to left image"
* -\-input-right "path to right image"
* -\-output "path to output PFM file"
* -\-cuda use GPU or not (default=True)


## Reference:
    @inproceedings{Zhang2019GANet,
      title={GA-Net: Guided Aggregation Net for End-to-end Stereo Matching},
      author={Zhang, Feihu and Prisacariu, Victor and Yang, Ruigang and Torr, Philip HS},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={185--194},
      year={2019}
    }
