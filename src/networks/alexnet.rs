use dfdx::prelude::*;
// https://www.kaggle.com/code/drvaibhavkumar/alexnet-in-pytorch-cifar10-clas-83-test-accuracy


#[derive(Default, Clone, Sequential)]
#[built(AlexNet)]
pub struct AlexNetConfig<const NUM_CLASSES: usize> {
    // Conv2DConstConfig<INPUT_CHANNELS (3 for RGB), 1, 3>
    // 3072 / 3 = 1024 * 1 * 1 = 1024; 3072 / 3 = 1024 * 2 * 1 = 2048
    // Conv2DConstConfig<IN_CHAN,OUT_CHAN,KERNEL_SIZE,STRIDE,PADDING,DILATION,GROUPS>
    conv1: Conv2DConstConfig<3, 64, 11, 4, 2>,
    relu1: ReLU,
    // MaxPool2DConst<KERNEL_SIZE,STRIDE,PADDING, DILATION>
    mp1: MaxPool2DConst<3, 2, 0, 1>,
    conv2: Conv2DConstConfig<64, 192, 5, 2, 1>,
    relu2: ReLU,
    mp2: MaxPool2DConst<3, 2, 0, 1>,
    conv3: Conv2DConstConfig<192, 384, 3, 1, 1>,
    relu3: ReLU,
    conv4: Conv2DConstConfig<384, 256, 3, 1, 1>,
    relu4: ReLU,
    conv5: Conv2DConstConfig<256, 256, 3, 1, 1>,
    relu5: ReLU,
    mp3: MaxPool2DConst<3, 2>,
    // avg: AvgPool2DConst<6>,
    dropout: Dropout,
    flatten: Flatten2D,
    
    classifier: Classifier<NUM_CLASSES>
    // softmax: Softmax
}

#[derive(Default, Clone, Sequential)]
struct Classifier<const NUM_CLASSES: usize> {
    fc1: LinearConstConfig<9216, 4096>,
    relu6: ReLU,
    fc2: LinearConstConfig<4096, 1024>,
    relu7: ReLU,
    fc3: LinearConstConfig<1024, NUM_CLASSES>
}

/* 



AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

import torch.nn as nn
AlexNet_Model.classifier[1] = nn.Linear(9216,4096)
AlexNet_Model.classifier[4] = nn.Linear(4096,1024)
AlexNet_Model.classifier[6] = nn.Linear(1024,10)

*/