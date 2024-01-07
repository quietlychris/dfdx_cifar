use dfdx::prelude::*;

/* struct ResidualBlock<const C: usize, const D: usize> = (
    (Conv2D<C, D, 3, 1, 1>, BatchNorm2D<D>, MaxPool2D<3>, ReLU),
    Residual<(Conv2D<D, D, 3, 1, 1>, BatchNorm2D<D>, ReLU)>,
);

struct SmallResnet<const NUM_CLASSES: usize> = (
    (Conv2D<3, 32, 3>, BatchNorm2D<32>, ReLU, MaxPool2D<3>),
    ResidualBlock<32, 64>,
    ResidualBlock<64, 128>,
    ResidualBlock<128, 256>,
    (AvgPoolGlobal, Linear<256, NUM_CLASSES>),
); */

/* GOOD!
// Conv2DConstConfig<INPUT_CHANNELS (3 for RGB), 1, 3>
// 3072 / 3 = 1024 * 1 * 1 = 1024; 3072 / 3 = 1024 * 2 * 1 = 2048
// conv1: Conv2DConstConfig<3, 2, 1>,
*/

// Mirroring https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#[derive(Default, Clone, Sequential)]
#[built(SimpleConv)]
pub struct SimpleConvConfig<const NUM_CLASSES: usize> {
    // Conv2DConstConfig<INPUT_CHANNELS (3 for RGB), 1, 3>
    // 3072 / 3 = 1024 * 1 * 1 = 1024; 3072 / 3 = 1024 * 2 * 1 = 2048
    conv1: Conv2DConstConfig<3, 6, 5>,
    relu1: ReLU,
    mp: MaxPool2DConst<2, 2>,
    conv2: Conv2DConstConfig<6, 16, 5>,
    relu2: ReLU,
    flatten: Flatten2D,
    fc1: LinearConstConfig<1600, 120>,
    dp1: Dropout,
    fc2: LinearConstConfig<120, 84>,
    dp2: Dropout,
    fc3: LinearConstConfig<84, NUM_CLASSES>,
    // softmax: Softmax
}

/*
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
*/
