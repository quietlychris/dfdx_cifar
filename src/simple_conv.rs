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


#[derive(Default, Clone, Sequential)]
#[built(SimpleConv)]
pub struct ConvNetworkConfig<const C: usize, const NUM_CLASSES: usize> {
    conv1: Conv2DConstConfig<C, C, 1, 6, 5>,
    pool: MaxPool2DConst<1, 2, 2>,
    conv2: Conv2DConstConfig<C, C, 2, 16, 5>,
    flatten: Flatten2D,
    fc1: LinearConstConfig<400, 120>,
    fc2: LinearConstConfig<120, 84>,
    fc3: LinearConstConfig<84, NUM_CLASSES>
}

/* 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
*/

