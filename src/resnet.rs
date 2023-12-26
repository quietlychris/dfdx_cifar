use dfdx::prelude::*;

#[derive(Default, Clone, Sequential)]
pub struct BasicBlockInternal<const C: usize> {
    conv1: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn1: BatchNorm2DConstConfig<C>,
    relu: ReLU,
    conv2: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<C>,
}

#[derive(Default, Clone, Sequential)]
pub struct DownsampleA<const C: usize, const D: usize> {
    conv1: Conv2DConstConfig<C, D, 3, 2, 1>,
    bn1: BatchNorm2DConstConfig<D>,
    relu: ReLU,
    conv2: Conv2DConstConfig<D, D, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<D>,
}

#[derive(Default, Clone, Sequential)]
pub struct DownsampleB<const C: usize, const D: usize> {
    conv1: Conv2DConstConfig<C, D, 1, 2, 0>,
    bn1: BatchNorm2DConstConfig<D>,
}

pub type BasicBlock<const C: usize> = ResidualAdd<BasicBlockInternal<C>>;

pub type Downsample<const C: usize, const D: usize> =
    GeneralizedAdd<DownsampleA<C, D>, DownsampleB<C, D>>;

#[derive(Default, Clone, Sequential)]
pub struct Head {
    conv: Conv2DConstConfig<3, 64, 7, 2, 3>,
    bn: BatchNorm2DConstConfig<64>,
    relu: ReLU,
    pool: MaxPool2DConst<3, 2, 1>,
}

#[derive(Default, Clone, Sequential)]
#[built(Resnet18)]
pub struct Resnet18Config<const NUM_CLASSES: usize> {
    head: Head,
    l1: (BasicBlock<64>, ReLU, BasicBlock<64>, ReLU),
    l2: (Downsample<64, 128>, ReLU, BasicBlock<128>, ReLU),
    l3: (Downsample<128, 256>, ReLU, BasicBlock<256>, ReLU),
    // l4: (Downsample<256, 512>, ReLU, BasicBlock<512>, ReLU),
    l5: (AvgPoolGlobal, LinearConstConfig<256, NUM_CLASSES>),
}

//------------ SMALL RESNET ----------------

/* #[derive(Default, Clone, Sequential)]
pub struct ResidualBlock<const C: usize, const D: usize> (
    (Conv2D<C, D, 3, 1, 1>, BatchNorm2D<D>, MaxPool2D<3>, ReLU),
    Residual<(Conv2D<D, D, 3, 1, 1>, BatchNorm2D<D>, ReLU)>,
);


pub struct BlockA<const C: usize, const D: usize> {
    conv1: Conv2DConstConfig<C, D, 3, 1, 1>,
    bn: BatchNorm2DConstConfig<D>,
    mp: MaxPool2DConst<3>,
    relu: ReLU,

}

#[derive(Default, Clone, Sequential)]
#[built(SmallResnet)]
pub struct SmallResnet<const NUM_CLASSES: usize> = (
    (Conv2D<3, 32, 3>, BatchNorm2D<32>, ReLU, MaxPool2D<3>),
    ResidualBlock<32, 64>,
    ResidualBlock<64, 128>,
    ResidualBlock<128, 256>,
    (AvgPoolGlobal, Linear<256, NUM_CLASSES>),
); */
