
use dfdx::prelude::*;

// dfdx
// Conv2DConstConfig<CHANNEL_IN, CHANNEL_OUT, STRIDE, PADDING, DILATION>
// conv1: Conv2DConstConfig<C, D, 3, 2, 1>,

// tch-rs
// conv_bn(vs: &nn::Path, c_in: i64, c_out: i64)


/* #[derive(Default, Clone, Sequential)]
pub struct ConvBN<const C_IN: usize, const C_OUT: usize> {
    conv1: Conv2DConstConfig<C_IN, C_OUT, 3, 1, 1>,
    bn1: BatchNorm2DConstConfig<C_OUT>,
    relu: ReLU,
    conv2: Conv2DConstConfig<C_OUT, C_OUT, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<C_OUT>,
} */

#[derive(Default, Clone, Sequential)]
pub struct ConvBN<const C: usize> {
    conv1: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn1: BatchNorm2DConstConfig<C>,
    relu: ReLU,
    conv2: Conv2DConstConfig<C, C, 3, 1, 1>,
    bn2: BatchNorm2DConstConfig<C>,
}

#[derive(Default, Clone, Sequential)]
#[built(FastResNet)]
pub struct FastResNetConfig<const NUM_CLASSES: usize> {
    pre: ConvBN<
    layer1:
    inter:
    
}

/*
#[derive(Default, Clone, Sequential)]
#[built(SimpleConv)]
pub struct SimpleConvConfig<const NUM_CLASSES: usize> {
    // Conv2DConstConfig<INPUT_CHANNELS (3 for RGB), 1, 3>
    // 3072 / 3 = 1024 * 1 * 1 = 1024; 3072 / 3 = 1024 * 2 * 1 = 2048
    conv1: Conv2DConstConfig<3, 6, 5>,
    mp: MaxPool2DConst<2, 2>,
    conv2: Conv2DConstConfig<6, 16, 5>,
    flatten: Flatten2D,
    fc1: LinearConstConfig<1600, 120>,
    fc2: LinearConstConfig<120, 84>,
    fc3: LinearConstConfig<84, NUM_CLASSES>,
    // softmax: Softmax
}
 */