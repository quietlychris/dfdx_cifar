#![feature(generic_const_exprs)]

use cifar_ten::*;
use std::time::Instant;

use dfdx::{data::*, optim::Sgd, prelude::*};
use indicatif::ProgressIterator;
use rand::prelude::*;

use ndarray::prelude::*;

type Dev = Cpu;
type Dtype = f32;

type ResidualBlock<const C: usize, const D: usize> = (
    (Conv2D<C, D, 3, 1, 1>, BatchNorm2D<D>, MaxPool2D<3>, ReLU),
    Residual<(Conv2D<D, D, 3, 1, 1>, BatchNorm2D<D>, ReLU)>,
);

type SmallResnet<const NUM_CLASSES: usize> = (
    (Conv2D<3, 32, 3>, BatchNorm2D<32>, ReLU, MaxPool2D<3>),
    ResidualBlock<32, 64>,
    ResidualBlock<64, 128>,
    ResidualBlock<128, 256>,
    (AvgPoolGlobal, Linear<256, NUM_CLASSES>),
);

const BATCH_SIZE: usize = 10_000;
const IMAGE_LENGTH: usize = 3 * 32 * 32;
const TEST_NUM: usize = 100;

fn main() {
    let result = Cifar10::default()
        .download_and_extract(true)
        .encode_one_hot(true)
        .build()
        .unwrap();

    let mut rng = StdRng::seed_from_u64(0);
    let dev: Dev = Default::default();

    let mut model = dev.build_module::<SmallResnet<10>, Dtype>();
    let mut grads = model.alloc_grads();
    let mut opt = Sgd::new(&model, Default::default());

    let train_data: Vec<f32> = result.0.iter().map(|x| *x as f32).collect();
    let train_labels: Vec<f32> = result.1.iter().map(|x| *x as f32).collect();
    let test_data: Vec<f32> = result.2.iter().map(|x| *x as f32).collect();
    let test_labels: Vec<f32> = result.3.iter().map(|x| *x as f32).collect();

    let train_imgs = dev.tensor_from_vec(
        train_data[0..(BATCH_SIZE * IMAGE_LENGTH)].to_vec(),
        (Const::<BATCH_SIZE>, Const::<3>, Const::<32>, Const::<32>),
    );

    let train_lbls = dev.tensor_from_vec(
        train_labels[0..(BATCH_SIZE * 10)].to_vec(),
        (Const::<BATCH_SIZE>, Const::<10>),
    );

    for i in 0..BATCH_SIZE {
        let start = Instant::now();

        let img: Tensor<(Const<3>, Const<32>, Const<32>), f32, Cpu> =
            train_imgs.clone().select(dev.tensor(i));
        let lbl = train_lbls.clone().select(dev.tensor(i));

        let logits = model.forward_mut(img.traced(grads));
        let loss = cross_entropy_with_logits_loss(logits, lbl);
        dev.synchronize();
        let fwd_dur = start.elapsed();
        let loss_val = loss.array();

        let start = Instant::now();
        grads = loss.backward();
        dev.synchronize();
        let bwd_dur = start.elapsed();

        let start = Instant::now();
        opt.update(&mut model, &grads).unwrap();
        model.zero_grads(&mut grads);
        dev.synchronize();
        let opt_dur = start.elapsed();
        dbg!(opt_dur);
    }

    let test_imgs = dev.tensor_from_vec(
        test_data[0..IMAGE_LENGTH * TEST_NUM].to_vec(),
        (Const::<TEST_NUM>, Const::<3>, Const::<32>, Const::<32>),
    );
    dbg!(test_imgs.shape());

    let test_lbls = dev.tensor_from_vec(
        test_labels[0..TEST_NUM * 10].to_vec(),
        (Const::<TEST_NUM>, Const::<10>),
    );

    // let img: Tensor<(Const<3>, Const<32>, Const<32>), f32, Cpu> = test_imgs.select(dev.tensor(1));

    let mut num_correct = 0.0;
    let mut num_total = 100;
    dbg!(num_total);

    for i in 0..num_total {
        let img: Tensor<(Const<3>, Const<32>, Const<32>), f32, Cpu> =
            test_imgs.clone().select(dev.tensor(i));
        let lbl = test_lbls.clone().select(dev.tensor(i));

        let p: [Dtype; 10] = model.forward(img).softmax().array();
        let truth = lbl.array();
        println!("{:.2?} vs. {:.2?}", &p, &truth);

        let p_index = get_max_index(p);
        dbg!(p_index);
        let t_index = get_max_index(truth);
        dbg!(t_index);
        if p_index == t_index {
            num_correct += 1.0;
        }
    }
    println!("{}", num_correct / num_total as f64);
}

#[inline]
fn get_max_index(array: [f32; 10]) -> usize {
    let (mut max, mut index) = (array[0], 0);
    for i in 0..10 {
        if array[i] > max {
            max = array[i];
            index = i;
        }
    }
    index
}
