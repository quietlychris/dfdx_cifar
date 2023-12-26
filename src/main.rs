#![feature(generic_const_exprs)]

/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::prelude::*;
use dfdx::tensor_ops::softmax;

mod resnet;
use crate::resnet::*;
mod helper;
use crate::helper::*;
//mod simple_conv;
//use crate::simple_conv::*;

#[derive(Default, Clone, Sequential)]
#[built(FcNet)]
struct FcNetConfig<const NUM_CLASSES: usize> {
    flatten: Flatten2D,
    fc1: LinearConstConfig<3072, 784>,
    fc2: LinearConstConfig<784, 256>,
    fc3: LinearConstConfig<256, NUM_CLASSES>,
    sigmoid: Sigmoid,
}

use cifar_ten::*;
use ndarray::{s, Array1, Array3, Array4};

fn main() {
    //---- Resnet---------
    let mut dev = AutoDevice::default();
    // let arch = Resnet18Config::<10>::default();
    // type Model = Resnet18Config<10>;
    type Model = FcNetConfig<10>;
    let mut model = dev.build_module::<f32>(Model::default());

    // Set up the optimizer using either Sgd or Adam

    let mut opt = dfdx::nn::optim::Sgd::new(
        &model,
        SgdConfig {
            lr: 1e-3,
            momentum: Some(dfdx::nn::Momentum::Classic(0.9)),
            weight_decay: None,
        },
    );

    /*     let mut opt = dfdx::nn::optim::Adam::new(
        &model,
        AdamConfig {
            lr: 1e-7,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
        },
    ); */

    // Use cifar_ten library to download/parse data set
    let (train_data, train_labels, test_data, test_labels) = Cifar10::default()
        .download_and_extract(true)
        .base_path("data")
        .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
        // .download_url("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
        .encode_one_hot(true)
        .build()
        .unwrap()
        .to_ndarray::<f32>()
        .unwrap();

    // Create a training data set using ndarray for convenience
    let mut data = Vec::new();
    for num in 0..1000 {
        let img: Array3<f32> = train_data
            .slice(s![num, .., .., ..])
            .to_owned()
            .into_shape((3, 32, 32))
            .unwrap();
        let inp: Tensor<Rank3<3, 32, 32>, f32, _> = dev.tensor(img.into_raw_vec());

        let label: Array1<f32> = train_labels
            .slice(s![num, ..])
            .to_owned()
            .into_shape(10)
            .unwrap();
        let lbl: Tensor<Rank1<10>, f32, _> = dev.tensor(label.into_raw_vec());
        data.push((inp, lbl));
    }

    // Classification train loop taken from dfdx example     let epochs = 2;
    classification_train(
        &mut model,
        &mut opt,
        // binary_cross_entropy_with_logits_loss,
        cross_entropy_with_logits_loss,
        //mse_loss,
        data.clone().into_iter(),
        5,
        &mut dev,
    )
    .unwrap();

    // Create an eval data set of just a few images for comparison

    let mut total_true = 0;
    let num_eval = 1000;
    for num in 0..num_eval {
        let img: Array3<f32> = test_data
            .slice(s![num, .., .., ..])
            .to_owned()
            .into_shape((3, 32, 32))
            .unwrap();
        let inp: Tensor<Rank3<3, 32, 32>, f32, _> = dev.tensor(img.into_raw_vec());

        let label: Array1<f32> = test_labels
            .slice(s![num, ..])
            .to_owned()
            .into_shape(10)
            .unwrap();
        let lbl: Tensor<Rank1<10>, f32, _> = dev.tensor(label.into_raw_vec());
        let lbl = lbl.as_vec();

        let output = model.forward(inp).softmax().as_vec();

        // Use this to check the actual numerical output of each vector
        println!("Actual: {:.3?}\nLabel: {:.3?}", output, lbl);
        let max_index_output = max_index(&output);
        let max_index_label = max_index(&lbl);
        if max_index_output == max_index_label {
            // if output[max_index_output] > 0.2 {
            total_true += 1;
            // }
        }
    }
    println!("total_true: {}/{}", total_true, num_eval);
    println!("% true: {}", total_true as f32 / num_eval as f32);

    // Compare eval set ouputs from output of just a zeroed input
    println!("Do it with all zeros");
    let inp: Tensor<Rank3<3, 32, 32>, f32, _> = dev.zeros();
    let lbl: Tensor<Rank1<10>, f32, _> = dev.zeros();

    let output: Tensor<Rank1<10>, f32, _> = model.forward(inp);
    // dbg!(output.as_vec());
    println!(
        "Actual: {:.3?}\nLabel: {:.3?}",
        output.softmax().as_vec(),
        lbl.as_vec()
    );
}

/// Our generic training function. Works with any model/optimizer/loss function!
fn classification_train<
    // The input to our network, since we are training, we need it to implement Trace
    // so we can put gradients into it.
    Inp: Trace<E, D>,
    // The type of our label, we specify it here so we guaruntee that the dataset
    // and loss function both work on this type
    Lbl,
    // Our model just needs to implement these two things! ModuleMut for forward
    // and TensorCollection for optimizer/alloc_grads/zero_grads
    Model: Module<Inp::Traced> + ZeroGrads<E, D> + UpdateParams<E, D>,
    // optimizer, pretty straight forward
    Opt: Optimizer<Model, E, D>,
    // our data will just be any iterator over these items. easy!
    Data: Iterator<Item = (Inp, Lbl)>,
    // Our loss function that takes the model's output & label and returns
    // the loss. again we can use a rust builtin
    Criterion: FnMut(Model::Output, Lbl) -> Loss,
    // the Loss needs to be able to call backward, and we also use
    // this generic as an output
    Loss: Backward<E, D> + AsArray<Array = E>,
    // Dtype & Device to tie everything together
    E: Dtype,
    D: Device<E>,
>(
    model: &mut Model,
    opt: &mut Opt,
    mut criterion: Criterion,
    data: Data,
    batch_accum: usize,
    device: &mut AutoDevice,
) -> Result<(), Error> {
    // Should this be inside or outside the enumeration loop?
    let mut grads = model.try_alloc_grads()?;
    for (i, (inp, lbl)) in data.enumerate() {
        let y = model.try_forward_mut(inp.traced(grads))?;

        let loss = criterion(y, lbl);
        let loss_value = loss.array();
        // println!("Loss value for {}: {:?}", i, &loss_value);
        grads = loss.try_backward().unwrap();
        device.synchronize();
        if i % batch_accum == 0 {
            println!("Updating!");
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
            device.synchronize();
            println!("batch {i} | loss = {loss_value:?}");
        }
    }
    Ok(())
}
