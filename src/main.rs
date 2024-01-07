#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::prelude::*;
// use dfdx::tensor_ops::softmax;

use std::error::Error;
mod networks;
use crate::networks::*;
mod helper;
use crate::helper::*;

use cifar_ten::*;
use ndarray::{s, Array1, Array2, Array3, Array4};
use pbr::ProgressBar;
use rand::seq::SliceRandom;
use rand::thread_rng;

const BS: usize = 1; // BATCH_SIZE

fn main() -> Result<(), Box<dyn Error>> {
    dfdx::flush_denormals_to_zero();
    //---- Resnet---------
    let mut dev = AutoDevice::default();
    type Model = SimpleConvConfig<10>;
    //type Model = Resnet18Config<10>;
    let mut model = dev.build_module::<f32>(Model::default());

    // Set up the optimizer using either Sgd or Adam

    let mut opt = dfdx::nn::optim::Sgd::new(
        &model,
        SgdConfig {
            lr: 1e-4,
            momentum: Some(dfdx::nn::Momentum::Classic(0.9)),
            weight_decay: None,
        },
    );

    /*     let mut opt = dfdx::nn::optim::Adam::new(
        &model,
        AdamConfig {
            lr: 1e-8,
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
    // Normalize the imagery
    let train_data = train_data.mapv(|x| x / 256.0);
    let test_data = test_data.mapv(|x| x / 256.0);

    // Create a training data set using ndarray for convenience
    let mut data = Vec::new();
    for num in 0..(50_000 / BS) {
        let base = BS * num;
        let img: Array4<f32> = train_data
            .slice(s![base..base+BS, .., .., ..])
            .to_owned()
            .into_shape((BS, 3, 32, 32))
            .unwrap();
        // println!("{:.2?}", &img);
        let inp: Tensor<Rank4<BS, 3, 32, 32>, f32, _> = dev.tensor(img.into_raw_vec());

        let label: Array2<f32> = train_labels
            .slice(s![base..base+BS, ..])
            .to_owned()
            .into_shape((BS, 10))
            .unwrap();
        let lbl: Tensor<Rank2<BS, 10>, f32, _> = dev.tensor(label.into_raw_vec());
        data.push((inp, lbl));
    }

    for epoch in 0..7 {
        let mut data = data.clone();
        if epoch > 2 {
            opt.cfg.lr = 1e-5;
        }
        if epoch > 5 {
            opt.cfg.lr = 1e-6;
        }
        data.shuffle(&mut thread_rng());

        // Classification train loop taken from dfdx example
        println!("Epoch #{}", epoch);
        classification_train(
            &mut model,
            &mut opt,
            // binary_cross_entropy_with_logits_loss,
            cross_entropy_with_logits_loss,
            // mse_loss,
            data.into_iter(),
            1,
            &mut dev,
        )
        .unwrap();

        // Create an eval data set of just a few images for comparison
        let mut total_true = 0;
        let num_eval = 1000;
        for num in 0..num_eval {
            let img: Array4<f32> = test_data
                .slice(s![num, .., .., ..])
                .to_owned()
                .into_shape((1, 3, 32, 32))
                .unwrap();
            let inp: Tensor<Rank4<1, 3, 32, 32>, f32, _> = dev.tensor(img.into_raw_vec());

            let label: Array2<f32> = test_labels
                .slice(s![num, ..])
                .to_owned()
                .into_shape((1, 10))
                .unwrap();
            let lbl: Tensor<Rank2<1, 10>, f32, _> = dev.tensor(label.into_raw_vec());
            let lbl = lbl.as_vec();

            let output = model.try_forward(inp)?.softmax::<Axis<1>>().as_vec();

            // Use this to check the actual numerical output of each vector
            // println!("Actual: {:.3?}\nLabel: {:.3?}", output, lbl);
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
    }

    // Compare eval set ouputs from output of just a zeroed input
    /*
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
    */

    Ok(())
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
    Data: ExactSizeIterator<Item = (Inp, Lbl)>,
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
) -> Result<(), Box<dyn Error>> {
    // Should this be inside or outside the enumeration loop?
    let mut grads = model.try_alloc_grads()?;

    let full_size = data.len();
    let mut pb = ProgressBar::new(full_size.try_into().unwrap());
    pb.format("╢=> ╟");

    for (i, (inp, lbl)) in data.into_iter().enumerate() {
        let y = model.try_forward_mut(inp.traced(grads))?;

        let loss = criterion(y, lbl);
        let loss_value = loss.array();
        // println!("Loss value for {}: {:?}", i, &loss_value);
        grads = loss.try_backward().unwrap();
        device.synchronize();
        pb.inc();
        if i % batch_accum == 0 {
            // println!("Updating!");
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
            device.synchronize();
            pb.message(format!("loss = {loss_value:.3?} |").as_str());
        }
    }
    Ok(())
}
