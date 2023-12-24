#![feature(generic_const_exprs)]

/// This advanced example shows how to work with dfdx in a generic
/// training setting.
use dfdx::prelude::*;
mod resnet;

use crate::resnet::*;

use cifar_ten::*;
use ndarray::{s, Array1, Array3};

fn main() {
    //---- Resnet---------
    let dev = AutoDevice::default();
    // let arch = Resnet18Config::<10>::default();
    type Model = Resnet18Config<10>;
    let mut model = dev.build_module::<f32>(Model::default());

    let mut opt = dfdx::nn::optim::Adam::new(&model, Default::default());

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

    let mut data = Vec::new();
    for num in 0..100 {
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

    classification_train(
        &mut model,
        &mut opt,
        //binary_cross_entropy_with_logits_loss,
        cross_entropy_with_logits_loss,
        data.into_iter(),
        10,
    )
    .unwrap();
    //---------------

    for num in 0..3 {
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

        let output: Tensor<Rank1<10>, f32, _> = model.forward(inp);
        // dbg!(output.as_vec());
        println!(
            "Input: {:.3?}\nOutput: {:.3?}",
            output.as_vec(),
            lbl.as_vec()
        );
    }

    println!("Do it with all zeros");
    let inp: Tensor<Rank3<3, 32, 32>, f32, _> = dev.zeros();
    let lbl: Tensor<Rank1<10>, f32, _> = dev.zeros();

    let output: Tensor<Rank1<10>, f32, _> = model.forward(inp);
        // dbg!(output.as_vec());
        println!(
            "Input: {:.3?}\nOutput: {:.3?}",
            output.as_vec(),
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
) -> Result<(), Error> {
    let mut grads = model.try_alloc_grads()?;
    for (i, (inp, lbl)) in data.enumerate() {
        let y = model.try_forward_mut(inp.traced(grads))?;
        let loss = criterion(y, lbl);
        let loss_value = loss.array();
        grads = loss.try_backward().unwrap();
        if i % batch_accum == 0 {
            println!("Updating!");
            opt.update(model, &grads).unwrap();
            model.try_zero_grads(&mut grads)?;
            println!("batch {i} | loss = {loss_value:?}");
        }
    }
    Ok(())
}
