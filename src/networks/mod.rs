//pub mod resnet;
// pub mod simple_conv;
//pub mod fast_resnet;
//pub mod alexnet;

mod smallconv;
pub use crate::networks::smallconv::*;

mod simple_conv;
pub use crate::networks::simple_conv::*;

mod resnet18;
pub use crate::networks::resnet18::*;
