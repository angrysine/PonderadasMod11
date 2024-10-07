mod tensor;

mod layer;

mod network;

pub use tensor::Tensor;

pub use layer::layer::LayerTrait;

pub use layer::poolingLayers::{MaxPoolingLayer,AvgPoolingLayer};

// pub use network::Network;

#[cfg(test)]
mod test;