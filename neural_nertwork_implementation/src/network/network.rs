use crate::layer::layer::LayerTrait;
use crate::layer::poolingLayers::{MaxPoolingLayer,PoolingLayers};

struct Network {
    layers: Vec<Box<dyn LayerTrait>>,
    first_input_row_size: usize,
    first_input_col_size: usize,
}

impl Network {
    pub fn new(first_input_row_size: usize, first_input_col_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            first_input_row_size,
            first_input_col_size,
        }
    }

    pub fn add_pooling_layer(
        &mut self,
        pooling_type: PoolingLayers,
        kernel_rows: usize,
        kernel_cols: usize,
    ) {
        match pooling_type {
            PoolingLayers::MAX => {
                if self.layers.len() > 0 {
                    let last_layer = self.layers.last().unwrap();

                    let (input_rows, input_cols) = last_layer.get_output_shape();

                    self.layers.push(Box::new(MaxPoolingLayer::new(
                        kernel_rows,
                        kernel_cols,
                        input_rows,
                        input_cols,
                    )));
                }
            }
            PoolingLayers::AVG => (),
        }
    }
}