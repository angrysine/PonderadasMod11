use std::f32::NEG_INFINITY;
use layer_macros::layer;
use crate::Tensor;
use super::layer::{LayerTrait, LayerDefaultTrait};
#[layer]
pub struct  MaxPoolingLayer {

}

impl MaxPoolingLayer {
    pub fn new(kernel_rows: usize,kernel_cols:usize,input_rows: usize,input_cols:usize) -> Self
    {
        return MaxPoolingLayer {
            input_col_size:input_cols,
            input_row_size:input_rows,
            output_col_size:input_cols/kernel_rows,
            output_row_size:input_rows/kernel_cols,
            last_output: Tensor::new(0, 0, vec![vec![0.0;0];0]),
            last_input: Tensor::new(0, 0, vec![vec![0.0;0];0]),
        };
    }
}

impl LayerTrait for MaxPoolingLayer 
{
    fn feedfoward(&mut self, tensor: &Tensor) -> &Tensor {
        self.last_input = tensor.clone();
        self.last_output = tensor.max_pooling(
            self.input_row_size / self.output_row_size,
            self.input_col_size / self.output_col_size,
        );

        return &self.last_output;
        
    }

    fn backpropagate(&self) -> Tensor 
    {
        let last_input_rows = &self.input_row_size;
        let last_input_cols = &self.input_col_size;

        let (kernel_row, kernel_col) = (
            self.input_row_size / self.output_row_size,
            self.input_col_size / self.output_col_size,
        );

        let mut new_data: Vec<Vec<f32>> = vec![vec![0.0; *last_input_cols]; *last_input_rows];

        for i in 0..*last_input_rows {
            for j in 0..*last_input_cols {
                let mut local_max: f32 = NEG_INFINITY;
                let mut local_row =0;
                let mut local_col =0;
                for k in 0..kernel_row {
                    for l in 0..kernel_col {
                        if self.last_input.data[i+k][j+l] > local_max {
                            local_row = i+k;
                            local_col= j+l;
                            local_max = self.last_input.data[i+k][j+l];
                        }
                    }
                }
                new_data[local_row][local_col] = self.last_output.data[local_row][local_col];
            }
        }
        return &Tensor::new(*last_input_rows, *last_input_cols, new_data)*&self.last_input;
    }
}

#[layer]
pub struct AvgPoolingLayer {}

impl AvgPoolingLayer {
    pub fn new(
        kernel_rows: usize,
        kernel_cols: usize,
        input_rows: usize,
        input_cols: usize,
    ) -> Self {
        Self {
            input_col_size: input_cols,
            input_row_size: input_rows,
            output_row_size: input_cols/ kernel_rows,
            output_col_size: input_rows/ kernel_cols,
            last_input: Tensor::new(0, 0, vec![vec![0.0; 0]; 0]),
            last_output: Tensor::new(0, 0, vec![vec![0.0; 0]; 0]),
        }
    }
}

impl LayerTrait for AvgPoolingLayer {
    fn feedfoward(&mut self, tensor: &Tensor) -> &Tensor {
        self.last_input = tensor.clone();
        self.last_output = tensor.average_pooling(
            self.input_row_size / self.output_row_size,
            self.input_col_size / self.output_col_size,
        );

        &self.last_output
    }

    fn backpropagate(&self) -> Tensor {
        let kernel_cols = self.input_col_size / self.output_col_size;
        let kernel_rows = self.input_row_size / self.output_row_size;

        let mut new_data = vec![vec![0.0; self.input_row_size]; self.input_col_size];

        for i in 0..self.output_col_size {
            for j in 0..self.output_row_size {
                for cols in 0..kernel_cols {
                    for rows in 0..kernel_rows {
                        new_data[(i * kernel_cols) + cols][(i * kernel_rows) + rows] =
                            self.last_output.data[i][j]
                    }
                }
            }
        }

        Tensor::new(self.input_row_size, self.input_col_size, new_data)
    }
}

pub enum PoolingLayers {
    MAX,
    AVG
}