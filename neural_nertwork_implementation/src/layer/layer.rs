use crate::Tensor;

pub struct Layer 
{
    pub last_input : Tensor,
    pub feedfoward : fn(Tensor) -> Tensor,
    pub backpropagate : fn(),
    col_input_size: usize,
    row_input_size: usize,
    col_output_size: usize,
    row_output_size: usize,
}


pub struct  LayerFactory {

}


impl LayerFactory {

    // fn max_pooling(row_size: usize,col_size:usize) -> Layer{

    // }
}

pub trait LayerTrait {
    fn feedfoward(tensor: Tensor) -> Tensor;
    fn backpropagate(&self,tensor: &Tensor) -> Tensor;
}


// struct LinearLayer {
//     weight: f32,
// }

// impl LayerTrait for LinearLayer {
//     fn feedfoward(tensor: Tensor) -> Tensor {
//         // Multiply the input tensor by a constant weight for feedforward
//         tensor * 2.0 // assuming Tensor implements basic arithmetic ops
//     }

//     fn backpropagate(&self, tensor: &Tensor) -> Tensor {
//         // Multiply by weight during backpropagation as a simple test
//         tensor * self.weight
//     }
// }

// struct ReLULayer;

// impl LayerTrait for ReLULayer {
//     fn feedfoward(tensor: Tensor) -> Tensor {
//         // Apply ReLU activation function: max(0, x)
//         tensor.map(|x| if x > 0.0 { x } else { 0.0 }) // assuming Tensor has a map function
//     }

//     fn backpropagate(&self, tensor: &Tensor) -> Tensor {
//         // Apply gradient of ReLU (1 for x > 0, else 0)
//         tensor.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
//     }
// }