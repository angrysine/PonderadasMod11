use crate::Tensor;


pub trait LayerDefaultTrait {
    fn get_output_shape(&self) -> (usize,usize);
    fn get_input_shape(&self) -> (usize,usize);
}
pub trait LayerTrait :LayerDefaultTrait
{
    fn feedfoward(& mut self,tensor: &Tensor) -> &Tensor;
    fn backpropagate(&self) -> Tensor;
}



