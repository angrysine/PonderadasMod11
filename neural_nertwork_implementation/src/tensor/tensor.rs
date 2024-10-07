use std::f32::INFINITY;
use std::ops::Mul;
use std::ops::Add;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<Vec<f32>>,
    pub cols: usize,
    pub rows: usize,
}

impl Tensor 
{
    pub fn new(rows: usize, cols: usize,data :Vec<Vec<f32>>) -> Tensor {
        if !data.is_empty() {
            if (data.len() != rows ) || (data[0].len() != cols) {
                panic!("invalid data shape");
            }
        }
        else {
            if rows !=0 || cols !=0 {
                panic!("invalid data shape")
            }
        }

        Tensor {
            data,
            cols,
            rows,
        }
        //vec![rows, cols ],
        // kernel[rows,cols]
    }

    pub fn convolution(& self,kernel :&Vec<Vec<f32>>) -> Tensor
    {
        let mut new_data = vec![vec![0.0; self.cols-kernel[0].len()+1]; self.rows-kernel.len()+1];
        if  ( self.rows <kernel.len() ) || (self.cols <kernel[0].len()) {
            panic!("invalid kernel size");
        }
     
        for i in 0.. (self.rows - kernel.len()+1) {
            for j in 0.. (self.cols - kernel[0].len()+1) {
                let mut sum =0.0;
                for k in 0.. kernel.len() {
                    for l in 0..kernel[0].len(){
                        sum += self.data[i+k][j+l] * kernel[k][l];
                    }
                }
                new_data[i][j] = sum;
            }
        }
        return Tensor{
            rows:self.rows - kernel.len()+1
            ,cols:self.cols-kernel[0].len()+1
            ,data:new_data};
    }

    pub fn apply_function(& self, f: fn(f32) -> f32) -> Tensor {
        let mut new_data = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i][j] = f(self.data[i][j]);
            }
        }
        return Tensor{
            cols:self.cols,
            rows:self.rows,
            data:new_data};
    }
    pub fn average_pooling(& self, rows: usize, cols: usize) -> Tensor 
    {
        if (self.rows % rows != 0) || (self.cols % cols != 0) {
            panic!("invalid pooling size");
        }
        let mut new_data = vec![vec![0.0;self.cols/cols]; self.rows/rows];
        for i in 0..(self.rows/rows){
            for j in 0..(self.cols/cols) {
                let mut sum =0.0;
                for k in 0..rows {
                    for l in 0..cols {
                        sum += self.data[i*rows+k][j*cols+l];
                    }
                }
                new_data[i][j] = sum/(rows*cols) as f32;    
            }
        }
        return Tensor{rows:rows,cols:cols,data:new_data};
    }

    pub fn max_pooling(& self, rows: usize, cols: usize) -> Tensor 
    {
        if (self.rows % rows != 0) || (self.cols % cols != 0) {
            panic!("invalid pooling size");
        }
        let mut new_data = vec![vec![0.0;self.cols/cols]; self.rows/rows];
        for i in 0..(self.rows/rows){
            for j in 0..(self.cols/cols) {
                let mut max =-INFINITY;
                for k in 0..rows {
                    for l in 0..cols {
                        max = if self.data[i*rows+k][j*cols+l]> max {self.data[i*rows+k][j*cols+l]} else {max};
                    }
                }
                new_data[i][j] = max;    
            }
        }
        return Tensor{rows:rows,cols:cols,data:new_data};
    }
    pub fn cross_product(&self, other: &Tensor) -> Tensor {
        if self.cols != other.rows {
            panic!("invalid matrix size");
        }
        let mut new_data = vec![vec![0.0;other.cols];self.rows];
        for i in 0..self.rows 
        {
            for j in 0..other.cols 
            {
                for k in 0..self.cols 
                {
                    new_data[i][j]+= self.data[i][k]*other.data[k][j];
                }
            }
        }
        return Tensor {data:new_data,cols:other.cols,rows:self.rows}
    }

    pub fn flatten(&self) -> Vec<f32>
    {
        let mut new_data = vec![0.0;self.rows*self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i*self.rows +j] = self.data[i][j]; 
            }
        }
        return new_data;
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    
    fn mul(self, other: &Tensor) -> Tensor {
        let mut new_data : Vec<Vec<f32>> = vec![vec![0.0;self.cols];self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i][j] = self.data[i][j]*other.data[i][j];
            }
        }    
        Tensor::new(self.rows,self.cols,new_data)
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        let mut new_data : Vec<Vec<f32>> = vec![vec![0.0;self.cols];self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }    
        Tensor::new(self.rows,self.cols,new_data)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] == other.data[i][j] {return false;}
            }
        }    
        return true
    }
}
