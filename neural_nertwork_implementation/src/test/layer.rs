use crate::layer::poolingLayers::{MaxPoolingLayer,AvgPoolingLayer};
use crate::tensor::Tensor;
use crate::LayerTrait;


#[test]
fn test_max_pooling_feedfoward() {
    let tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );
    let mut layer = MaxPoolingLayer::new(3, 2, tensor.rows, tensor.cols);
    let result = layer.feedfoward(&tensor);
    let expected_result = Tensor::new(
        2,
        3,
        vec![
            vec![40.0,25.0,19.],
            vec![31.0, 11.0,30.]
        ],
        
    );
    println!("Results for test_max_pooling_feedfoward:");
    for i in 0..(result.data.len()) {
        println!("{:?}", result.data[i]);
    }
    println!("\n");

    
    for i in 0..(result.data.len()) {
        assert_eq!(result.data[i], expected_result.data[i]);
    }
    
}
#[test]
fn test_avg_pooling_feedfoward() {
    let tensor = Tensor::new(
        6,
        6,
        vec![
            vec![20.0, 24.0, 11.0, 12.0, 16.0, 19.0],
            vec![19.0, 17.0, 20.0, 23.0, 15.0, 9.0],
            vec![21.0, 40.0, 25.0, 13.0, 14.0, 8.0],
            vec![9.0, 18.0, 8.0, 6.0, 11.0, 22.0],
            vec![31.0, 3.0, 7.0, 9.0, 17.0, 23.0],
            vec![20.0, 12.0, 3.0, 11.0, 19.0, 30.0],
        ],
    );
    let mut layer = AvgPoolingLayer::new(3, 2, tensor.rows, tensor.cols);
    let result = layer.feedfoward(&tensor);
    let expected_result = Tensor::new(
        2,
        3,
        vec![
            vec![23.5,17.3333333,13.5],
            vec![15.5, 7.3333333,20.3333333]
        ],
        
    );
    println!("Results for test_avg_pooling_feedfoward:");
    for i in 0..(result.data.len()) {
        println!("{:?}", result.data[i]);
    }
    println!("\n");

    
    for i in 0..(result.data.len()) {
        assert_eq!(result.data[i], expected_result.data[i]);
    }
    
}