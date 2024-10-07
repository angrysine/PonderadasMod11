use crate::Tensor;

#[test]
fn test_tensor_convolutional() {
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

    let kernel = vec![
        vec![1.0, 0.0, -1.0],
        vec![2.0, 0.0, -2.0],
        vec![1.0, 0.0, -1.0],
    ];

    let convoluted_value = tensor.convolution(&kernel);
    println!();
    for i in 0..(convoluted_value.data.len()) {
        
        println!("{:?}", convoluted_value.data[i]);
    }
}

#[test]
fn test_tensor_apply_function() {
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

    let new_tensor = tensor.apply_function(|val| -> f32 { val * 2.0 });
    println!();
    for i in 0..(new_tensor.data.len()) {
        
        println!("{:?}", new_tensor.data[i]);
    }
}

#[test]
fn test_tensor_avarege_pooling() {
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

    let new_tensor = tensor.average_pooling(3,3);
    println!();
    for i in 0..(new_tensor.data.len()) {
        
        println!("{:?}", new_tensor.data[i]);
    }
}

#[test]
fn test_tensor_max_pooling() {
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

    let result = vec![vec![40.0, 23.0], vec![31.0, 30.0]];

    let new_tensor = tensor.max_pooling(3, 3);

    println!("Results for test_tensor_max_pooling:");
    for i in 0..(new_tensor.data.len()) {
        println!("{:?}", new_tensor.data[i]);
    }
    println!("\n");

    assert_eq!(new_tensor.data, result);
}

#[test]
fn test_tensor_cross_product() {
        let matrix_a = vec![
        vec![5.0, 8.0, -4.0],
        vec![6.0, 9.0, -5.0],
        vec![4.0, 7.0, -2.0],
    ];
    
    let matrix_b = vec![
        vec![2.0],
        vec![-3.0],
        vec![1.0],
    ];
    let tensor = Tensor::new(
        3,
        3,
        matrix_a,
    );

    let tensor2 = Tensor::new(
        3,
        1,
        matrix_b
    );

    // let result = vec![vec![40.0, 23.0], vec![31.0, 30.0]];

    let new_tensor = tensor.cross_product(&tensor2);

    println!("Results for test_tensor_cross_product:");
    for i in 0..(new_tensor.data.len()) {
        println!("{:?}", new_tensor.data[i]);
    }
    println!("\n");

    // assert_eq!(new_tensor.data, result);
}

#[test]
fn test_tensor_cross_product2() {
    let tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let tensor_b = Tensor::new(
        3,
        2,
        vec![vec![20.0, 24.0], vec![19.0, 17.0], vec![21.0, 40.0]],
    );

    let result = vec![vec![1087., 1328.], vec![1123., 1545.], vec![1705., 2184.]];

    let new_tensor = tensor.cross_product(&tensor_b);

    println!("Results for test_tensor_cross_product2:");
    for i in 0..(new_tensor.data.len()) {
        println!("{:?}", new_tensor.data[i]);
    }
    println!("\n");

    assert_eq!(new_tensor.data, result);
}

#[test]
fn test_tensor_flattern() {
    let tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let result = vec![20.0, 24.0, 11.0, 19.0, 17.0, 20.0, 21.0, 40.0, 25.0];

    let new_tensor = tensor.flatten();

    println!("Results for test_tensor_flattern:");
    for i in 0..(new_tensor.len()) {
        println!("{:?}", new_tensor[i]);
    }
    println!("\n");

    assert_eq!(new_tensor, result);
}

#[test]
fn test_tensor_add() {
    let tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let tensor2 = Tensor::new(
        3,
        3,
        vec![
            vec![21.0, 40.0, 25.0],
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
        ],
    );

    let result = Tensor::new(
        3,
        3,
        vec![
            vec![41.0, 64.0, 36.0],
            vec![39.0, 41.0, 31.0],
            vec![40.0, 57.0, 45.0],
        ],
    );

    let new_tensor = &tensor + &tensor2;

    println!("Results for test_tensor_add:");
    for i in 0..(new_tensor.data.len()) {
        println!("{:?}", new_tensor.data[i]);
    }
    println!("\n");

    
    for i in 0..(new_tensor.data.len()) {
        assert_eq!(new_tensor.data[i], result.data[i]);
    }
}

#[test]
fn test_tensor_multiply() {
    let tensor = Tensor::new(
        3,
        3,
        vec![
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
            vec![21.0, 40.0, 25.0],
        ],
    );

    let tensor2 = Tensor::new(
        3,
        3,
        vec![
            vec![21.0, 40.0, 25.0],
            vec![20.0, 24.0, 11.0],
            vec![19.0, 17.0, 20.0],
        ],
    );

    let result = Tensor::new(
        3,
        3,
        vec![
            vec![420.0, 960.0, 275.0],
            vec![380.0, 408.0, 220.0],
            vec![399.0,680.0,500.0],
        ],
    );

    let new_tensor = &tensor*&tensor2;

    println!("Results for test_tensor_multiply:");
    for i in 0..(new_tensor.data.len()) {
        println!("{:?}", new_tensor.data[i]);
    }
    println!("\n");

    
    for i in 0..(new_tensor.data.len()) {
        assert_eq!(new_tensor.data[i], result.data[i]);
    }
}

// #[test]
// fn 