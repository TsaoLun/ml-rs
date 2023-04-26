//! # 神经网络
#![allow(non_snake_case)]
#![allow(dead_code)]
use ndarray::prelude::*;

/// 矩阵乘积
#[test]
fn _matrix() {
    let A = array![[1, 2, 3], [4, 5, 6]];
    let B = array![[1, 2], [3, 4], [5, 6]];
    println!("{:?} {:?} -> {:?}", A.shape(), B.shape(), A.dot(&B).shape());
}

#[test]
#[should_panic]
fn _matrix_err() {
    let A = array![[1, 2, 3], [4, 5, 6]];
    let C = array![[1, 2], [3, 4]];
    A.dot(&C);
}

/// 神经网络的重要性质是可以自动地从数据中学到合适的权重参数
/// 激活函数 h(x) 即 activation function 将输入信号的总和转换为输出信号

/// sigmoid 函数是神经网络和感知器（阶跃函数）之间的主要差别
fn _sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmod<D>(list: Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    list.map(|e| _sigmoid(*e))
}

#[test]
fn _test_sigmod(){
    let X = array![1.0, 0.5];
    let W1 = array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]];
    let B1 = array![0.1, 0.2, 0.3];
    println!("{:?}", W1.shape());
    let A1 = X.dot(&W1) + B1;
    println!("{:?}", A1);
    let Z1 = sigmod(A1);
    println!("{:?}", Z1);
}

/// ReLU 函数 Rectified Linear Unit，在大于 0 时输出，其他情况输出 0
pub fn relu(x: f64) -> f64 {
    if x > 0. {
        x
    } else {
        0.
    }
}