//! # 神经网络
#![allow(non_snake_case)]
#![allow(dead_code)]
use std::collections::HashMap;

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
fn _test_sigmod() {
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

/// 输出层的激活函数 sigma()
/// 对于回归问题可以使用恒等函数，二元分类问题可以使用 sigmoid 函数，多元分类问题可以使用 softmax 函数
fn identity_function<T>(x: T) -> T {
    x
}

fn init_network() -> HashMap<&'static str, Array<f64, IxDyn>> {
    let mut network = HashMap::new() as HashMap<&str, Array<f64, IxDyn>>;
    network.insert("W1", array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]].into_dyn());
    network.insert("b1", array![0.1, 0.2, 0.3].into_dyn());
    network.insert("W2", array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]].into_dyn());
    network.insert("b2", array![0.1, 0.2].into_dyn());
    network.insert("W3", array![[0.1, 0.3], [0.2, 0.4]].into_dyn());
    network.insert("b3", array![0.1, 0.2].into_dyn());
    return network;
}

fn forward(network: HashMap<&str, Array<f64, IxDyn>>, x: Array1<f64>) -> Array<f64, IxDyn> {
    let (W1, W2, W3) = (&network["W1"], &network["W2"], &network["W3"]);
    let (b1, b2, b3) = (&network["b1"], &network["b2"], &network["b3"]);
    let a1 = x.dot(&W1.clone().into_dimensionality::<Ix2>().unwrap()) + b1;
    let z1 = sigmod(a1).into_dimensionality::<Ix1>().unwrap();
    let a2 = z1.dot(&W2.clone().into_dimensionality::<Ix2>().unwrap()) + b2;
    let z2 = sigmod(a2).into_dimensionality::<Ix1>().unwrap();
    let a3 = z2.dot(&W3.clone().into_dimensionality::<Ix2>().unwrap()) + b3;
    a3
}

#[test]
fn test_forward() {
    let network = init_network();
    let x = array![1.0, 0.5];
    let y = forward(network, x);
    println!("{:?}", y);
}
