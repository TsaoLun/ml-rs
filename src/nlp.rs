//! # 神经网络
use ndarray::prelude::*;

/// 神经网络的重要性质是可以自动地从数据中学到合适的权重参数
/// 激活函数 h(x) 即 activation function 将输入信号的总和转换为输出信号

/// sigmoid 函数是神经网络和感知器（阶跃函数）之间的主要差别
pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

#[test]
fn _sigmod() {
    println!("{}", array![-1., 1., 2.].map(|e| sigmoid(*e)))
}
