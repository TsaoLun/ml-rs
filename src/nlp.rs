//! # 神经网络
use ndarray::{prelude::*, DataMut};

/// 神经网络的重要性质是可以自动地从数据中学到合适的权重参数
/// 激活函数 h(x) 即 activation function 将输入信号的总和转换为输出信号

/// sigmoid 函数是神经网络和感知器（阶跃函数）之间的主要差别
fn _sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn sigmod<S, D>(list: ArrayBase<S, D>)
where
    S: DataMut,
    D: Dimension,
{
    list.map(|e| _sigmoid(*e));
}

/// ReLU 函数 Rectified Linear Unit，在大于 0 时输出，其他情况输出 0
pub fn relu(x: f64) -> f64 {
    if x > 0. {
        x
    } else {
        0.
    }
}

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
