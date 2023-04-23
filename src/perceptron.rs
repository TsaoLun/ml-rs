//! # 感知机
use ndarray::prelude::*;

/// 与门
pub fn and(x1: f64, x2: f64) -> usize {
    let x = array![x1, x2];
    let w = array![0.5, 0.5];
    let b = -0.7; // 偏执值，神经元被激活的容易程度
    let _tmp = (&w * &x).sum();
    println!("{}", _tmp);
    if _tmp + b <= 0. {
        0
    } else {
        1
    }
}

#[test]
fn _and() {
    let _t = and(0.5, 0.5);
    assert!(_t == 0);
    assert!(and(0., 1.) == 0);
    assert!(and(1., 0.) == 0);
    assert!(and(0., 1.) == 0);
    assert!(and(1., 1.) == 1);
}

/// 与非门
pub fn nand(x1: f64, x2: f64) -> usize {
    let x = array![x1, x2];
    let w = array![-0.5, -0.5];
    let b = 0.7; // 偏执值，神经元被激活的容易程度
    let _tmp = (&w * &x).sum();
    if _tmp + b <= 0. {
        0
    } else {
        1
    }
}

#[test]
fn _nand() {
    assert!(nand(0., 1.) == 1);
    assert!(nand(0., 0.) == 1);
    assert!(nand(0., 0.) == 1);
    assert!(nand(1., 1.) == 0);
}

/// 或门
pub fn or(x1: f64, x2: f64) -> usize {
    let x = array![x1, x2];
    let w = array![0.5, 0.5];
    let b = -0.2; // 偏执值，神经元被激活的容易程度
    let _tmp = (&w * &x).sum();
    if _tmp + b <= 0. {
        0
    } else {
        1
    }
}

#[test]
fn _or() {
    assert!(or(0., 0.) == 0);
    assert!(or(1., 0.) == 1);
    assert!(or(0., 1.) == 1);
    assert!(or(1., 1.) == 1);
    let x = array![0.8, 0.8];
    let w = array![0.5, 0.5];
    println!("{:?}", (&x)*(&w));
}

/// 异或门无法通过一条直线分隔，单层感知器无法表示异或门
///
/// 可通过多层感知器实现 XOR: NAND + OR => AND
fn xor(x1: f64, x2: f64) -> usize {
    let s1 = nand(x1, x2);
    let s2 = or(x1, x2);
    and(s1 as f64, s2 as f64)
}

#[test]
fn _xor() {
    assert!(xor(0., 0.) == 0);
    assert!(xor(1., 0.) == 1);
    assert!(xor(0., 1.) == 1);
    assert!(xor(1., 1.) == 0);
}
