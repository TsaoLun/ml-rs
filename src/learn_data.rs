#![allow(non_snake_case)]
#![allow(dead_code)]
use mnist::MnistBuilder;
use ndarray::{Array2, Array1};

pub static STANDARD_DISTRIBUTION: f32 = 0.02;
pub static OUTPUT_SIZE: usize = 10;
pub static IMAGE_DIMENSION: usize = 2;
pub static IMAGE_SIZE: usize = IMAGE_DIMENSION * IMAGE_DIMENSION;
pub static BATCH_SIZE: usize = 20;
pub static LEARN_FACTOR: f64 = 0.005;
pub static UPDATE_FACTOR: f64 = LEARN_FACTOR / BATCH_SIZE as f64;

pub struct LearnData {
    pub image_parts: Array2<u8>,
    pub expected_class: Array1<u8>,
}

impl LearnData {
    pub fn load_mnist() -> (Vec<Self>, Vec<Self>) {
        let mnist = MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(60_000)
            .test_set_length(10_000)
            .finalize();

        let train_data = mnist
            .trn_img
            .chunks(IMAGE_SIZE)
            .map(|chunk| Array2::from_shape_vec((IMAGE_DIMENSION, IMAGE_DIMENSION), chunk.into()))
            .zip(mnist.trn_lbl.chunks(OUTPUT_SIZE))
            .map(|(image_parts, expected_class)| Self {
                image_parts: image_parts.unwrap(),
                expected_class: Array1::from_shape_vec(OUTPUT_SIZE, expected_class.into()).unwrap(),
            })
            .collect();
        let test_data = mnist
            .tst_img
            .chunks(IMAGE_SIZE)
            .map(|chunk| Array2::from_shape_vec((IMAGE_DIMENSION, IMAGE_DIMENSION), chunk.into()))
            .zip(mnist.tst_lbl.chunks(OUTPUT_SIZE))
            .map(|(image_parts, expected_class)| Self {
                image_parts: image_parts.unwrap(),
                expected_class: Array1::from_shape_vec(OUTPUT_SIZE, expected_class.into()).unwrap(),
            })
            .collect();
        (train_data, test_data)
    }
}