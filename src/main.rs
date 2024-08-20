mod activation;
mod mnist;

use activation::Activation;
use itertools::zip_eq;
use ndarray::{Array1, Array2, Axis, LinalgScalar, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand_distr::{Distribution, Normal, StandardNormal};

struct Layer<Signal> {
    pub neuron_weights: Array2<Signal>,
    pub bias_weights: Array1<Signal>,
}

struct NeuralNetwork<Signal> {
    pub layers: Vec<Layer<Signal>>,
}

struct LayerCache<Signal> {
    inputs: Array1<Signal>,
    outputs: Array1<Signal>,
    error: Array1<Signal>,
}

struct NeuralNetworkCache<Signal> {
    layers: Vec<LayerCache<Signal>>,
}

pub fn softmax<Signal: Float>(input: &Array1<Signal>) -> Array1<Signal> {
    let sum = input.map(|x| x.exp()).sum();
    input.map(|x| x.exp() / sum)
}

impl<Signal> Layer<Signal> {
    pub fn len(&self) -> usize {
        self.neuron_weights.nrows()
    }

    pub fn forward<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut LayerCache<Signal>,
        inputs: &Array1<Signal>,
    ) -> &'a Array1<Signal>
    where
        Signal: LinalgScalar,
    {
        cache.inputs = self.neuron_weights.dot(inputs) + &self.bias_weights;
        cache.outputs.assign(&cache.inputs);
        cache.outputs.mapv_inplace(A::function);
        &cache.outputs
    }

    pub fn backpropagate<A: Activation<Signal>>(
        &self,
        cache: &mut LayerCache<Signal>,
        next_layer_weighted_error: Array1<Signal>,
    ) where
        Signal: LinalgScalar + ScalarOperand,
    {
        cache.error = cache.inputs.mapv(A::derivative) * next_layer_weighted_error;
    }

    pub fn learn(
        &mut self,
        cache: &LayerCache<Signal>,
        inputs: &Array1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand,
    {
        let weight_gradient = cache
            .error
            .view()
            .insert_axis(Axis(1))
            .dot(&inputs.view().insert_axis(Axis(0)));
        self.neuron_weights = &self.neuron_weights - &weight_gradient * learning_rate;
        self.bias_weights = &self.bias_weights - &cache.error * learning_rate
    }
}

impl<Signal> NeuralNetwork<Signal> {
    pub fn new(layer_sizes: impl Iterator<Item = usize> + Clone) -> Self
    where
        Signal: LinalgScalar + FromPrimitive + Float,
        StandardNormal: Distribution<Signal>,
    {
        let normal = Normal::new(Signal::zero(), Signal::one()).unwrap();
        let two = Signal::one() + Signal::one();
        Self {
            layers: layer_sizes
                .clone()
                .zip(layer_sizes.skip(1))
                .map(|(input_size, output_size)| Layer {
                    neuron_weights: Array2::from_shape_fn((output_size, input_size), |_| {
                        normal.sample(&mut rand::thread_rng())
                            * (two / Signal::from_usize(input_size).unwrap()).sqrt()
                    }),
                    bias_weights: Array1::from_shape_fn(output_size, |_| {
                        normal.sample(&mut rand::thread_rng())
                    }),
                })
                .collect(),
        }
    }

    pub fn forward<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut NeuralNetworkCache<Signal>,
        inputs: &'a Array1<Signal>,
    ) -> &'a Array1<Signal>
    where
        Signal: LinalgScalar,
    {
        zip_eq(self.layers.iter(), cache.layers.iter_mut())
            .fold(inputs, |inputs, (layer, cache)| {
                layer.forward::<A>(cache, inputs)
            })
    }

    pub fn backpropagate<A: Activation<Signal>>(
        &self,
        cache: &mut NeuralNetworkCache<Signal>,
        expected_outputs: &Array1<Signal>,
    ) where
        Signal: LinalgScalar + ScalarOperand + Float,
    {
        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let (_, [layer_cache, post_caches @ ..]) = cache.layers.split_at_mut(i) else {
                panic!("Couldn't split cache layers");
            };
            let next_layer_weighted_error = if i == self.layers.len() - 1 {
                // TODO: Generalize loss function and its derivative?
                &softmax(&layer_cache.outputs) - expected_outputs
            } else {
                self.layers[i + 1]
                    .neuron_weights
                    .view()
                    .t()
                    .dot(&post_caches.first().unwrap().error.view())
            };
            layer.backpropagate::<A>(layer_cache, next_layer_weighted_error);
        }
    }

    pub fn learn(
        &mut self,
        cache: &NeuralNetworkCache<Signal>,
        inputs: &Array1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand,
    {
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];
            let layer_cache = &cache.layers[i];
            let inputs = if i == 0 {
                inputs
            } else {
                &cache.layers[i - 1].outputs
            };
            layer.learn(layer_cache, inputs, learning_rate);
        }
    }

    pub fn train<A: Activation<Signal>>(
        &mut self,
        cache: &mut NeuralNetworkCache<Signal>,
        inputs: &Array1<Signal>,
        expected_outputs: &Array1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand + Float,
    {
        self.forward::<A>(cache, inputs);
        self.backpropagate::<A>(cache, expected_outputs);
        self.learn(cache, inputs, learning_rate);
    }
}

impl<Signal> NeuralNetworkCache<Signal> {
    pub fn new(network: &NeuralNetwork<Signal>) -> Self
    where
        Signal: LinalgScalar,
    {
        Self {
            layers: network
                .layers
                .iter()
                .map(|layer| LayerCache {
                    inputs: Array1::from_elem(layer.len(), Signal::zero()),
                    outputs: Array1::from_elem(layer.len(), Signal::zero()),
                    error: Array1::from_elem(layer.len(), Signal::zero()),
                })
                .collect(),
        }
    }
}

fn one_hot<T: LinalgScalar>(index: usize, size: usize) -> Array1<T> {
    Array1::from_shape_fn(size, |i| if i == index { T::one() } else { T::zero() })
}

fn main() -> Result<(), std::io::Error> {
    let mut network = NeuralNetwork::<f32>::new([28 * 28, 100, 100, 100, 50, 10].iter().cloned());
    let mut cache = NeuralNetworkCache::new(&network);
    let images = mnist::Image::<f32>::load("train")?;
    for i in 1..6 {
        println!("Training on {:?} images", images.len());
        for image in images.iter() {
            // println!("{}", image);
            // println!("Before training");
            // let output = network.forward::<activation::LeakyRelU>(&mut cache, image.image.view());
            // println!("{:?}", output);
            network.train::<activation::LeakyRelU>(
                &mut cache,
                &image.image,
                &one_hot(image.classification as _, 10),
                0.01 * (1.0 / i as f32),
            );
            // println!("After training");
            // let output = network.forward::<activation::LeakyRelU>(&mut cache, image.image.view());
            // println!("{:?}", output);
        }
    }
    let images = mnist::Image::<f32>::load("t10k")?;
    println!("Evaluating on {:?} images", images.len());
    let mut correct = 0;
    let mut incorrect = 0;
    for image in images.iter() {
        let output = network.forward::<activation::LeakyRelU>(&mut cache, &image.image);
        let result = softmax(output)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        if result == image.classification as usize {
            correct += 1;
        } else {
            println!("{}", image);
            println!("Prediction: {}", result);
            incorrect += 1;
        }
    }
    println!(
        "Correct: {}, Incorrect: {}, {}%",
        correct,
        incorrect,
        correct as f64 / (correct + incorrect) as f64 * 100.0
    );
    Ok(())
}
