mod activation;

use std::ops::{Sub, SubAssign};

use itertools::zip_eq;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, LinalgScalar, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand_distr::{Distribution, Normal, StandardNormal};

trait Activation<Signal> {
    fn function(input: Signal) -> Signal;
    fn derivative(input: Signal) -> Signal;
}

struct Layer<Signal> {
    pub neuron_weights: Array2<Signal>,
}

struct NeuralNetwork<Signal> {
    pub layers: Vec<Layer<Signal>>,
}

struct LayerCache<Signal> {
    outputs_1: Array1<Signal>,
    error_1: Array1<Signal>,
}

struct NeuralNetworkCache<Signal> {
    layers: Vec<LayerCache<Signal>>,
}

impl<Signal> Layer<Signal> {
    pub fn forward<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut LayerCache<Signal>,
        inputs_1: ArrayView1<Signal>,
    ) -> ArrayView1<'a, Signal>
    where
        Signal: LinalgScalar,
    {
        let mut outputs = cache.outputs_1.slice_mut(s![..-1]);
        outputs.assign(&self.neuron_weights.dot(&inputs_1));
        outputs.mapv_inplace(A::function);
        cache.outputs_1.slice(s![..])
    }

    pub fn backpropagate<A: Activation<Signal>>(
        &mut self,
        cache: &mut LayerCache<Signal>,
        inputs_1: &ArrayView1<Signal>,
        next_layer_neuron_weights: &ArrayView2<Signal>,
        next_layer_error1: &ArrayView1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand,
    {
        let mut error_1 = cache.error_1.slice_mut(s![..]);
        error_1.assign(&next_layer_neuron_weights.t().dot(next_layer_error1));
        let mut error = error_1.slice_mut(s![..-1]);
        error.mapv_inplace(A::derivative);
        let weight_gradient = error_1
            .insert_axis(Axis(0))
            .dot(&inputs_1.insert_axis(Axis(1)));
        self.neuron_weights
            .assign(&(&self.neuron_weights - &weight_gradient.t() * learning_rate));
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
                    neuron_weights: Array2::from_shape_fn(
                        (output_size, input_size + 1),
                        |(_, col)| {
                            if col == input_size {
                                Signal::zero()
                            } else {
                                normal.sample(&mut rand::thread_rng())
                                    * (two / Signal::from_usize(input_size).unwrap()).sqrt()
                            }
                        },
                    ),
                })
                .collect(),
        }
    }

    pub fn forward<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut NeuralNetworkCache<Signal>,
        inputs_1: ArrayView1<'a, Signal>,
    ) -> ArrayView1<'a, Signal>
    where
        Signal: LinalgScalar,
    {
        zip_eq(self.layers.iter(), cache.layers.iter_mut())
            .fold(inputs_1, |inputs, (layer, cache)| {
                layer.forward::<A>(cache, inputs)
            })
    }

    pub fn backpropagate<A: Activation<Signal>>(
        &mut self,
        cache: &mut NeuralNetworkCache<Signal>,
        inputs_1: &ArrayView1<Signal>,
        expected_outputs: &ArrayView1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand,
    {
        let outputs = cache.layers.last().unwrap().outputs_1.slice(s![..-1]);
        let error_1 = Array1::from_shape_fn(outputs.len() + 1, |i| {
            if i == outputs.len() {
                Signal::zero()
            } else {
                expected_outputs[i] - outputs[i]
            }
        });
        let output_neuron_weights = Array2::from_elem(
            (
                self.layers.last().unwrap().neuron_weights.nrows(),
                self.layers.last().unwrap().neuron_weights.nrows() + 1,
            ),
            Signal::one(),
        );

        for i in (0..self.layers.len()).rev() {
            let (_, [layer, post_layers @ ..]) = self.layers.split_at_mut(i) else {
                panic!("Couldn't split layers");
            };
            let (pre_caches, [layer_cache, post_caches @ ..]) = cache.layers.split_at_mut(i) else {
                panic!("Couldn't split cache layers");
            };
            let inputs_1 = pre_caches.last().map_or(inputs_1.slice(s![..]), |cache| {
                cache.outputs_1.slice(s![..])
            });
            let next_layer_neuron_weights = post_layers
                .first()
                .map_or(output_neuron_weights.slice(s![.., ..]), |layer| {
                    layer.neuron_weights.slice(s![.., ..])
                });
            let next_layer_error1 = post_caches
                .first()
                .map_or(error_1.slice(s![..]), |cache| cache.error_1.slice(s![..]));
            layer.backpropagate::<A>(
                layer_cache,
                &inputs_1,
                &next_layer_neuron_weights,
                &next_layer_error1,
                learning_rate,
            )
        }
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
                    outputs_1: Array1::from_shape_fn(layer.neuron_weights.nrows() + 1, |i| {
                        if i == layer.neuron_weights.nrows() {
                            Signal::one()
                        } else {
                            Signal::zero()
                        }
                    }),
                    error_1: Array1::zeros(layer.neuron_weights.nrows() + 1),
                })
                .collect(),
        }
    }
}

fn main() {
    let network = NeuralNetwork::<f64>::new([20 * 20, 100, 100, 10].iter().cloned());
    let cache = NeuralNetworkCache::new(&network);
    println!("Hello, world!");
}
