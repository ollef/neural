mod activation;

use std::ops::SubAssign;

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
    outputs: Array1<Signal>,
    error: Array1<Signal>,
}

struct NeuralNetworkCache<Signal> {
    levels: Vec<LayerCache<Signal>>,
}

impl<Signal> Layer<Signal> {
    pub fn forward<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut LayerCache<Signal>,
        inputs: ArrayView1<Signal>,
    ) -> ArrayView1<'a, Signal>
    where
        Signal: LinalgScalar,
    {
        let mut result = cache.outputs.slice_mut(s![..-1]);
        result.assign(&self.neuron_weights.dot(&inputs));
        result.mapv_inplace(A::function);
        cache.outputs.slice(s![..])
    }

    pub fn backpropagate<'a, A: Activation<Signal>>(
        &mut self,
        next_layer_neuron_weights: &ArrayView2<Signal>,
        inputs: &ArrayView1<Signal>,
        cache: &'a mut LayerCache<Signal>,
        next_layer_error: &ArrayView1<Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar + ScalarOperand + SubAssign,
    {
        let error = &mut cache.error;
        error.assign(&next_layer_neuron_weights.t().dot(next_layer_error));
        error.mapv_inplace(A::derivative);
        let weight_gradient = error.dot(&inputs.insert_axis(Axis(0)));
        self.neuron_weights
            .assign(&(&self.neuron_weights - weight_gradient * learning_rate));
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
        inputs: ArrayView1<'a, Signal>,
    ) -> ArrayView1<'a, Signal>
    where
        Signal: LinalgScalar,
    {
        zip_eq(self.layers.iter(), cache.levels.iter_mut())
            .fold(inputs, |inputs, (layer, cache)| {
                layer.forward::<A>(cache, inputs)
            })
    }

    pub fn backpropagate<'a, A: Activation<Signal>>(
        &self,
        cache: &'a mut NeuralNetworkCache<Signal>,
        targets: ArrayView1<'a, Signal>,
        learning_rate: Signal,
    ) where
        Signal: LinalgScalar,
    {
    }
}

impl<Signal> NeuralNetworkCache<Signal> {
    pub fn new(network: &NeuralNetwork<Signal>) -> Self
    where
        Signal: LinalgScalar,
    {
        Self {
            levels: network
                .layers
                .iter()
                .map(|layer| LayerCache {
                    outputs: Array1::from_shape_fn(layer.neuron_weights.nrows() + 1, |i| {
                        if i == layer.neuron_weights.nrows() {
                            Signal::one()
                        } else {
                            Signal::zero()
                        }
                    }),
                    error: Array1::zeros(layer.neuron_weights.nrows()),
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
