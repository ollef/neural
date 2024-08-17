use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Neg},
};

use itertools::zip_eq;
use num_traits::{Float, FromPrimitive, One, Zero};
use rand_distr::{Distribution, Normal, StandardNormal};

pub fn dot<T: Mul<Output = T> + Sum>(a: impl Iterator<Item = T>, b: impl Iterator<Item = T>) -> T {
    zip_eq(a, b).map(|(x, y)| x * y).sum()
}

struct Neuron<Signal> {
    pub bias: Signal,
    pub weights: Vec<Signal>,
}

trait Activation {
    type Signal;
    fn activate(input: Self::Signal) -> Self::Signal;
}

impl<Signal: Clone + Add<Output = Signal> + Mul<Output = Signal> + Sum> Neuron<Signal> {
    fn forward<A: Activation<Signal = Signal>>(
        &self,
        inputs: impl Iterator<Item = Signal>,
    ) -> Signal {
        A::activate(dot(inputs, self.weights.iter().cloned()) + self.bias.clone())
    }
}

struct Sigmoid<Signal>(std::marker::PhantomData<Signal>);
struct RelU<Signal>(std::marker::PhantomData<Signal>);

struct LeakyRelU<Signal>(std::marker::PhantomData<Signal>);

impl<Signal: One + Div<Output = Signal> + Neg<Output = Signal> + Float> Activation
    for Sigmoid<Signal>
{
    type Signal = Signal;

    fn activate(input: Signal) -> Signal {
        Signal::one() / (Signal::one() + (-input).exp())
    }
}

impl<Signal: Ord + Zero> Activation for RelU<Signal> {
    type Signal = Signal;

    fn activate(input: Signal) -> Signal {
        input.max(Signal::zero())
    }
}

impl<Signal: Ord + Zero + Mul<Output = Signal> + FromPrimitive> Activation for LeakyRelU<Signal> {
    type Signal = Signal;

    fn activate(input: Signal) -> Signal {
        if input > Signal::zero() {
            input
        } else {
            Signal::from_f64(0.01).unwrap() * input
        }
    }
}

struct Layer<Signal> {
    pub neurons: Vec<Neuron<Signal>>,
}

struct NeuralNetwork<Signal> {
    pub layers: Vec<Layer<Signal>>,
}

impl<Signal> Layer<Signal> {
    pub fn forward<'a, A: Activation<Signal = Signal>>(
        &'a self,
        inputs: impl Iterator<Item = Signal> + Clone + 'a,
    ) -> impl Iterator<Item = Signal> + 'a
    where
        Signal: Clone + Add<Output = Signal> + Mul<Output = Signal> + Sum,
    {
        self.neurons
            .iter()
            .map(move |neuron| neuron.forward::<A>(inputs.clone()))
    }
}

impl<Signal> NeuralNetwork<Signal> {
    pub fn new(layer_input_sizes: impl Iterator<Item = usize> + Clone) -> Self
    where
        Signal: Zero + One + Float + FromPrimitive + Div<Output = Signal> + Mul<Output = Signal>,
        StandardNormal: Distribution<Signal>,
    {
        let normal = Normal::new(Signal::zero(), Signal::one()).unwrap();
        let two = Signal::one() + Signal::one();
        Self {
            layers: layer_input_sizes
                .clone()
                .zip(layer_input_sizes.skip(1))
                .map(|(input_size, output_size)| Layer {
                    neurons: (0..output_size)
                        .map(|_| Neuron {
                            weights: (0..input_size)
                                .map(|_| {
                                    normal.sample(&mut rand::thread_rng())
                                        * (two / Signal::from_usize(input_size).unwrap()).sqrt()
                                })
                                .collect(),
                            bias: Signal::zero(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    pub fn forward<A: Activation<Signal = Signal>>(&self, inputs: Vec<Signal>) -> Vec<Signal>
    where
        Signal: Clone + Add<Output = Signal> + Mul<Output = Signal> + Sum,
    {
        self.layers.iter().fold(inputs, |inputs, layer| {
            layer.forward::<A>(inputs.iter().cloned()).collect()
        })
    }
}

fn main() {
    let network = NeuralNetwork::<f32>::new([20 * 20, 100, 10].iter().cloned());
    println!("Hello, world!");
}
