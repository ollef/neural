use std::ops::Neg;

use ndarray::LinalgScalar;
use num_traits::{Float, FromPrimitive};

trait Activation<Signal> {
    fn function(input: Signal) -> Signal;
    fn derivative(input: Signal) -> Signal;
}

struct Sigmoid;
struct RelU;

struct LeakyRelU;

impl<Signal: LinalgScalar + Neg<Output = Signal> + Float> Activation<Signal> for Sigmoid {
    fn function(input: Signal) -> Signal {
        Signal::one() / (Signal::one() + (-input).exp())
    }

    fn derivative(input: Signal) -> Signal {
        let sigmoid = Sigmoid::function(input);
        sigmoid * (Signal::one() - sigmoid)
    }
}

impl<Signal: LinalgScalar + Ord> Activation<Signal> for RelU {
    fn function(input: Signal) -> Signal {
        input.max(Signal::zero())
    }

    fn derivative(input: Signal) -> Signal {
        if input > Signal::zero() {
            Signal::one()
        } else {
            Signal::zero()
        }
    }
}

impl<Signal: LinalgScalar + Ord + FromPrimitive> Activation<Signal> for LeakyRelU {
    fn function(input: Signal) -> Signal {
        if input > Signal::zero() {
            input
        } else {
            Signal::from_f64(0.01).unwrap() * input
        }
    }

    fn derivative(input: Signal) -> Signal {
        if input > Signal::zero() {
            Signal::one()
        } else {
            Signal::from_f64(0.01).unwrap()
        }
    }
}
