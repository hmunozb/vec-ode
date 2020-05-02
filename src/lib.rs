#![allow(dead_code)]
#![allow(non_snake_case)]

#[macro_use]
extern crate ndarray;

mod base;
mod dat;
pub mod exp;
pub mod impls;
pub mod diff;
pub use base::*;
use std::fmt::Display;
use num_traits::{FromPrimitive, Num};
use num_complex::{Complex64, Complex};

pub trait RealField :
num_traits::real::Real
+ num_traits::NumAssign
+ Display
+ num_traits::FromPrimitive
+ approx::RelativeEq
{ }

impl<T> RealField for T
where T: num_traits::real::Real
+ num_traits::NumAssign
+ Display
+ num_traits::FromPrimitive
+ approx::RelativeEq
{ }

mod macros{
    #[macro_export]
    macro_rules! from_f64 {
        ($N : expr ) => {
            FromPrimitive::from_f64($N).unwrap()
        };
        ($T: ty, $N : expr ) => {
            <$T as num_traits::FromPrimitive>::from_f64($N).unwrap()
        };
    }
}

fn from_c64<T: Clone + Num + FromPrimitive>(z: Complex64) -> Complex<T>{
    Complex::<T>::new(from_f64!(z.re), from_f64!(z.im))
}