use alga::linear::{VectorSpace, FiniteDimVectorSpace};
use alga::general::{RealField, RingCommutative, Module, Ring, DynamicModule};
use approx::RelativeEq;
use nalgebra::{Scalar, Matrix, DimName, MatrixN, VectorN, DimNameMul, MatrixMN, Matrix5};
use std::ops::{Mul, AddAssign};
use num_traits::Zero;
use alga::general::{Identity, Additive};
use nalgebra::{NamedDim, U1, U6};
use nalgebra::base::storage::Storage;

use nalgebra::{ArrayStorage, SliceStorage};
use nalgebra::base::iter::RowIter;
use std::borrow::Borrow;

pub enum ODEState{
    Ok,
    Done,
    Err
}

#[derive(Debug)]
pub struct ButcherTableu<T: Scalar, S: DimName, S1: Storage<T, S, S>,
                    S2: Storage<T, S, U1>> {
    ac: Matrix<T, S, S, S1>,
    b: Matrix<T, S, U1, S2>,
    b_err: Option<Matrix<T, S, U1, S2>>
}

pub type ButcherTableuSlices<'a, T, S> =
    ButcherTableu<T, S, SliceStorage<'a, T,  S, S,  S, U1>, SliceStorage<'a, T,  S, U1,  U1, U1>>;


impl<'a, T, S > ButcherTableuSlices<'a, T, S>
where T: Scalar, S:DimName{
    pub fn from_slices(ac: &[T], b: &[T], b_err: Option<&[T]>, s: S) -> Self{
        unsafe{
            //
            //Some(Matrix::from_data(SliceStorage::from_raw_parts(
            //    b_err.map(|b| b.as_ptr() ), (S, U1), (U1, U1))))
            ButcherTableu{
                ac: Matrix::from_data(SliceStorage::from_raw_parts(ac.as_ptr(),(s, s),(s, U1))),
                b: Matrix::from_data(SliceStorage::from_raw_parts(b.as_ptr(), (s, U1), (U1, U1))),
                b_err:  b_err.map(|b|
                                      Matrix::from_data(SliceStorage::from_raw_parts(
                                              b.as_ptr(), (s, U1), (U1, U1)))
                                 )
            }
        }
    }
}

impl<T: Scalar, D: DimName, S1: Storage<T, D, D>,
    S2: Storage<T, D, U1>> ButcherTableu<T, D, S1, S2>{
    pub fn num_stages(&self) -> usize{
        self.b.len()
    }

    pub fn ac_iter(&self) -> RowIter<T, D, D, S1>{
        self.ac.row_iter()
    }

    pub fn b_iter(&self) -> RowIter<T, D, U1, S2>{
        self.b.row_iter()
    }

    pub fn b_err_iter(&self) -> Option<RowIter<T, D, U1, S2>>{
        match &self.b_err{
            None => None,
            Some(b) => Some(b.row_iter())
        }

    }
}

pub trait ODESolverBase{
    type TField: RealField;
    type RangeType: DynamicModule;

    fn try_step(&mut self, dt: Self::TField) -> Result<(), ()>;
    fn accept_step(&mut self);
}

pub trait ODESolver{
    type TField: RealField;
    type RangeType: DynamicModule;

    fn step(&mut self) -> ODEState;

    fn current(&self) -> (Self::TField, & Self::RangeType);

}

pub trait AdaptiveODESolver: ODESolverBase{

}

pub fn check_step<T : RealField>(t0: T, tf: T, dt: T) -> Option<T>{
    let rem_t: T = tf.clone() - t0.clone();
    if rem_t.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative()){
        return None;
    }
    if rem_t.clone() < dt.clone(){
        return Some(rem_t);
    } else {
        return Some(dt);
    }
}
