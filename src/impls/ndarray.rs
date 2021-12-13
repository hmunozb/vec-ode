//use alga::general::{ClosedAdd, ClosedMul, ClosedSub};
use std::ops::{Add, Mul, AddAssign, MulAssign, SubAssign};
use ndarray::LinalgScalar;
use ndarray::{ArrayBase, DataMut, Dimension, ScalarOperand};

use crate::lc::LinearCombinationSpace;

impl<A, S, D> LinearCombinationSpace<A> for ArrayBase<S, D>
where   A: Copy + Add<Output=A> + AddAssign + SubAssign + Mul<Output=A> + MulAssign + ScalarOperand,
        S: DataMut<Elem=A>,
        D: Dimension
{
    #[inline]
    fn scale(&mut self, k: A) {
        *self *= k;
    }

    fn scalar_multiply_to(&self, k: A, target: &mut ArrayBase<S, D>) {
        target.zip_mut_with(self, | t, &s| *t = k * s);
    }

    fn add_scalar_mul(&mut self, k: A, rhs: &ArrayBase<S, D>) {
        self.zip_mut_with(rhs, move |y, x| *y = *y + (k * *x));
    }

    fn add_assign_ref(&mut self, other: &Self){
        *self += other;
    }

    fn delta(&mut self, y: &Self) {
        *self -= y;
    }
}