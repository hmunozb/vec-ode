use alga::general::{ClosedAdd, ClosedMul, ClosedSub};
use ndarray::{ArrayBase, DataMut, Dimension, ScalarOperand};

use crate::LinearCombination;

impl<A, S, D> LinearCombination<A> for ArrayBase<S, D>
where   A: Copy + ClosedAdd + ClosedSub + ClosedMul + ScalarOperand,
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