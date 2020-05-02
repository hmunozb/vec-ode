use alga::general::{ClosedAdd, ClosedMul, ClosedSub};
use nalgebra::{Scalar,Dim, Matrix};




use crate::{LinearCombinationSpace};
use nalgebra::base::storage::StorageMut;

impl<N, R, C, S, > LinearCombinationSpace<N> for Matrix<N, R, C, S>
where N: Scalar + ClosedAdd + ClosedSub + ClosedMul,
      R: Dim, C: Dim, S: StorageMut<N, R, C>
{
    fn scale(&mut self, k: N) {
        *self *= k;
    }

    fn scalar_multiply_to(&self, k: N, target: &mut Self) {
        for (s, t) in self.iter().zip(target.iter_mut()){
            *t = *s * k;
        }
    }

    fn add_scalar_mul(&mut self, k: N, other: &Self) {
        for (s, t) in self.iter_mut().zip(other.iter()){
            *s = k * *t + *s ;
        }
    }

    fn add_assign_ref(&mut self, other: &Self) {
        *self += other;
    }

    fn delta(&mut self, y: &Self) {
        *self -= y;
    }
}