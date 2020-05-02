pub mod split_exp;
pub mod magnus;
pub mod cfm;

use crate::RealField;



/// Trait to define an exponential split for operator splitting solvers
/// The linear operators must have linear combinations defined
pub trait ExponentialSplit<T, S, V>
where T:  RealField,
      S:  Copy + From<T>,
      V: Clone
{
    type L: Clone ;//+ LinearCombinationSpace<S>;  //+ MulAssign<S>;
    type LC: LinearCombination<S, Self::L>;
    type U: Sized;

    fn lin_zero(&self) -> Self::L;

    /// Returns the exponential of the linear operator
    fn exp(&mut self, l: Self::L) -> Self::U;
    /// Applies the exponential on a vector x
    fn map_exp(&mut self, u: &Self::U, x: & V) -> V;

    /// Evaluate multiple exponentials exp(k l) of rescalings of l
    fn multi_exp(&mut self, l: Self::L, k_arr: &[S]) -> Vec<Self::U>{
        k_arr.iter().map(|&k| {
            let mut l2= l.clone();
            Self::LC::scale(&mut l2, k );
            self.exp(l2) })
             .collect_vec()
    }
}

pub trait NormedExponentialSplit<T, S, V> : ExponentialSplit<T, S, V>
where T:  RealField,
      S:  Copy + From<T>,
      V: Clone
{
    /// Calculates the norm of a vector x
    fn norm(&self, x: &V) -> T;

}

pub trait Commutator<T, S, V> : ExponentialSplit<T, S, V>
where T:  RealField,
      S:  Copy + From<T>,
      V: Clone
{
    /// Compute the commutator of the two linear operators
    fn commutator(&self, la: &Self::L, lb: &Self::L) -> Self::L;
}



pub use split_exp::*;
pub use magnus::MidpointExpLinearSolver;
use itertools::Itertools;
use crate::LinearCombination;