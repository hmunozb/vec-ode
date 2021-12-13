#![cfg(feature="nalgebra")]

use nalgebra::{ClosedAdd, ClosedMul, ClosedSub};
use nalgebra::{Scalar,Dim, Matrix};




use crate::lc::LinearCombinationSpace;
use nalgebra::base::storage::StorageMut;

impl<N, R, C, S, > LinearCombinationSpace<N> for Matrix<N, R, C, S>
where N: Copy + Scalar + ClosedAdd + ClosedSub + ClosedMul,
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


#[cfg(test)]
#[cfg(feature = "nalgebra")]
mod tests{
    use super::*;
    use crate::base::{ODESolver, AdaptiveODESolver};
    use crate::base::{RK45RealSolver, RK45ComplexSolver};
    use nalgebra::{Vector2, DVector};
    use num_complex::Complex64 as c64;
    use crate::ODEState;

    #[test]
    fn test_rk45_1(){
        let g = |t: f64, x: & Vector2<c64>,  y: &mut Vector2<c64>|{
            y[0] = - x[0];
            y[1] =  x[1] * -2.0;
            Ok(())
        };

        let x0 = Vector2::new(c64::from(1.0), c64::from(1.0));
        let mut solver = RK45ComplexSolver::new(g, 0., 2., x0.clone(), 0.0001);
        while let ODEState::Ok(_) = solver.step() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);
        //println!("RK45 Butche Tableu Data: {}", rk45_tableu.ac);

    }

    #[test]
    fn test_rk45_2(){
        let g = |t: f64, x: & DVector<f64>,  y: &mut DVector<f64>|{
            y[0] = - x[0];
            y[1] = -2.0 * x[1];
            Ok(())
        };
        let x0 = DVector::from_column_slice(&[1.0, 1.0]);
        let mut solver = RK45RealSolver::new(g, 0., 2., x0.clone(), 0.0001);
        while let ODEState::Ok(_) = solver.step() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);
        //println!("RK45 Butche Tableu Data: {}", rk45_tableu.ac);

    }

    #[test]
    fn test_rk45_f64(){
        let g = |t: f64, x: & f64,  y: &mut f64|{
            *y = - *x;
            Ok(())
        };
        let x0 :f64 = 1.0;
        let mut solver = RK45RealSolver::new(g, 0., 2., x0, 0.0001)
            .with_tolerance(1.0e-10, 1.0e-10);
        while let ODEState::Ok(_) = solver.step_adaptive() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);

    }

}