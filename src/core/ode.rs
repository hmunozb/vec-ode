use alga::linear::{VectorSpace, FiniteDimVectorSpace};
use alga::general::{RealField, RingCommutative, Module, Ring};
use nalgebra::{Scalar, Matrix, DimName, MatrixN, VectorN, DimNameMul, MatrixMN, Matrix5};
use std::ops::{Mul, AddAssign};
use num_traits::Zero;
use alga::general::{Identity, Additive};
use nalgebra::{NamedDim, U1, U6};
use nalgebra::base::storage::Storage;

use nalgebra::{ArrayStorage, SliceStorage};
use std::borrow::Borrow;

enum ODEState{
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
    fn num_stages(&self) -> usize{
        self.b.len()
    }
}

/// Implements a single step of the Runge-Kutta algorithm on the following data
///     t : The current time (type T: RealField)
///     x0 : the current vector (type V: VectorSpace)
///     dt : Time step size (type T)
///     f : A function with signature Fn(T t, &V x, &mut V dx_dt) -> Result<,>
///         f should store the derivative dx/dt = f(x, t) to the mutable reference dx_dt
///     stages : A vector with length s+1. The stages are stored in stages[0:s-1] while stages[s]
///             is used for working arithmetic (will be unnecessary with add_scalar_multiple
///             available as an operation)
///
fn rk_step<Fun, T, V, D, S1, S2>(
    f: &Fun, t: T, x0: &V, xf: &mut V, x_err: Option<&mut V>,
            dt: T, tabl: &ButcherTableu<T, D, S1, S2>,  K: &mut Vec<V>) -> Result<(),()>
    where D: DimName, T: Scalar+Ring+Copy,
          V: Module,
          Fun: Fn(T, &V, &mut V) -> Result<(),()>,
          S1: Storage<T, D, D>, S2: Storage<T, D, U1>,
          V::Ring : From<T>+Copy,
          for <'b> V: AddAssign<&'b V>{

    let _dt = V::Ring::from(dt);
    let _zero =  <V::Ring as Zero>::zero();
    //Check that the number of stages is consistent
    let k_len = K.len();
    let s = k_len - 1;
    if tabl.num_stages() != s {
        panic!("rk_step: number of stages in ButcherTableu and stages does not match")
    }

    //Calculate the first stage
    f(t, x0, K.get_mut(0).unwrap())?;

    //Split the K vector into the stages vector and the entry used as an arithmetic register
    let (k_stages, k_work) = K.split_at_mut(s);
    let k = k_work.get_mut(0).unwrap();

    //Iterate over the tableu, skipping the first trivial entry
    for (i, ac) in tabl.ac.row_iter()
            .enumerate().skip(1){
        let ti: T = t + (*ac.get(i).unwrap() ) * dt;
        // Calculate x for the current stage and store in xf
        *xf *= _zero;
        for j in 0..i{
            //*k *= _zero;
            //*k += k_stages.get(j).unwrap();
            k.clone_from(k_stages.get(j).unwrap());
            *k *= V::Ring::from(*ac.index((0, j)));
            *xf += &*k;
        }
        *xf *= _dt.clone();
        *xf += x0;
        let ki = k_stages.get_mut(i).unwrap();
        //Evaluate dx for this stage and store in ki
        f(ti, xf, ki)?;
    };

    //Finally calculate xf
    *xf *= _zero.clone();
    for (b, ki) in tabl.b.row_iter()
            .zip(k_stages.iter()){
        //*k *= _zero;
        //*k += &*ki;
        k.clone_from(ki);
        *k *= V::Ring::from(*b.get(0).unwrap());
        *xf += &*k;
    }
    *xf *= _dt.clone();
    *xf += x0;

    //Also handle x_err if the butcher tableu contains error terms
    match &tabl.b_err{
        None => {},
        Some(e) => {
            match x_err{
                None => {},
                Some(xe) =>{
                    *xf *= _zero;
                    for (b, ki) in e.row_iter()
                            .zip(k_stages.iter()){
                        k.clone_from(ki);
                        *k *=  V::Ring::from(*b.get(0).unwrap());
                        *xf += &*k;
                    }
                    *xf *= _dt.clone();
                    *xf += x0;
                }
            }
        }
    }

    Result::Ok(())

}

trait ODESolver{
    type TField: RealField;
    type RangeType: Module;

    fn step(&mut self) -> ODEState;

    fn current(&self) -> (Self::TField, & Self::RangeType);

}

static rk45_ac : [f64; 36] = [
                            0.,                     0.,0.,0.,0.,0.,
    1./4.,
                            1./4.,                  0.,0.,0.,0.,
    3.0/32.,     9.0/32.,
                            3./8.,                  0.,0.,0.,
    1932./2197., -7200./2197., 7296./2197.,
                            12./13.,                0.,0.,
    439./216.,   -8.,          3680./513.,   -845./4104.,
                            1.0,                    0.,
    -8./27.,     2.,           -3544./2526., 1859./4104.,    -11./40.,
                            1.0/2.0 ];

static rk45_b: [f64; 6] = [
    16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55.];

static rk45_berr: [f64; 6] = [
    25./216.,0.,1408./2565.,2197./4104., -1./5., 0.
];

pub struct RK45Solver<'a,V,Fun,T=f64>
where   T:Scalar+RealField,
        Fun: Fn(T, &V, &mut V) -> Result<(),()>,
        V: Module,
        V::Ring : From<T>+Copy,
        for <'b> V: AddAssign<&'b V>
{
    f: Fun,
    t0: T,
    tf: T,
    x0: V,

    t:  T,
    x: V,
    next_x: V,
    x_err: V,
    h: T,
    tabl: ButcherTableuSlices<'a, T, U6>,
    K: Vec<V>
}

impl<'a,V,Fun> RK45Solver<'a,V,Fun,f64>
where
      Fun: Fn(f64, &V, &mut V) -> Result<(),()>,
      V: Module,
      V::Ring : From<f64>+Copy,
      for <'b> V: AddAssign<&'b V>{

    pub fn new(f: Fun, t0: f64, tf: f64, x0: V, h: f64 ) -> Self{
        let x = x0.clone();
        let next_x = x0.clone();
        let x_err = x0.clone();
        let t = t0.clone();
        let tabl = ButcherTableuSlices::from_slices(&rk45_ac, &rk45_b, Some(&rk45_berr), U6);
        let mut K: Vec<V> = Vec::new();
        K.resize(7, x0.clone());

        RK45Solver{f, t0, tf, x0, t, x, next_x, x_err, h, tabl, K}
    }
}

impl<'a,V,Fun,T> ODESolver for RK45Solver<'a,V,Fun,T>
where T:Scalar+RealField,
      Fun: Fn(T, &V, &mut V) -> Result<(),()>,
      V: Module,
      V::Ring : From<T>+Copy,
      for <'b> V: AddAssign<&'b V>{
    type TField=T;
    type RangeType=V;

    fn step(&mut self) -> ODEState{
        let rem_t: T = self.tf.clone() - self.t.clone();
        let mut dt: T = T::zero();
        if rem_t.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative()){
            return ODEState::Done;
        }
        if rem_t.clone() < self.h.clone(){
             dt = rem_t.clone();
        } else {
            dt = self.h.clone();
        }
        let res = rk_step(&self.f, self.t.clone(), &self.x, &mut self.next_x, Some(&mut self.x_err),
                dt.clone(), &self.tabl, &mut self.K);

        self.x.clone_from(&self.next_x);
        self.t += dt;

        match res{
            Ok(()) => ODEState::Ok,
            Err(()) => ODEState::Err
        }
    }

    fn current(&self) -> (T, &V){
        (self.t.clone(), & self.x)
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use nalgebra::Vector2;
    #[test]
    fn test_rk45(){
        let g = |t: f64, x: & Vector2<f64>,  y: &mut Vector2<f64>|{
            y[0] = - x[0];
            y[1] = -2.0 * x[1];
            Ok(())
        };
        let x0 = Vector2::new(1.0, 1.0);
        let mut solver = RK45Solver::new(g, 0., 2., x0.clone(), 0.0001);
        while let ODEState::Ok = solver.step() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);
        //println!("RK45 Butche Tableu Data: {}", rk45_tableu.ac);

    }

}
