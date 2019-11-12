use std::ops::{AddAssign, MulAssign};
use itertools::Itertools;
use num_traits::Zero;
use alga::general::{RealField, ClosedAdd};
use alga::general::{Ring};
use ndarray::{Ix1, Array1, Array2};
use super::ode::{ODESolver, ODEState};
use std::marker::PhantomData;
use crate::{ODESolverBase, ODEData, ODEError, ODEStep};

use crate::dat::rk::{rk45_ac, rk45_b, rk45_berr};
use ndarray::iter::Lanes;
use super::num_complex::Complex;

pub trait LinearCombination<S>: Sized{
    fn scale(&mut self, k: S);
    fn scalar_multiply_to(&self, k: S, target: &mut Self);
    fn add_scalar_mul(&mut self, k: S, other: &Self);
    fn add_assign_ref(&mut self, other: &Self);
    /// Subtracts the vector y from self
    fn delta(&mut self, y: &Self);
}


#[derive(Debug)]
pub struct ButcherTableu<T: RealField> {
    ac: Array2<T>,
    b:Array1<T>,
    b_err: Option<Array1<T>>
}

impl<T: RealField> ButcherTableu<T>
{
    pub fn from_slices<T2>(ac: &[T2], b: &[T2], b_err: Option<&[T2]>, s: usize) -> Self
    where T2: RealField + Into<T>,
    {
        ButcherTableu{
            ac: Array2::from_shape_vec((s,s),
                Vec::from(ac).into_iter().map_into().collect_vec()).unwrap(),
            b: Array1::from_shape_vec(s,
                Vec::from(b).into_iter().map_into().collect_vec()).unwrap(),
            b_err: b_err.map(|b|Array1::from_shape_vec(s,
                Vec::from(b).into_iter().map_into().collect_vec()).unwrap())
        }
    }
}

impl<T: RealField> ButcherTableu<T>{
    pub fn num_stages(&self) -> usize{
        self.b.len()
    }

    pub fn ac_iter(&self) -> Lanes<T, Ix1> {
        self.ac.genrows()
    }

    pub fn b_iter(&self) -> ndarray::iter::Iter<T, Ix1>{
        self.b.iter()
    }

    pub fn b_err_iter(&self) -> Option<ndarray::iter::Iter<T, Ix1>>{
        self.b_err.as_ref().map(|b| b.iter())
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
/// The scalar field S needs to be hacked in via PhantomData because dynamically allocated
/// vectors do not have a compile time dimension, and so cannot implement Zero, Module, VectorSpace...
/// at compile time. Thus, the only required traits for V are Clone, AddAssign with &V, and MulAssign
/// with S
fn rk_step<Fun, S, T, V>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, x_err: Option<&mut V>,
    dt: T, tabl: &ButcherTableu<T>,  K: &mut Vec<V>, _phantom: PhantomData<S>)
        -> Result<(), ODEError>
    where T: RealField + Into<S>,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          S: Ring + Copy,
          for <'b> V: AddAssign<&'b V>,
          V: Clone + MulAssign<S>
    {

    let _dt : S = dt.into();
    let _zero =  <S as Zero>::zero();
    //Check that the number of stages is consistent
    let k_len = K.len();
    let s = k_len - 1;
    if tabl.num_stages() != s {
        panic!("rk_step: number of stages in ButcherTableu and stages does not match")
    }

    //Calculate the first stage
    f(t, x0, K.get_mut(0).unwrap());

    //Split the K vector into the stages vector and the entry used as an arithmetic register
    let (k_stages, k_work) = K.split_at_mut(s);
    let k = k_work.get_mut(0).unwrap();

    //Iterate over the tableu, skipping the first trivial entry
    for (i, ac) in tabl.ac_iter().into_iter().enumerate().skip(1){
        let ti: T = t + (ac[i]) * dt;
        // Calculate x for the current stage and store in xf
        *xf *= _zero;
        for j in 0..i{
            //*k *= _zero;
            //*k += k_stages.get(j).unwrap();
            k.clone_from(k_stages.get(j).unwrap());
            *k *= ac[j].into();
            *xf += &*k;
        }
        *xf *= _dt;
        *xf += x0;
        let ki = k_stages.get_mut(i).unwrap();
        //Evaluate dx for this stage and store in ki
        f(ti, xf, ki);
    };

    //Finally calculate xf
    *xf *= _zero;
    for (b, ki) in tabl.b_iter()
        .zip(k_stages.iter() ){
        k.clone_from(ki);
        *k *= (*b).into();
        *xf += &*k;
    }
    *xf *= _dt;
    *xf += x0;

    //Also handle x_err if the butcher tableu contains error terms
    match tabl.b_err_iter(){
        None => {},
        Some(e) => {
            match x_err{
                None => {},
                Some(xe) =>{
                    *xf *= _zero;

                    for (b, ki) in e
                            .zip(k_stages.iter()){
                        k.clone_from(ki);
                        *k *=  (*b).into();
                        *xf += &*k;
                    }
                    *xf *= _dt;
                    *xf += x0;
                }
            }
        }
    }

    Result::Ok(())

}


pub struct RK45Solver<V,Fun,S,T=f64>
    where   T: RealField,
            Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
            S : Copy,
            V: Clone,
{
    f: Fun,
    dat: ODEData<T, V>,
    x_err: V,
    h: T,
    tabl: ButcherTableu<T>,
    K: Vec<V>,
    _phantom: PhantomData<S>
}

pub type RK45RealSolver<V, Fun, T> = RK45Solver<V, Fun, T, T>;
pub type RK45ComplexSolver<V, Fun, T> = RK45Solver<V, Fun, Complex<T>, T>;

impl<'a,V,S,Fun> RK45Solver<V,Fun,S,f64>
    where
        Fun: FnMut(f64, &V, &mut V) -> Result<(),()>,
        V: Clone + MulAssign<S>,
        S: Ring + From<f64> + Copy{

    pub fn new(f: Fun, t0: f64, tf: f64, x0: V, h: f64 ) -> Self{

        let x_err = x0.clone();
        let tabl = ButcherTableu::from_slices(&rk45_ac, &rk45_b, Some(&rk45_berr), 6);
        let mut K: Vec<V> = Vec::new();
        K.resize(7, x0.clone());
        let dat = ODEData::new(t0, tf, x0);
        RK45Solver{f, dat, x_err, h, tabl, K, _phantom: PhantomData}
    }
}

impl<V,Fun,S,T> ODESolverBase for RK45Solver<V,Fun,S,T>
    where T: RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          for <'b> V: AddAssign<&'b V>,
          V: Clone + MulAssign<S>,
          S: Ring + From<T> + Copy{
    type TField=T;
    type RangeType=V;

    fn ode_data(&self) -> &ODEData<T, V>{
        &self.dat
    }
    fn ode_data_mut(&mut self) -> &mut ODEData<T, V>{
        &mut self.dat
    }
    fn into_ode_data(self) -> ODEData<T, V>{
        self.dat
    }

    fn step_size(&self) -> ODEStep<T>{
        self.dat.step_size(self.h.clone())
    }

    fn try_step(&mut self, dt: T) -> Result<(), ODEError>{
        let dat = &mut self.dat;
//        dat.next_dt = dt;
        let res = rk_step(&mut self.f, dat.t.clone(), &dat.x,
                          &mut dat.next_x, Some(&mut self.x_err),
                          dat.next_dt.clone(), &self.tabl, &mut self.K,
                          PhantomData::<S>);
        res
    }

}
impl<V,Fun,S,T> ODESolver for RK45Solver<V,Fun,S,T>
    where T: RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          for <'b> V: AddAssign<&'b V>,
          V: Clone + MulAssign<S>,
          S: Ring + From<T> + Copy{

}

pub struct RK45AdaptiveSolver<V,Fun,S,T=f64>
    where   T: RealField,
            Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
            for <'b> V: AddAssign<&'b V>,
            V: Clone ,
            S : From<T>+Copy
{
    f: Fun,
    dat: ODEData<T, V>,
    x_err: V,
    h: T,
    tabl: ButcherTableu<T>,
    K: Vec<V>,
    _phantom: PhantomData<S>
}

impl<V,Fun,S> RK45AdaptiveSolver<V,Fun,S,f64>
    where
        Fun: FnMut(f64, &V, &mut V) -> Result<(),()>,
        for <'b> V: AddAssign<&'b V>,
        V: Clone + MulAssign<S>,
        S: From<f64>+Copy{

    pub fn new(f: Fun, t0: f64, tf: f64, x0: V) -> Self{
        let h = (tf - t0) * 1.0e-5;
        let atol = 1.0e-6;
        let rtol = 1.0e-6;

        let x_err = x0.clone();
        let tabl = ButcherTableu::from_slices(&rk45_ac, &rk45_b, Some(&rk45_berr), 6);
        let mut K: Vec<V> = Vec::new();
        K.resize(7, x0.clone());
        let dat = ODEData::new(t0, tf, x0);
        RK45AdaptiveSolver{f, dat, x_err, h, tabl, K, _phantom: PhantomData}
    }

    pub fn with_tolerance(atol: f64, rtol: f64){

    }

    pub fn with_init_step(h: f64){

    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use nalgebra::{Vector2, DVector};
    use num_complex::Complex64 as c64;

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
        let mut solver = RK45RealSolver::new(g, 0., 2., x0, 0.0001);
        while let ODEState::Ok(_) = solver.step() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);

    }

}