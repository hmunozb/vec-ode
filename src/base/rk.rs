use std::ops::{AddAssign, MulAssign};
use num_traits::Zero;
use alga::general::{DynamicModule, AbstractRingCommutative};
use alga::general::{Module, Ring, RingCommutative};
use nalgebra::{Scalar, RealField, DimName, U1, U6};
use nalgebra::base::storage::Storage;
use super::ode::{ODESolver, ODEState, ButcherTableu, ButcherTableuSlices};
use std::marker::PhantomData;
use crate::{ODESolverBase, check_step, ODEData, ODEError, ODEStep};

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
fn rk_step<Fun, S, T, V, D, S1, S2>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, x_err: Option<&mut V>,
    dt: T, tabl: &ButcherTableu<T, D, S1, S2>,  K: &mut Vec<V>, _phantom: PhantomData<S>)
        -> Result<(), ODEError>
    where D: DimName, T: Scalar+Ring+Copy,
          //V: Clone,
          //V: DynamicModule,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          S1: Storage<T, D, D>, S2: Storage<T, D, U1>,
          S: Ring + From<T> + Copy,
          for <'b> V: AddAssign<&'b V>,
          V: Clone + MulAssign<S>
    {

    let _dt = S::from(dt);
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
    for (i, ac) in tabl.ac_iter()
        .enumerate().skip(1){
        let ti: T = t + (*ac.get(i).unwrap() ) * dt;
        // Calculate x for the current stage and store in xf
        *xf *= _zero.clone();
        for j in 0..i{
            //*k *= _zero;
            //*k += k_stages.get(j).unwrap();
            k.clone_from(k_stages.get(j).unwrap());
            *k *= S::from(*ac.index((0, j)));
            *xf += &*k;
        }
        *xf *= _dt.clone();
        *xf += x0;
        let ki = k_stages.get_mut(i).unwrap();
        //Evaluate dx for this stage and store in ki
        f(ti, xf, ki);
    };

    //Finally calculate xf
    *xf *= _zero.clone();
    for (b, ki) in tabl.b_iter()
        .zip(k_stages.iter()){
        //*k *= _zero;
        //*k += &*ki;
        k.clone_from(ki);
        *k *= S::from(*b.get(0).unwrap());
        *xf += &*k;
    }
    *xf *= _dt.clone();
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
                        *k *=  S::from(*b.get(0).unwrap());
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
            Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
            V: DynamicModule,
            V::Ring : From<T>+Copy,
            for <'b> V: AddAssign<&'b V>
{
    f: Fun,
    dat: ODEData<T, V>,
    x_err: V,
    h: T,
    tabl: ButcherTableuSlices<'a, T, U6>,
    K: Vec<V>
}

impl<'a,V,Fun> RK45Solver<'a,V,Fun,f64>
    where
        Fun: FnMut(f64, &V, &mut V) -> Result<(),()>,
        V: DynamicModule,
        V::Ring : From<f64>+Copy,
        for <'b> V: AddAssign<&'b V>{

    pub fn new(f: Fun, t0: f64, tf: f64, x0: V, h: f64 ) -> Self{

        let x_err = x0.clone();
        let tabl = ButcherTableuSlices::from_slices(&rk45_ac, &rk45_b, Some(&rk45_berr), U6);
        let mut K: Vec<V> = Vec::new();
        K.resize(7, x0.clone());
        let dat = ODEData::new(t0, tf, x0);
        RK45Solver{f, dat, x_err, h, tabl, K}
    }
}

impl<'a,V,Fun,T> ODESolverBase for RK45Solver<'a,V,Fun,T>
    where T:Scalar+RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          V: DynamicModule,
          V::Ring : From<T> + Copy,
          for <'b> V: AddAssign<&'b V>{
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
                          PhantomData::<V::Ring>);
        res
    }

}
impl<'a,V,Fun,T> ODESolver for RK45Solver<'a,V,Fun,T>
    where T:Scalar+RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          V: DynamicModule,
          V::Ring : From<T> + Copy,
          for <'b> V: AddAssign<&'b V>{

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
        let mut solver = RK45Solver::new(g, 0., 2., x0.clone(), 0.0001);
        while let ODEState::Ok = solver.step() {

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
        let mut solver = RK45Solver::new(g, 0., 2., x0.clone(), 0.0001);
        while let ODEState::Ok = solver.step() {

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
        let mut solver = RK45Solver::new(g, 0., 2., x0, 0.0001);
        while let ODEState::Ok = solver.step() {

        }
        let (tf, xf) = solver.current();
        println!("Initial Coditions: t0 = {},\n x0 = {}", 0.0, x0);
        println!("Final tf = {}\n xf={}", tf, xf);
        //println!("RK45 Butche Tableu Data: {}", rk45_tableu.ac);

    }

}