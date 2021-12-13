
use std::ops::{AddAssign, MulAssign, Mul, SubAssign};
use crate::lc::LinearCombination;
use itertools::Itertools;

//use alga::general::{RealField, ClosedAdd};
use crate::RealField;

use ndarray::{Ix1, Array1, Array2};
use std::marker::PhantomData;
use crate::{ODESolver, ODEData, ODEError, ODEAdaptiveData, AdaptiveODESolver};
use crate::base::ode::Normed;
use crate::dat::rk::{RK45_AC, RK45_B, RK45_BERR};
use ndarray::iter::Lanes;
use super::num_complex::Complex;

use itertools::zip_eq;
use num_traits::Float;



#[derive(Debug)]
pub struct ButcherTableu<T> {
    ac: Array2<T>,
    b:Array1<T>,
    b_err: Option<Array1<T>>
}

impl<T> ButcherTableu<T>
{
    pub fn from_slices<T2>(ac: &[T2], b: &[T2], b_err: Option<&[T2]>, s: usize) -> Self
    where T2: Clone + Into<T>,
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

    pub fn from_vecs(ac: Vec<T>, b: Vec<T>, b_err: Option<Vec<T>>, s: usize) -> Self
    {
        ButcherTableu{
            ac: Array2::from_shape_vec((s,s),ac).unwrap(),
            b: Array1::from_shape_vec(s,b).unwrap(),
            b_err: b_err.map(|b|Array1::from_shape_vec(s,b).unwrap())
        }
    }
}

impl<T> ButcherTableu<T>{
    pub fn num_stages(&self) -> usize{
        self.b.len()
    }

    pub fn ac_iter(&self) -> Lanes<T, Ix1> {
        self.ac.genrows()
    }

    pub fn b_iter(&self) -> ndarray::iter::Iter<T, Ix1>{
        self.b.iter()
    }

    pub fn b_sl(&self) -> &[T] {
        self.b.as_slice().unwrap()
    }

    pub fn b_err_iter(&self) -> Option<ndarray::iter::Iter<T, Ix1>>{
        self.b_err.as_ref().map(|b| b.iter())
    }

    pub fn b_err_sl(&self) -> Option<&[T]>{
        self.b_err.as_ref().map(|b| b.as_slice().unwrap())
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
fn rk_step<Fun, S, T, V, LC>(
    mut f: Fun, t: T, x0: &V, xf: &mut V, x_err: Option<&mut V>,
    dt: T, tabl: &ButcherTableu<T>,  K: &mut Vec<V>,
    _lc: &LC,
)
        -> Result<(), ODEError>
    where T: RealField + Into<S>,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          S: Copy ,
          LC: LinearCombination<S, V>
    {

    let _dt : S = dt.into();
    //Check that the number of stages is consistent
    let k_len = K.len();
    let s = k_len - 1;
    if tabl.num_stages() != s {
        panic!("rk_step: number of stages in ButcherTableu and stages does not match")
    }

    //Calculate the first stage
    f(t, x0, K.get_mut(0).unwrap()).unwrap();

    //Split the K vector into the stages vector and the entry used as an arithmetic register
    let (k_stages, k_work) = K.split_at_mut(s);
    let _k = k_work.get_mut(0).unwrap();

    //Iterate over the tableu, skipping the first trivial entry
    for (i, ac) in tabl.ac_iter().into_iter().enumerate().skip(1){
        let ti: T = t + (ac[i]) * dt;
        // Calculate x for the current stage and store in xf
        LC::linear_combination_iter(xf, k_stages.iter().take(i),
        ac.iter().take(i)).unwrap();
        LC::scale(xf, _dt);
        LC::add_assign_ref(xf, x0);
        let ki = k_stages.get_mut(i).unwrap();
        //Evaluate dx for this stage and store in ki
        f(ti, xf, ki).unwrap();
    };

    //Finally calculate xf
    LC::linear_combination(xf, &*k_stages, tabl.b_sl());
    LC::scale(xf, _dt);
    LC::add_assign_ref(xf, x0);

    //Also handle x_err if the butcher tableu contains error terms
    match tabl.b_err_iter(){
        None => {},
        Some(e) => {
            match x_err{
                None => {},
                Some(xe) =>{
                    std::mem::swap(xe, xf);
                    LC::linear_combination_iter(xf, k_stages.iter(), e)
                        .expect("linear_combination_iter error");
                    LC::scale(xf, _dt);
                    LC::add_assign_ref(xf, x0);
                    LC::delta(xe, &*xf);
                }
            }
        }
    }

    Result::Ok(())

}


pub struct RK45Solver<V,Fun,S,T=f64, LC=RK45SolverDefaultLC>
    where   T: RealField,
            Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
            S : Copy,
            V: Clone,
            LC: LinearCombination<S, V>
{
    f: Fun,
    dat: ODEData<T, V>,
    adaptive_dat: ODEAdaptiveData<T, V>,
    x_err: Option<V>,
    tabl: ButcherTableu<T>,
    K: Vec<V>,
    lc: LC,
    _phantom: PhantomData<S>
}

#[derive(Copy, Clone, Default)]
pub struct RK45SolverDefaultLC;

impl<S: Copy, V> LinearCombination<S, V> for RK45SolverDefaultLC
where V: AddAssign<V> + MulAssign<S>,
      for <'b> &'b V: Mul<S, Output=V>,
      for <'b> V: AddAssign<&'b V> + SubAssign<&'b V>
{
    fn scale(v: &mut V, k: S) {
        *v *= k;
    }

    fn scalar_multiply_to(v: &V, k: S, target: &mut V) {
        *target = v * k;
    }

    fn add_scalar_mul(v: &mut V, k: S, other: &V) {
        *v += other * k;
    }

    fn add_assign_ref(v: &mut V, other: &V) {
        *v += other;
    }

    fn delta(v: &mut V, y: &V) {
        *v -= y;
    }
}

impl<R: RealField> Normed<R, R> for RK45SolverDefaultLC
{
    fn norm(v: &R) -> R {
        v.abs()
    }
}impl<R: RealField+Float> Normed<R, Complex<R>> for RK45SolverDefaultLC
{
    fn norm(v: &Complex<R>) -> R {
        v.norm()
    }
}


pub type RK45RealSolver<V, Fun, T> = RK45Solver<V, Fun, T, T, RK45SolverDefaultLC>;
pub type RK45ComplexSolver<V, Fun, T> = RK45Solver<V, Fun, Complex<T>, T, RK45SolverDefaultLC>;

impl<'a,V,S,Fun,T,LC> RK45Solver<V,Fun,S,T,LC>
    where
        Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
        V: Clone,
        S: From<T> + Copy,
        T: RealField,
        LC: LinearCombination<S, V> + Default
{

    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T ) -> Self{
        Self::new_with_lc(f, t0, tf, x0, h, Default::default())
    }

    pub fn no_adaptive(self) -> Self{
        let mut me = self;
        me.x_err = None;
        me
    }
}

impl<'a,V,S,Fun,T,LC> RK45Solver<V,Fun,S,T,LC>
    where
        Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
        V: Clone,
        S: From<T> + Copy,
        T: RealField,
        LC: LinearCombination<S, V>
{
    pub fn new_with_lc(f: Fun, t0: T, tf: T, x0: V, h: T, lc: LC) -> Self{
        let x_err = Some(x0.clone());
        let ac = RK45_AC.iter().map(|&x| T::from_f64(x).unwrap()).collect_vec();
        let b = RK45_B.iter().map(|&x| T::from_f64(x).unwrap()).collect_vec();
        let b_err = RK45_BERR.iter().map(|&x| T::from_f64(x).unwrap()).collect_vec();

        let tabl = ButcherTableu::from_vecs(ac, b, Some(b_err), 6);
        let mut K: Vec<V> = Vec::new();
        K.resize(7, x0.clone());
        let dat = ODEData::new(t0, tf, x0.clone(), h);
        let adaptive_dat  = ODEAdaptiveData::new_with_defaults(
            x0, T::from_f64(3.0).unwrap()).with_alpha(
            T::from_f64(0.9).unwrap());
        RK45Solver{f, dat, adaptive_dat, x_err,  tabl, K, lc, _phantom: PhantomData}
    }


}

impl<V,Fun,S,T,LC> ODESolver for RK45Solver<V,Fun,S,T,LC>
    where T: RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          V: Clone,
          S: From<T> + Copy,
          LC: LinearCombination<S, V> {
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


    fn try_step(&mut self, dt: T) -> Result<(), ODEError>{
        let dat = &mut self.dat;
        let res = rk_step(&mut self.f, dat.t.clone(), &dat.x,
                          &mut dat.next_x, self.x_err.as_mut(),
                          dt, &self.tabl, &mut self.K, &self.lc);
        res
    }

}

impl<V,Fun,S,T,LC> AdaptiveODESolver<T> for RK45Solver<V, Fun, S,T,LC>
    where T: RealField,
          Fun: FnMut(T, &V, &mut V) -> Result<(),()>,
          V: Clone,
          S: From<T> + Copy,
          LC: LinearCombination<S, V> + Normed<T, V>
{
    fn ode_adapt_data(&self) -> &ODEAdaptiveData<T, Self::RangeType> {
        &self.adaptive_dat
    }

    fn ode_adapt_data_mut(&mut self) -> &mut ODEAdaptiveData<T, Self::RangeType> {
        &mut self.adaptive_dat
    }

    fn norm(&mut self) -> Self::TField {
        //let dat = self.ode_adapt_data();
        self.x_err.as_ref().map_or(T::zero(),|dx| LC::norm(dx))
    }

    fn validate_adaptive(&self) -> Result<(), ()> {
        if self.x_err.is_some() { Ok(())} else {Err(())}
    }
}

