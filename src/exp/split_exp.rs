use std::ops::{AddAssign, MulAssign};
use alga::general::{Module, Ring, DynamicModule, SupersetOf, RealField};
use num_traits::Num;
use std::mem::swap;
use crate::base::{ODESolver, ODEState};
use crate::base::check_step;
use alga::morphisms::{ModuleHomomorphism, MapTo, MapRef};
use crate::{ODEData, ODESolverBase, ODEError, ODEStep};
use std::marker::PhantomData;

/// Trait to define an exponential split for operator splitting solvers
/// The linear operators must have linear combinations defined
pub trait ExponentialSplit<T, S, V>
where T: Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone
{
    type L: Clone;  //+ MulAssign<S>;
    type U: Sized;

    /// Returns the exponential of the linear operator
    fn exp(&mut self, l: &Self::L) -> Self::U;
    /// Applies the exponential on a vector x
    fn map_exp(&mut self, u: &Self::U, x: & V) -> V;
}

pub trait Commutator<T, S, V> : ExponentialSplit<T, S, V>
where T: Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone
{
    /// Compute the commutator of the two linear operators
    fn commutator(&self, la: &Self::L, lb: &Self::L) -> Self::L;
}


/// Defines an exponential split exp(A+B), where A and B are known to be
/// commutative operator, and performs an exponential action exp(A) exp(B)
pub struct CommutativeExpSplit<T, S, V, SpA, SpB>
where   T: Ring + Copy + SupersetOf<f64>,
        S: Ring + Copy + From<T>,
        V: Clone,
        SpA: ExponentialSplit<T, S, V>,
        SpB: ExponentialSplit<T, S, V>
{
    sp_a: SpA,
    sp_b: SpB,
    _phantom: PhantomData<(T, S, V)>
}

#[derive(Clone)]
pub struct CommutativeExpL<A, B, S>
where A: Clone, B: Clone, S: Clone
{
    pub a: A,
    pub b: B,
    _phantom: PhantomData<S>
}

impl<A, B, S> CommutativeExpL<A, B, S>
where A: Clone, B: Clone, S: Clone
{
    pub fn new(a: A, b: B) -> Self{
        Self{a, b, _phantom: PhantomData}
    }
}

impl<A, B, S> MulAssign<S> for CommutativeExpL<A, B, S>
where A: Clone + MulAssign<S>,
      B: Clone + MulAssign<S>,
      S: Clone{
    fn mul_assign(&mut self, s: S){
        self.a *= s.clone();
        self.b *= s;
    }
}

impl<T, S, V, SpA, SpB> ExponentialSplit<T, S, V>
for CommutativeExpSplit<T, S, V, SpA, SpB>
where T: Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone,
      SpA: ExponentialSplit<T, S, V>,
      SpB: ExponentialSplit<T, S, V>{
    type L = CommutativeExpL<SpA::L, SpB::L, S>;
    type U = (SpA::U, SpB::U);

    fn exp(&mut self, l: &Self::L) -> Self::U{
        let ua = self.sp_a.exp(&l.a);
        let ub = self.sp_b.exp(&l.b);
        (ua, ub)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V{
        self.sp_b.map_exp(&u.1, &self.sp_a.map_exp(&u.0, x))
    }

}


pub fn linear_operator_split_exp_step<SpA, SpB, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    KV: &mut Vec<V>, sp_a : &mut SpA, sp_b: &mut SpB) -> Result<(), ODEError>
where   Fun: FnMut(T) -> (SpA::L, SpB::L),
        SpA :ExponentialSplit<T, S, V>,
        SpB :ExponentialSplit<T, S, V>,
        SpA::L : MulAssign<S>, SpB::L : MulAssign<S>,
        T: Ring + Copy + SupersetOf<f64>,
        S: Ring + Copy + From<T>,
        V: Clone
{
    let k_len = KV.len();
    let s = k_len - 1;
    if s < 2 {
        panic!("linear_split_exp_step 2 stages is required")
    }
    let mut KA :Vec<SpA::L> = Vec::new();
    let mut KB :Vec<SpB::L> = Vec::new();

    let dt0 = S::from(dt.clone() * T::from_subset(&0.5));
    let dt1 = S::from(dt);
    let (la, lb) : (SpA::L, SpB::L) = f(t);
    KA.push(la.clone()); KA.push(la);
    KB.push(lb);
    KA[0] *= dt0; KA[1] *= dt1.clone();
    KB[0] *= dt1;

    let UA0 = sp_a.exp(&KA[0]);
    let UB0 = sp_b.exp(&KB[0]);

    let (kv_init, kv_rest) = KV.split_at_mut(s);
    let kvf = &mut kv_rest[0];

    *kvf = sp_a.map_exp(&UA0, x0);
    kv_init[0] = sp_b.map_exp(&UB0, &*kvf);
    *xf = sp_a.map_exp(&UA0, &kv_init[0]);

    Ok(())
}

pub struct ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where
    Fun: FnMut(T) -> (SpA::L, SpB::L),
    SpA :ExponentialSplit<T, S, V>,
    SpB :ExponentialSplit<T, S, V>,
    SpA::L : MulAssign<S>, SpB::L : MulAssign<S>,
    T: Ring + Copy + SupersetOf<f64>,
    S: Ring + Copy + From<T>,
    V: Clone
{
    f: Fun,
    sp_a: SpA,
    sp_b: SpB,
    dat: ODEData<T, V>,
    h: T,
    K: Vec<V>,
    _phantom: PhantomData<S>
}
impl<SpA, SpB, Fun, S, V, T> ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
      SpA::L : MulAssign<S>, SpB::L : MulAssign<S>,
      T: RealField,
      S: Ring + Copy + From<T>,
      V: Clone
{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T, sp_a: SpA, sp_b: SpB) -> Self{
        let mut K: Vec<V> = Vec::new();
        K.resize(3, x0.clone());
        let dat = ODEData::new(t0, tf, x0);

        Self{f, sp_a, sp_b, dat, h, K, _phantom: PhantomData}
    }
}

impl<SpA, SpB, Fun, S, V, T> ODESolverBase
for ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
      SpA::L : MulAssign<S>, SpB::L : MulAssign<S>,
      T: RealField,
      S: Ring + Copy + From<T>,
      V: Clone
{
    type TField = T;
    type RangeType = V;

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
        dat.next_dt = dt;

        linear_operator_split_exp_step(&mut self.f, dat.t,  & dat.x, &mut dat.next_x,
        dat.next_dt.clone(), &mut self.K, &mut self.sp_a, &mut self.sp_b)
    }

}

impl<SpA, SpB, Fun, S, V, T> ODESolver
for ExpSplitSolver<SpA, SpB, Fun, S, V, T>
    where Fun: FnMut(T) -> (SpA::L, SpB::L),
          SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
          SpA::L : MulAssign<S>, SpB::L : MulAssign<S>,
          T: RealField,
          S: Ring + Copy + From<T>,
          V: Clone
{

}