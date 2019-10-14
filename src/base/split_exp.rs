use std::ops::{AddAssign, MulAssign};
use alga::general::{Module, Ring, DynamicModule, SupersetOf, RealField};
use num_traits::Num;
use std::mem::swap;
use crate::base::ode::{ODESolver, ODEState};
use crate::base::ode::check_step;
use alga::morphisms::{ModuleHomomorphism, MapTo, MapRef};
use crate::{ODEData, ODESolverBase, ODEError, ODEStep};
use std::marker::PhantomData;

pub trait ExponentialSplit<T, S, V>
where T:Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone
{
    type L: Clone + MulAssign<S>;
    type U: Sized;
    fn exp(l: &Self::L) -> Self::U;
    fn map_exp(u: &Self::U, x: & V) -> V;
}


pub fn linear_operator_split_exp_step<SpA, SpB, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    KV: &mut Vec<V>, _phantom: PhantomData<(SpA, SpB)>) -> Result<(), ODEError>
where   Fun: FnMut(T) -> (SpA::L, SpB::L),
        SpA :ExponentialSplit<T, S, V>,
        SpB :ExponentialSplit<T, S, V>,
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

    let UA0 = SpA::exp(&KA[0]);
    let UB0 = SpB::exp(&KB[0]);

    let (kv_init, kv_rest) = KV.split_at_mut(s);
    let kvf = &mut kv_rest[0];

    *kvf = SpA::map_exp(&UA0, x0);
    kv_init[0] = SpB::map_exp(&UB0, &*kvf);
    *xf = SpA::map_exp(&UA0, &kv_init[0]);

    Ok(())
}

pub struct ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where
    Fun: FnMut(T) -> (SpA::L, SpB::L),
    SpA :ExponentialSplit<T, S, V>,
    SpB :ExponentialSplit<T, S, V>,
    T: Ring + Copy + SupersetOf<f64>,
    S: Ring + Copy + From<T>,
    V: Clone
{
    f: Fun,
    dat: ODEData<T, V>,
    h: T,
    K: Vec<V>,
    _phantom: PhantomData<(SpA, SpB, S)>
}
impl<SpA, SpB, Fun, S, V, T> ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
      T: RealField,
      S: Ring + Copy + From<T>,
      V: Clone
{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T) -> Self{
        let mut K: Vec<V> = Vec::new();
        K.resize(3, x0.clone());
        let dat = ODEData::new(t0, tf, x0);

        Self{f, dat, h, K, _phantom: PhantomData}
    }
}

impl<SpA, SpB, Fun, S, V, T> ODESolverBase
for ExpSplitSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
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
        dat.next_dt.clone(), &mut self.K, PhantomData::<(SpA,SpB)>)
    }

}

impl<SpA, SpB, Fun, S, V, T> ODESolver
for ExpSplitSolver<SpA, SpB, Fun, S, V, T>
    where Fun: FnMut(T) -> (SpA::L, SpB::L),
          SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
          T: RealField,
          S: Ring + Copy + From<T>,
          V: Clone
{

}