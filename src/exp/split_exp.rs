use std::ops::{MulAssign, AddAssign};
use alga::general::{Ring, SupersetOf, RealField, ComplexField};
use crate::base::{ODESolver};
use crate::{ODEData, ODESolverBase, ODEError, ODEStep};
use crate::base::LinearCombination;
use std::marker::PhantomData;
use nalgebra::{Scalar};
use ndarray::{ ArrayView1, ArrayView2};
use itertools::Itertools;

/// Trait to define an exponential split for operator splitting solvers
/// The linear operators must have linear combinations defined
pub trait ExponentialSplit<T, S, V>
where T: Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone
{
    type L: Clone + LinearCombination<S>;  //+ MulAssign<S>;
    type U: Sized;

    /// Returns the exponential of the linear operator
    fn exp(&mut self, l: &Self::L) -> Self::U;
    /// Applies the exponential on a vector x
    fn map_exp(&mut self, u: &Self::U, x: & V) -> V;
}

pub trait NormedExponentialSplit<T, S, V>
    where T: Ring + Copy + SupersetOf<f64>,
          S: Ring + Copy + From<T>,
          V: Clone
{
    /// Calculates the norm of a vector x
    fn norm(&self, x: &V) -> T;
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
/// If both splits have a commutator, the commutator trait is also implemented
/// as the direct sum of each commutator.
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

impl<A, B, S: Clone> LinearCombination<S> for CommutativeExpL<A, B, S>
where   A : Clone + LinearCombination<S>,
        B : Clone + LinearCombination<S>
{
    fn scale(&mut self, k: S) {
        self.a.scale(k.clone());
        self.b.scale(k)
    }

    fn scalar_multiply_to(&self, k: S, target: &mut Self) {
        self.a.scalar_multiply_to(k.clone(), &mut target.a);
        self.b.scalar_multiply_to(k, &mut target.b);
    }

    fn add_scalar_mul(&mut self, k: S, other: &Self) {
        self.a.add_scalar_mul(k.clone(), & other.a);
        self.b.add_scalar_mul(k, &other.b);
    }

    fn add_assign_ref(&mut self, other: &Self){
        self.a.add_assign_ref(&other.a);
        self.b.add_assign_ref(&other.b);
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
      SpB: ExponentialSplit<T, S, V>
{
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

impl<T, S, V, SpA, SpB> Commutator<T, S, V>
for CommutativeExpSplit<T, S, V, SpA, SpB>
where T: Ring + Copy + SupersetOf<f64>,
      S: Ring + Copy + From<T>,
      V: Clone,
      SpA: Commutator<T, S, V>,
      SpB: Commutator<T, S, V>{

    fn commutator(&self, l1: &Self::L, l2: &Self::L) -> Self::L{
        CommutativeExpL::new(   self.sp_a.commutator(&l1.a, &l2.a),
                                self.sp_b.commutator(&l1.b, &l2.b))
    }
}

/// Evaluates one step of the midpoint method for a splitting SpA and SpB
pub fn split_exp_midpoint<SpA, SpB, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    KV: &mut Vec<V>, sp_a : &mut SpA, sp_b: &mut SpB) -> Result<(), ODEError>
where   Fun: FnMut(T) -> (SpA::L, SpB::L),
        SpA :ExponentialSplit<T, S, V>,
        SpB :ExponentialSplit<T, S, V>,
        SpA::L : LinearCombination<S>, //+ MulAssign<S>,
        SpB::L : LinearCombination<S>, //+ MulAssign<S>,
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
    KA[0].scale(dt0); KA[1].scale(dt1);
    KB[0].scale(dt1);
//    KA[0] *= dt0; KA[1] *= dt1.clone();
//    KB[0] *= dt1;

    let UA0 = sp_a.exp(&KA[0]);
    let UB0 = sp_b.exp(&KB[0]);

    let (kv_init, kv_rest) = KV.split_at_mut(s);
    let kvf = &mut kv_rest[0];

    *kvf = sp_a.map_exp(&UA0, x0);
    kv_init[0] = sp_b.map_exp(&UB0, &*kvf);
    *xf = sp_a.map_exp(&UA0, &kv_init[0]);

    Ok(())
}

///
/// Evaluates the linear combination of operators k := a.m
/// Then evaluates the exponential action x1 := exp(k) x0 with the splitting sp
///
fn split_cfm_exp<'a, Sp, T, S, V>(
    x0: &V, x1: &mut V, dt: T, m: &Vec<Sp::L>, k: &mut Sp::L, temp: &mut Sp::L,
    a: ArrayView1<'a, S>, sp: &mut Sp
)
where   Sp: ExponentialSplit<T, S, V>,
        Sp::L : LinearCombination<S>,// + MulAssign<S> + for <'b> AddAssign<&'b Sp::L>,
        T: Ring + Copy + SupersetOf<f64>,
        S: Scalar + Ring + Copy + From<T>,
        V: Clone
{
//    m[0].clone_into(k);
//    *k *= a[0].clone();
    m[0].scalar_multiply_to(a[0].clone(), k);
    for (ai, mi) in a.iter().skip(1)
            .zip(m.iter().skip(1)){
        k.add_scalar_mul(ai.clone(), mi);
//        m.clone_into(temp);
//        *temp *= *ai;
//        *k += temp;
    }
    k.scale(S::from(dt));
    //*k *= S::from(dt);
    let u = sp.exp(&*k);
    *x1 = sp.map_exp(&u, x0);

}

///
/// Evaluates one step of the BAB exponential split
pub fn split_cfm<'a, SpA, SpB, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    c: &Vec<T>,
    rho: ArrayView2<'a, S>, sigma: ArrayView2<'a, S>,
    KV: &mut Vec<V>, KA: &mut Vec<SpA::L>, KB: &mut Vec<SpB::L>,
    sp_a : &mut SpA, sp_b: &mut SpB) -> Result<(), ODEError>
where
    Fun: FnMut(&[T]) -> (Vec<SpA::L>, Vec<SpB::L>),
    SpA :ExponentialSplit<T, S, V>,
    SpB :ExponentialSplit<T, S, V>,
    SpA::L : LinearCombination<S>,//MulAssign<S> + for <'b> AddAssign<&'b SpA::L>,
    SpB::L : LinearCombination<S>, //MulAssign<S> + for <'b> AddAssign<&'b SpB::L>,
    T: Ring + Copy + SupersetOf<f64>,
    S: Scalar + Ring + Copy + From<T>,
    V: Clone
{
    let k = c.len();
    if rho.ncols() != k || sigma.ncols() != k{
        panic!("split_cfm: Incompatible array dimensions")
    };
    let s = rho.nrows();
    if sigma.nrows() != s + 1 {
        panic!("split_cfm: Incompatible array dimensions")
    }

    let (KV, tail) = KV.split_at_mut(s+1);
    let (KA, KA_tail) = KA.split_at_mut(1);
    let (KB, KB_tail) = KB.split_at_mut(1);
    let t_arr = c.iter().map(|ci| t + (*ci)*dt).collect_vec();

    let (va, vb) = (*f)(&t_arr);
    KV[0] = x0.clone();
    for i in 0..s{
        split_cfm_exp(&KV[i], &mut tail[0], dt,&vb, &mut KB[0],
                      &mut KB_tail[0],sigma.slice(s![i,..]), sp_b );
        split_cfm_exp(&tail[0], &mut KV[i+1], dt,&va, &mut KA[0],
                      &mut KA_tail[1], rho.slice(s![i,..]), sp_a);
//        let bi = vb[0].clone();
//        bi *= sigma.index((i,0));
//        for j in 1..k{
//            KB[0] = vb[j].clone();
//            KB[0] *= sigma.index((i, j));
//            bi += KB[0];
//        }
//        let ub = sp_b.exp(&bi);
//        tail[0] = sp_b.map_exp(&ub, &KV[i]);

//        let ai = va[0].clone();
//        ai *= rho.index((i, 0));
//        for j in 1..k {
//            KA[0] = va[j].clone();
//            KA[0] *= rho.index((i, j));
//            ai += KA[0];
//        }
//        let ua = sp_a.exp(&ai);
//        KV[i+1] = sp_a.map_exp(&ua, &tail[0]);

    }
    split_cfm_exp(&KV[s], xf, dt,&vb, &mut KB[0],
                  &mut KB_tail[0],sigma.slice(s![s,..]), sp_b );
    Ok(())
}

/// Solves the linear system dx/dt = ( A(t) + B(t)) x, where
/// A and B are two operator splittings defined by the ExponentialSplit trait
pub struct ExpSplitMidpointSolver<SpA, SpB, Fun, S, V, T>
where
    Fun: FnMut(T) -> (SpA::L, SpB::L),
    SpA :ExponentialSplit<T, S, V>,
    SpB :ExponentialSplit<T, S, V>,
    SpA::L : LinearCombination<S>, // MulAssign<S>,
    SpB::L : LinearCombination<S>, // MulAssign<S>,
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

impl<SpA, SpB, Fun, S, V, T> ExpSplitMidpointSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
      SpA::L : LinearCombination<S>, // MulAssign<S>,
      SpB::L : LinearCombination<S>, // MulAssign<S>,
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
for ExpSplitMidpointSolver<SpA, SpB, Fun, S, V, T>
where Fun: FnMut(T) -> (SpA::L, SpB::L),
      SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
      SpA::L : LinearCombination<S>, // MulAssign<S>,
      SpB::L : LinearCombination<S>, // MulAssign<S>,
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

        split_exp_midpoint(&mut self.f, dat.t, & dat.x, &mut dat.next_x,
                           dat.next_dt.clone(), &mut self.K, &mut self.sp_a, &mut self.sp_b)
    }

}

impl<SpA, SpB, Fun, S, V, T> ODESolver
for ExpSplitMidpointSolver<SpA, SpB, Fun, S, V, T>
    where Fun: FnMut(T) -> (SpA::L, SpB::L),
          SpA :ExponentialSplit<T, S, V>, SpB :ExponentialSplit<T, S, V>,
          SpA::L : LinearCombination<S>, // MulAssign<S>,
          SpB::L : LinearCombination<S>, // MulAssign<S>,
          T: RealField,
          S: Ring + Copy + From<T>,
          V: Clone
{

}

pub struct ExpSplitCFMSolver<SpA, SpB, Fun, S, V, T>
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