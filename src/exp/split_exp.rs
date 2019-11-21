use std::ops::{MulAssign, AddAssign};
use alga::general::{Ring, SupersetOf, RealField, ComplexField, SubsetOf};
use crate::exp::{ExponentialSplit, NormedExponentialSplit, Commutator};
use crate::exp::cfm::cfm_exp;
use crate::base::{ODESolver};
use crate::{ODEData, ODESolverBase, ODEError, ODEStep};
use crate::base::LinearCombination;
use std::marker::PhantomData;
use nalgebra::{Scalar};
use ndarray::{ ArrayView1, ArrayView2};
use itertools::Itertools;
use num_complex::Complex;



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

impl<T, S, V, SpA, SpB> CommutativeExpSplit<T, S, V, SpA, SpB>
where   T: Ring + Copy + SupersetOf<f64>,
        S: Ring + Copy + From<T>,
        V: Clone,
        SpA: ExponentialSplit<T, S, V>,
        SpB: ExponentialSplit<T, S, V>
{
    pub fn new(sp_a: SpA, sp_b: SpB) -> Self{
        Self{sp_a, sp_b, _phantom: PhantomData}
    }
}

#[derive(Clone)]
pub struct DirectSumL<A, B, S>
where A: Clone, B: Clone, S: Clone
{
    pub a: A,
    pub b: B,
    _phantom: PhantomData<S>
}

impl<A, B, S> DirectSumL<A, B, S>
where A: Clone, B: Clone, S: Clone
{
    pub fn new(a: A, b: B) -> Self{
        Self{a, b, _phantom: PhantomData}
    }
}

impl<A, B, S: Clone> LinearCombination<S> for DirectSumL<A, B, S>
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

    fn delta(&mut self, y: &Self) {
        self.a.delta(&y.a);
        self.b.delta(&y.b);
    }
}

impl<A, B, S> MulAssign<S> for DirectSumL<A, B, S>
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
    type L = DirectSumL<SpA::L, SpB::L, S>;
    type U = (SpA::U, SpB::U);

    fn lin_zero(&self) -> Self::L{
        DirectSumL::new(self.sp_a.lin_zero(), self.sp_b.lin_zero())
    }

    fn exp(&mut self, l: Self::L) -> Self::U{
        let ua = self.sp_a.exp(l.a);
        let ub = self.sp_b.exp(l.b);
        (ua, ub)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V{
        self.sp_b.map_exp(&u.1, &self.sp_a.map_exp(&u.0, x))
    }

    fn multi_exp(&mut self, l: Self::L, k_arr: &[S]) -> Vec<Self::U>{

        let ua_vec = self.sp_a.multi_exp(l.a, k_arr);
        let ub_vec = self.sp_b.multi_exp(l.b, k_arr);
        let vec = ua_vec.into_iter().zip(ub_vec.into_iter()).collect_vec();
        vec
    }

}

impl<T, S, V, SpA, SpB> NormedExponentialSplit<T, S, V>
for CommutativeExpSplit<T, S, V, SpA, SpB>
    where T: Ring + Copy + SupersetOf<f64>,
          S: Ring + Copy + From<T>,
          V: Clone,
          SpA: NormedExponentialSplit<T, S, V>,
          SpB: ExponentialSplit<T, S, V>{
    fn norm(&self, x: &V) -> T {
        self.sp_a.norm(x)
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
        DirectSumL::new(self.sp_a.commutator(&l1.a, &l2.a),
                        self.sp_b.commutator(&l1.b, &l2.b))
    }
}

#[derive(Clone)]
pub struct StrangSplit<T, S, V, SpA, SpB>
where   T: Ring + Copy + SupersetOf<f64>,
        S: Ring + Copy + From<T>,
        V: Clone,
        SpA: ExponentialSplit<T, S, V>,
        SpB: ExponentialSplit<T, S, V>{
    sp_a: SpA,
    sp_b: SpB,
    _phantom: PhantomData<(T, S, V)>
}

impl<T, S, V, SpA, SpB> StrangSplit<T, S, V, SpA, SpB>
    where   T: Ring + Copy + SupersetOf<f64>,
            S: Ring + Copy + From<T>,
            V: Clone,
            SpA: ExponentialSplit<T, S, V>,
            SpB: ExponentialSplit<T, S, V>
{
    pub fn new(sp_a: SpA, sp_b: SpB) -> Self{
        Self{sp_a, sp_b, _phantom: PhantomData}
    }
}

impl<T, S, V, SpA, SpB> ExponentialSplit<T, S, V>
for StrangSplit<T, S, V, SpA, SpB>
    where T: Ring + Copy + SupersetOf<f64>,
          S: Ring + Copy + From<T>,
          V: Clone,
          SpA: ExponentialSplit<T, S, V>,
          SpB: ExponentialSplit<T, S, V>,
          SpA::L : Clone + LinearCombination<S>,
          SpB::L : Clone + LinearCombination<S>,
{
    type L = DirectSumL<SpA::L, SpB::L, S>;
    type U = (SpA::U, SpB::U);

    fn lin_zero(&self) -> Self::L{
        DirectSumL::new(self.sp_a.lin_zero(), self.sp_b.lin_zero())
    }

    fn exp(&mut self, l: Self::L) -> Self::U{
        let la = l.a;
        let mut lb = l.b;
        lb.scale(S::from(T::from_subset(&0.5)));

        let ua = self.sp_a.exp(la);
        let ub = self.sp_b.exp(lb);
        (ua, ub)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V{
        let y = self.sp_a.map_exp(&u.0, &self.sp_b.map_exp(&u.1, x));
        self.sp_b.map_exp(&u.1, &y)
    }

    fn multi_exp(&mut self, l: Self::L, k_arr: &[S]) -> Vec<Self::U>{
        let la = l.a;
        let mut lb = l.b;
        lb.scale(S::from(T::from_subset(&0.5)));

        let ua_vec = self.sp_a.multi_exp(la, k_arr );
        let ub_vec = self.sp_b.multi_exp(lb, k_arr);
        let vec = ua_vec.into_iter().zip(ub_vec.into_iter()).collect_vec();
        vec
    }

}


/// Implements an exponential consisting of split SpA and SpB
/// over complex scalars
#[derive(Clone)]
pub struct SemiComplexO4ExpSplit<T, S, V, SpA, SpB>
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

/// Implements an exponential consisting of split SpA and SpB
/// over complex scalars
#[derive(Clone)]
pub struct TripleJumpExpSplit<T, S, V, SpA, SpB>
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

/// Implements an exponential consisting of split SpA and SpB
/// over complex scalars
#[derive(Clone)]
pub struct ExpSplit<T, S, V, SpA, SpB>
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

impl<T, S, V, SpA, SpB> SemiComplexO4ExpSplit<T, S, V, SpA, SpB>
    where   T: Ring + Copy + SupersetOf<f64>,
            S: Ring + Copy + From<T>,
            V: Clone,
            SpA: ExponentialSplit<T, S, V>,
            SpB: ExponentialSplit<T, S, V>
{
    pub fn new(sp_a: SpA, sp_b: SpB) -> Self{
        Self{sp_a, sp_b, _phantom: PhantomData}
    }
}


impl<T, V, SpA, SpB> ExponentialSplit<T, Complex<T>, V>
for SemiComplexO4ExpSplit<T, Complex<T>, V, SpA, SpB>
    where T: RealField,
          V: Clone,
          SpA: ExponentialSplit<T, Complex<T>, V>,
          SpA::L : Clone + LinearCombination<Complex<T>>,
          SpB::L : Clone + LinearCombination<Complex<T>>,
          SpB: ExponentialSplit<T, Complex<T>, V>
{
    type L = DirectSumL<SpA::L, SpB::L, Complex<T>>;
    type U = (SpA::U, Vec<SpB::U>);

    fn lin_zero(&self) -> Self::L{
        DirectSumL::new(self.sp_a.lin_zero(), self.sp_b.lin_zero())
    }

    fn exp(&mut self, l: Self::L) -> Self::U{
        use crate::dat::split_complex::SEMI_COMPLEX_O4_B as b_arr;

        let mut la1 = l.a;
        la1.scale(Complex::from(T::from_subset(&0.25)));
        let ua = self.sp_a.exp(la1);
        let lb1 = l.b;
        let k_arr = b_arr.iter()
            .map(|c| c.to_superset()).collect_vec();
        let ub_arr = self.sp_b.multi_exp(lb1, &k_arr);

        (ua, ub_arr)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V{
        let y1 = self.sp_a.map_exp(
            &u.0,&self.sp_b.map_exp(&u.1[0], x));
        let y2 = self.sp_a.map_exp(
            &u.0,&self.sp_b.map_exp(&u.1[1], &y1));
        drop(y1);
        let y3 = self.sp_a.map_exp(
            &u.0,&self.sp_b.map_exp(&u.1[2], &y2));
        drop(y2);
        let y4 = self.sp_a.map_exp(
            &u.0,&self.sp_b.map_exp(&u.1[1], &y3));
        drop(y3);
        self.sp_b.map_exp(&u.1[0], &y4)
    }
}

impl<T, V, SpA, SpB> NormedExponentialSplit<T, Complex<T>, V>
for SemiComplexO4ExpSplit<T, Complex<T>, V, SpA, SpB>
    where T: RealField,
          V: Clone,
          SpA: NormedExponentialSplit<T, Complex<T>, V>,
          SpA::L : Clone + LinearCombination<Complex<T>>,
          SpB::L : Clone + LinearCombination<Complex<T>>,
          SpB: ExponentialSplit<T, Complex<T>, V>{
    fn norm(&self, x: &V) -> T {
        self.sp_a.norm(x)
    }
}

impl<T, S, V, SpA, SpB> TripleJumpExpSplit<T, S, V, SpA, SpB>
    where   T: Ring + Copy + SupersetOf<f64>,
            S: Ring + Copy + From<T>,
            V: Clone,
            SpA: ExponentialSplit<T, S, V>,
            SpB: ExponentialSplit<T, S, V>
{
    pub fn new(sp_a: SpA, sp_b: SpB) -> Self{
        Self{sp_a, sp_b, _phantom: PhantomData}
    }
}

impl<T, V, SpA, SpB> ExponentialSplit<T, Complex<T>, V>
for TripleJumpExpSplit<T, Complex<T>, V, SpA, SpB>
    where T: RealField,
          V: Clone,
          SpA: ExponentialSplit<T, Complex<T>, V>,
          SpA::L : Clone + LinearCombination<Complex<T>>,
          SpB::L : Clone + LinearCombination<Complex<T>>,
          SpB: ExponentialSplit<T, Complex<T>, V>
{
    type L = DirectSumL<SpA::L, SpB::L, Complex<T>>;
    type U = (Vec<SpA::U>, Vec<SpB::U>);

    fn lin_zero(&self) -> Self::L {
        DirectSumL::new(self.sp_a.lin_zero(), self.sp_b.lin_zero())
    }

    fn exp(&mut self, l: Self::L) -> Self::U {
        use crate::dat::split_complex::{TJ_O4_A, TJ_O4_B};
        let ka_arr = TJ_O4_A.iter()
            .map(|c| c.to_superset()).collect_vec();
        let kb_arr = TJ_O4_B.iter()
            .map(|c| c.to_superset()).collect_vec();

        let ua_arr = self.sp_a.multi_exp(l.a, &ka_arr);
        let ub_arr = self.sp_b.multi_exp(l.b, &kb_arr);

        (ua_arr, ub_arr)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V {
        let y0 = self.sp_a.map_exp(&u.0[0],&self.sp_b.map_exp(&u.1[0], x));
        let y1 = self.sp_a.map_exp(&u.0[1], &self.sp_b.map_exp(&u.1[1], &y0));
        let y2 = self.sp_a.map_exp(&u.0[0], &self.sp_b.map_exp(&u.1[1], &y1));
        self.sp_b.map_exp(&u.1[0], &y2)
    }
}


pub struct RKNR4ExpSplit<T, S, V, SpA, SpB>
where  T: RealField,
       S: Ring + Copy + From<T>,
       V: Clone,
       SpA: ExponentialSplit<T, S, V>,
       SpB: ExponentialSplit<T, S, V>
{
    sp_a: SpA,
    sp_b: SpB,
    ka_arr: Vec<S>,
    kb_arr: Vec<S>,
    _phantom: PhantomData<(T, S, V)>
}

impl<T, S, V, SpA, SpB>
RKNR4ExpSplit<T, S, V, SpA, SpB>
where  T: RealField,
       S: Ring + Copy + From<T>,
       V: Clone,
       SpA: ExponentialSplit<T, S, V>,
       SpB: ExponentialSplit<T, S, V>
{
    pub fn new(sp_a: SpA, sp_b: SpB) -> Self{
        use crate::dat::split::{RKN_O4_A, RKN_O4_B};
        let ka_arr = RKN_O4_A.iter()
            .map(|c| S::from(T::from_subset(c))).collect_vec();
        let kb_arr = RKN_O4_B.iter()
            .map(|c| S::from(T::from_subset(c))).collect_vec();

        Self{sp_a, sp_b, ka_arr, kb_arr, _phantom: PhantomData}
    }
}

impl<T, S, V, SpA, SpB> ExponentialSplit<T, S, V>
for RKNR4ExpSplit<T, S, V, SpA, SpB>
    where T: RealField,
          S: Ring + Copy + From<T>,
          V: Clone,
          SpA: ExponentialSplit<T, S, V>,
          SpA::L : Clone + LinearCombination<S>,
          SpB::L : Clone + LinearCombination<S>,
          SpB: ExponentialSplit<T, S, V>
{
    type L = DirectSumL<SpA::L, SpB::L, S>;
    type U = (Vec<SpA::U>, Vec<SpB::U>);

    fn lin_zero(&self) -> Self::L {
        DirectSumL::new(self.sp_a.lin_zero(), self.sp_b.lin_zero())
    }

    fn exp(&mut self, l: Self::L) -> Self::U {
        let ua_arr = self.sp_a.multi_exp(l.a, &self.ka_arr);
        let ub_arr = self.sp_b.multi_exp(l.b, &self.kb_arr);

        (ua_arr, ub_arr)
    }

    fn map_exp(&mut self, u: &Self::U, x: &V) -> V {
        let y0 = self.sp_a.map_exp(&u.0[0],&self.sp_b.map_exp(&u.1[0], x));
        let y1 = self.sp_a.map_exp(&u.0[1], &self.sp_b.map_exp(&u.1[1], &y0)); drop(y0);
        let y2 = self.sp_a.map_exp(&u.0[2], &self.sp_b.map_exp(&u.1[2], &y1)); drop(y1);
        let y3 = self.sp_a.map_exp(&u.0[2], &self.sp_b.map_exp(&u.1[3], &y2)); drop(y2);
        let y4 = self.sp_a.map_exp(&u.0[1], &self.sp_b.map_exp(&u.1[2], &y3)); drop(y3);
        let y5 = self.sp_a.map_exp(&u.0[0], &self.sp_b.map_exp(&u.1[1], &y4)); drop(y4);

        self.sp_b.map_exp(&u.1[0], &y5)
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

    let UA0 = sp_a.exp(KA[0].clone());
    let UB0 = sp_b.exp(KB[0].clone());

    let (kv_init, kv_rest) = KV.split_at_mut(s);
    let kvf = &mut kv_rest[0];

    *kvf = sp_a.map_exp(&UA0, x0);
    kv_init[0] = sp_b.map_exp(&UB0, &*kvf);
    *xf = sp_a.map_exp(&UA0, &kv_init[0]);

    Ok(())
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
        cfm_exp(&KV[i], &mut tail[0], dt, &vb, &mut KB[0],
                 sigma.slice(s![i,..]), sp_b );
        cfm_exp(&tail[0], &mut KV[i+1], dt, &va, &mut KA[0],
                 rho.slice(s![i,..]), sp_a);
    }
    cfm_exp(&KV[s], xf, dt, &vb, &mut KB[0],
             sigma.slice(s![s,..]), sp_b );
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
        let dat = ODEData::new(t0, tf, x0, h);

        Self{f, sp_a, sp_b, dat, K, _phantom: PhantomData}
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

//    fn step_size(&self) -> ODEStep<T>{
//        self.dat.step_size_of(self.h.clone())
//    }

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

