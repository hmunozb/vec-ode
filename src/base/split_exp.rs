use std::ops::{AddAssign, MulAssign};
use alga::general::{Module, Ring, DynamicModule, SupersetOf, RealField};
use num_traits::Num;
use std::mem::swap;
use crate::base::ode::{ODESolver, ODEState};
use crate::base::ode::check_step;
use alga::morphisms::{ModuleHomomorphism, MapTo, MapRef};
use crate::{ODEData, ODESolverBase};
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

pub trait ExpSplitPair{
    type T: Ring + Copy + SupersetOf<f64>;
    type S: Ring + Copy + From<Self::T>;
    type V: Clone;
    type A: ExponentialSplit<Self::T, Self::S, Self::V>;
    type B: ExponentialSplit<Self::T, Self::S, Self::V>;
}

pub trait OperatorSplitting{
    type T: Ring + Copy + SupersetOf<f64>;
    type S: Ring + Copy + From<Self::T>;
    type V: Clone;
    type LA: Clone + MulAssign<Self::S>;
    type LB: Clone + MulAssign<Self::S>;
    type UA: Sized;
    type UB: Sized;

    fn exp_a(la: &Self::LA, ua: &mut Self::UA);
    fn exp_b(lb: &Self::LB, ub: &mut Self::UB);
    fn map_exp_a(ua: &Self::UA, x: & Self::V, y: &mut Self::V);
    fn map_exp_b(ub: &Self::UB, x: & Self::V, y: &mut Self::V);
}

pub fn linear_operator_split_exp_step<SpA, SpB, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    KV: &mut Vec<V>, _phantom: PhantomData<(SpA, SpB)>) -> Result<(),()>
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
    dat: ODEData<Fun, T, V>,
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
        K.resize(2, x0.clone());
        let dat = ODEData::new(f, t0, tf, x0);

        Self{dat, h, K, _phantom: PhantomData}
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

    fn current(&self) -> (T, &V){
        (self.dat.t.clone(), &self.dat.x)
    }
    fn step_size(&self) -> Option<T>{
        let dat = &self.dat;
        check_step(dat.t.clone(), dat.tf.clone(), self.h.clone())
    }
    fn try_step(&mut self, dt: T) -> Result<(), ()>{
        let dat = &mut self.dat;
        dat.next_dt = dt;

        linear_operator_split_exp_step(&mut dat.f, dat.t,  & dat.x0, &mut dat.next_x,
        dat.next_dt.clone(), &mut self.K, PhantomData::<(SpA,SpB)>)
    }
    fn accept_step(&mut self){
        self.dat.x.clone_from(&self.dat.next_x);
        self.dat.t += self.dat.next_dt.clone();
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
//
//fn linear_split_exp_step<Fun, FA, FB, LA, LB, UA, UB, S, T, V>(
//    f: &mut Fun, t: T, exp_a: &mut FA, exp_b: &mut FB,
//    x0: &V, xf: &mut V,  xerr: Option<&mut V>, dt: T,
//    KV: &mut Vec<V>,
//    //KA: &mut Vec<LA>, KB: &mut Vec<LB>,
//    KUA: &mut Vec<UA>, KUB: &mut Vec<UB>) -> Result<(),()>
//    where
//        Fun: FnMut(T) -> (LA, LB),
//        FA: FnMut(&LA) -> UA,  FB: FnMut(&LB) -> UB,
//        T: Ring+Copy+SupersetOf<f64>,
//        V: Clone,
//        LA: Clone + MulAssign<S>, LB: Clone + MulAssign<S>,
//        UA: MapRef<V, V>, UB: MapRef<V,V>,
//        V: MulAssign<S>,
//        S: From<T>+Copy
//        //V::Ring : From<T>+Copy
//{
//    let k_len = KV.len();
//    let s = k_len - 1;
//    if s < 2 {
//        panic!("linear_split_exp_step 2 stages is required")
//    }
//    let mut KA :Vec<LA> = Vec::new();
//    let mut KB :Vec<LA> = Vec::new();
//
//    let dt0 = S::from(dt / T::from_subset(&2.0));
//    let dt1 = S::from(dt);
//    let (la, lb) : (LA, LB) = f(t);
//    KA.push(la.clone()); KA.push(la);
//    KB.push(lb);
//    KA[0] *= dt0; KA[1] *= dt1;
//    KB[0] *= dt1;
//    KUA[0] = exp_a(&KA[0]);
//    KUB[0] = exp_b(&KB[0]);
//
//    let (kv_init, kv_rest) = KV.split_at_mut(s);
//    let kvf = &mut kv_rest[0];
//    *kvf = KUA[0].map_ref(x0);
//    kv_init[1] = KUB[0].map_ref(&*kvf);
//    KUA[1] = exp_a(&KA[1]);
//
//    Ok(())
//}
//
//fn linear_split_exp_step_w_map_to<Fun, FA, FB, LA, LB, UA, UB, T, V>(
//    f: &mut Fun, t: T, exp_a: &mut FA, exp_b: &mut FB,
//    x0: &V, xf: &mut V,  xerr: Option<&mut V>, dt: T, KV: &mut Vec<V>,
//    KA: &mut Vec<LA>, KB: &mut Vec<LB>,
//    KUA: &mut Vec<UA>, KUB: &mut Vec<UB>) -> Result<(),()>
//where
//    Fun: FnMut(T, &mut LA, &mut LB) -> Result<(),()>,
//    FA: FnMut(&LA, T, &mut UA) -> Result<(),()>,  FB: FnMut(&LB, T, &mut UB) -> Result<(),()>,
//    T: Ring+Copy+SupersetOf<f64>,
//    V: DynamicModule,
//    LA: DynamicModule, LB: DynamicModule,
//    UA: ModuleHomomorphism<V, V> + MapTo<V,V>, UB: ModuleHomomorphism<V, V> + MapTo<V,V>,
//    V::Ring : From<T>+Copy,
//    for <'b> V: AddAssign<&'b V>
//{
//    let k_len = KV.len();
//    let s = k_len - 1;
//    if s < 2 {
//        panic!("linear_split_exp_step 2 stages is required")
//    }
//
//    let dt0 = V::Ring::from(dt / T::from_subset(&2.0));
//    let dt1 = V::Ring::from(dt);
//
//    f(t, &mut KA[0], &mut KB[0])?;
//    KA[1].clone_from(&KA[0]);
//    KA[0] *= dt0;
//    KB[0] *= dt1;
//    KA[1] *= dt1;
//
//    let (kv_init, kv_rest) = KV.split_at_mut(s);
//    let kvf = &mut kv_rest[0];
//    // 2nd order Split Exp
//    exp_a(&KA[0], x0, kvf)?;
//    swap(kvf, &mut kv_init[0]);
//    exp_b(&KB[0], &kv_rest[0], xf)?;
//
//    // 1st order for error estimate
//    match xerr{
//        None => {},
//        Some(xerr_v) =>{
//            exp_a(&KA[1], x0, kvf)?;
//            exp_b(&KB[0], &*kvf, xerr)?;
//        }
//    }
//
//    return Ok(());
//}
//
//pub struct SplitExpLinSolver<V, Fun, FA, FB, LA, LB, T=f64>
//where
//    Fun: FnMut(T, &mut LA, &mut LB) -> Result<(),()>,
//    FA: FnMut(&LA, &V, &mut V) -> Result<(),()>,
//    FB: FnMut(&LB, &V, &mut V) -> Result<(),()>,
//    T: Ring+Copy+SupersetOf<f64>,
//    V: DynamicModule,
//    LA: DynamicModule,
//    LB: DynamicModule,
//    V::Ring : From<T>+Copy,
//    for <'b> V: AddAssign<&'b V>
//{
//    f: Fun,
//    exp_a: FA,
//    exp_b: FB,
//    t0: T,
//    tf: T,
//    x0: V,
//
//    t: T,
//    x: V,
//    next_x: V,
//    x_err: V,
//    K: Vec<V>,
//    KA: Vec<LA>,
//    KB: Vec<LB>,
//    h: T
//}
//impl<V, Fun, FA, FB, LA, LB, T> SplitExpLinSolver<V, Fun, FA, FB, LA, LB, T>
//where
//    Fun: FnMut(T, &mut LA, &mut LB) -> Result<(),()>,
//    FA: FnMut(&LA, &V, &mut V) -> Result<(),()>,
//    FB: FnMut(&LB, &V, &mut V) -> Result<(),()>,
//    T: Ring+Copy+SupersetOf<f64>,
//    V: DynamicModule,
//    LA: DynamicModule,
//    LB: DynamicModule,
//    V::Ring : From<T>+Copy,
//    for <'b> V: AddAssign<&'b V>{
//    pub fn new(f: Fun, exp_a: FA, exp_b: FB,
//               t0: T, tf: T, x0: V, h: T, a0: LA, b0: LB){
//        let x = x0.clone();
//        let next_x = x0.clone();
//        let x_err = x0.clone();
//        let t = t0.clone();
//
//        let mut K: Vec<V> = Vec::new();
//        K.resize(2, x0.clone());
//        let mut KA: Vec<LA> = Vec::new();
//        KA.resize_with(2, a0);
//        let mut KB: Vec<LA> = Vec::new();
//        KA.resize_with(2, b0);
//
//        Self{f, exp_a, exp_b, t0, tf, x0, t, x, next_x, x_err: x0.clone(), K, KA, KB, h}
//    }
//}
//
//impl<V, Fun, FA, FB, LA, LB, T> ODESolver
//for SplitExpLinSolver<V, Fun, FA, FB, LA, LB, T>
//    where
//        Fun: FnMut(T, &mut LA, &mut LB) -> Result<(),()>,
//        FA: FnMut(&LA, &V, &mut V) -> Result<(),()>,
//        FB: FnMut(&LB, &V, &mut V) -> Result<(),()>,
//        T: Ring+Copy+SupersetOf<f64>,
//        V: DynamicModule,
//        LA: DynamicModule,
//        LB: DynamicModule,
//        V::Ring : From<T>+Copy,
//        for <'b> V: AddAssign<&'b V>
//{
//    type TField=T;
//    type RangeType=V;
//
//    fn step(&mut self) -> ODEState{
//        let dt_opt = check_step(self.t, self.tf, self.h);
//        match dt_opt{
//            None => ODEState::Done,
//            Some(dt) => {
//                let res = linear_split_exp_step_w_map_to(
//                    &mut self.f, self.t, &mut self.exp_a, &mutself.exp_b,
//                                            &self.x, &mut self.next_x, Some(&mut self.x_err),
//                                        dt, &mut self.K, &mut self.KA, &mut self.KB);
//                match res{
//                    Ok(()) => {
//                        swap(&mut self.x, &mut self.next_x);
//                        self.t += dt;
//                        ODEState::Ok },
//                    Err(()) => ODEState::Err
//                }
//
//            }
//        }
//    }
//
//    fn current(&self) -> (T, &V) { (self.t, &self.x) }
//}
//
//#[cfg(test)]
//mod tests {
//    use super::*;
//    use nalgebra::{Vector2, DVector, DMatrix};
//    use num_complex::Complex64 as c64;
//
//    #[test]
//    fn test_split_exp(){
//        let A0 = DMatrix::<c64>::from_row_slice(2, 2
//            &[c64::from(1.0), c64::from(0.0),
//              c64::from(0.0), c64::from(1.0)   ]
//        );
//        let B0 = DMatrix::<c64>::from_row_slice(2, 2
//            &[c64::from(0.0), c64::i()*0.5,
//            -c64::i()*0.5, c64::from(1.0)   ]
//        );
//
//        let f = |t: f64, A: &mut DMatrix<c64>,
//                 B: &mut DMatrix<c64>|{
//            A.copy_from(&A0);
//            B.copy_from(&B0);
//            Ok(())
//        };
//
//    }
//}