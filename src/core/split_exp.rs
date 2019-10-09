use std::ops::{AddAssign, MulAssign};
use alga::general::{Module, Ring, DynamicModule, SupersetOf};
use num_traits::Num;
use std::mem::swap;
use crate::core::ode::{ODESolver, ODEState};
use crate::core::ode::check_step;
use alga::morphisms::{ModuleHomomorphism, MapTo, MapRef};

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

pub fn linear_operator_split_exp_step<Sp: OperatorSplitting, Fun>(
    f: &mut Fun, t: Sp::T, x0: &Sp::V, xf: &mut Sp::V, dt: Sp::T,
    KV: &mut Vec<Sp::V>,KUA: &mut Vec<Sp::UA>, KUB: &mut Vec<Sp::UB>)
where Fun: FnMut(Sp::T) -> (Sp::LA, Sp::LB)
{

    let k_len = KV.len();
    let s = k_len - 1;
    if s < 2 {
        panic!("linear_split_exp_step 2 stages is required")
    }
    let mut KA :Vec<Sp::LA> = Vec::new();
    let mut KB :Vec<Sp::LB> = Vec::new();

    let dt0 = Sp::S::from(dt.clone() * Sp::T::from_subset(&0.5));
    let dt1 = Sp::S::from(dt);
    let (la, lb) : (Sp::LA, Sp::LB) = f(t);
    KA.push(la.clone()); KA.push(la);
    KB.push(lb);
    KA[0] *= dt0; KA[1] *= dt1.clone();
    KB[0] *= dt1;

    Sp::exp_a(&KA[0],&mut KUA[0]);
    Sp::exp_b(&KB[0], &mut KUB[0]);

    let (kv_init, kv_rest) = KV.split_at_mut(s);
    let kvf = &mut kv_rest[0];

    Sp::map_exp_a(&KUA[0], x0, kvf);
    Sp::map_exp_b(&KUB[0], &*kvf, &mut kv_init[0]);
    Sp::map_exp_a(&KUA[0], &kv_init[0], xf);
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