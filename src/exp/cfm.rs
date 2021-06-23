use std::mem::swap;
use std::ops::SubAssign;

use itertools::Itertools;
use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::{AdaptiveODESolver, ODESolver};
use crate::base::{LinearCombination, ODEAdaptiveData, ODEData, ODEError,
                  ODESolverBase, ODEState, ODEStep};
use crate::dat::cfqm::{CFM_R2_J1_GL, CFM_R4_J2_GL};
use crate::dat::quad::C_GAUSS_LEGENDRE_4;
use crate::exp::ExponentialSplit;
use crate::from_f64;
use crate::RealField;

///
/// Evaluates the linear combination of operators k := a.m dt
/// Then evaluates the exponential action x1 := exp(k) x0 with the splitting sp
///
pub fn cfm_exp<'a, Sp, T, S, V>(
    x0: &V, x1: &mut V, dt: T, m: &Vec<Sp::L>, k: &mut Sp::L,
    a: ArrayView1<'a, S>, sp: &mut Sp
)
    where   Sp: ExponentialSplit<T, S, V>,
            //Sp::L : LinearCombinationSpace<S>,// + MulAssign<S> + for <'b> AddAssign<&'b Sp::L>,
            T: RealField,
            S: Copy + From<T>,
            V: Clone
{

    Sp::LC::scalar_multiply_to(&m[0], a[0].clone(), k);
    //m[0].scalar_multiply_to(a[0].clone(), k);
    for (ai, mi) in a.iter().skip(1)
        .zip(m.iter().skip(1)){
        Sp::LC::add_scalar_mul(k, ai.clone(), mi);
    }
    Sp::LC::scale(k, S::from(dt));
    let u = sp.exp(k.clone());
    *x1 = sp.map_exp(&u, x0);
}

///
pub fn cfm_general<'a, Sp, T, S, V, Fun>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    c: &Vec<T>,
    alpha: ArrayView2<'a, S>,
    KV: &mut Vec<V>, KA: &mut Vec<Sp::L>,
    sp : &mut Sp,
    x_err: Option<&mut V>,
    alph_err: Option<ArrayView2<'a, S>>,

) -> Result<(), ODEError>
    where
        Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
        Sp :ExponentialSplit<T, S, V>,
        //Sp::L : LinearCombinationSpace<S>,
        T: RealField,
        S: Copy + From<T>,
        V: Clone + for <'b> SubAssign<&'b V>
{
    let k = c.len();
    if alpha.ncols() != k {
        panic!("split_cfm: Incompatible array dimensions")
    };
    let s = alpha.nrows();

    let (KV, tail) = KV.split_at_mut(s);
    //let (KA, KA_tail) = KA.split_at_mut(1);

    let t_arr = c.iter().map(|ci| t + (*ci)*dt).collect_vec();

    let va = (*f)(&t_arr, (t, t+dt));

    cfm_exp(x0, &mut KV[0], dt, &va, &mut KA[0],
            alpha.slice(s![0,..]), sp );
    for i in 1..s{
        cfm_exp(&KV[i-1], &mut tail[0], dt, &va, &mut KA[0],
                 alpha.slice(s![i,..]), sp );
        swap(&mut tail[0], &mut KV[i]);
    }
    swap(&mut KV[s-1], xf);

    if let (Some(x_err), Some(alph_err)) = (x_err, alph_err){
        let s_err = alph_err.nrows();
        if s_err > s || alph_err.ncols() != k{
            panic!("split_cfm: Incompatible array dimensions for alph_err");
        }
        cfm_exp(x0, &mut KV[0], dt, &va, &mut KA[0],
                alph_err.slice(s![0,..]), sp );
        for i in 1..s_err{
            cfm_exp(&KV[i-1], &mut tail[0], dt, &va, &mut KA[0],
                     alph_err.slice(s![i,..]), sp );
            swap(&mut tail[0], &mut KV[i]);
        }
        swap(&mut KV[s_err-1], x_err);
        *x_err -= &*xf;
    }

    Ok(())
}

pub struct ExpCFMSolver<Sp, Fun, NormFn, S, V, T>
    where
        Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
        NormFn: FnMut(&V) -> T,
        Sp : ExponentialSplit<T, S, V>,
        T: RealField,
        S: Copy + From<T>,
        V: Clone
{
    f: Fun,
    norm: NormFn,
    sp: Sp,
    dat: ODEData<T, V>,
    adaptive_dat: ODEAdaptiveData<T, V>,
    K: Vec<V>,
    KA: Vec<Sp::L>,
    c: Vec<T>,
    alpha: Array2<S>,
    alph_err: Option<Array2<S>>
}

impl<Sp, Fun, NormFn, S, V, T> ExpCFMSolver<Sp, Fun, NormFn, S, V, T>
where
    Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
    NormFn: FnMut(&V) -> T,
    Sp : ExponentialSplit<T, S, V>,
    T: RealField ,
    S: Copy + From<T>,
    V: Clone + for <'b> SubAssign<&'b V>{
    pub fn new(f: Fun, norm: NormFn, t0: T, tf: T, x0: V, h: T, sp: Sp) -> Self{
        let mut K: Vec<V> = Vec::new();
        let mut KA: Vec<Sp::L> = Vec::new();
        K.resize(3, x0.clone());
        KA.resize_with(1, | | sp.lin_zero());
        let c = C_GAUSS_LEGENDRE_4.iter()
            .map(|&x| from_f64!(T, x))
            .collect_vec();
        let alpha = Array2::from_shape_vec((2,2),
            CFM_R4_J2_GL.iter()
                .map(|&x|S::from(from_f64!(T, x)))
                .collect_vec()
            ).unwrap();
        let alph_err = Array2::from_shape_vec((1,2),
             CFM_R2_J1_GL.iter()
                 .map(|&x|S::from(from_f64!(T, x)))
                 .collect_vec()
            ).unwrap().into();
        let dat = ODEData::new(t0, tf, x0.clone(), h);
        let adaptive_dat = ODEAdaptiveData::new_with_defaults(
            x0, from_f64!(T, 3.0)).with_alpha(from_f64!(T, 0.9));

        Self{f, norm, sp, dat, adaptive_dat, K, KA, c, alpha, alph_err}
    }

    /// Disable adaptive stage evaluation
    pub fn no_adaptive(self) -> Self{
        let mut me = self;
        me.alph_err = None;
        me
    }
}


impl<Sp, Fun, NormFn, S, V, T> ODESolverBase for ExpCFMSolver<Sp, Fun, NormFn, S, V, T>
    where       Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
                NormFn: FnMut(&V) -> T,
                Sp : ExponentialSplit<T, S, V>,
                T: RealField ,
                S: Copy + From<T>,
                V: Clone + for <'b> SubAssign<&'b V>
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
//        self.dat.step_size_of(self.dat.h)
//    }

    fn try_step(&mut self, dt: T) -> Result<(), ODEError> {
        let dat = &mut self.dat;
        dat.next_dt = dt;
        cfm_general(&mut self.f, dat.t,&dat.x, &mut dat.next_x, dt, &self.c, self.alpha.view(),
        &mut self.K, &mut self.KA, &mut self.sp, Some(&mut self.adaptive_dat.dx),
                    self.alph_err.as_ref().map(|a|a.view()))
    }

}

impl<Sp, Fun, NormFn, S, V, T> ODESolver for ExpCFMSolver<Sp, Fun, NormFn, S, V, T>
    where       Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
                NormFn: FnMut(&V) -> T,
                Sp : ExponentialSplit<T, S, V>,
                T: RealField ,
                S: Copy + From<T>,
                V: Clone + for <'b> SubAssign<&'b V>
{

    // fn handle_try_step(&mut self, step: ODEStep<T>)-> ODEStep<T>{
    //     let step = step.map_dt(|dt| {
    //         self.ode_data_mut().next_dt = dt.clone();
    //         self.try_step(dt)});
    //     let ad = &mut self.adaptive_dat;
    //     if let ODEStep::Step(_) = step.clone(){
    //         ad.dx_norm = (self.norm)(&ad.dx);
    //         let f = ad.rtol / ad.dx_norm;
    //         let fp_lim =T::min( T::max(ad.step_size_mul(f) , from_f64!(T, &0.3) ), from_f64!(T, &2.0));
    //         let new_h = T::min(T::max(fp_lim * self.dat.h, ad.min_dt), ad.max_dt);
    //
    //         self.dat.update_step_size(new_h);
    //
    //         if f <= from_f64!(T, &1.0){
    //             return ODEStep::Reject;
    //         }
    //     }
    //
    //     step
    // }
}

impl<Sp, Fun, NormFn, S, V, T> AdaptiveODESolver<T> for ExpCFMSolver<Sp, Fun, NormFn, S, V, T>
    where       Fun: FnMut(&[T],(T,T)) -> Vec<Sp::L>,
                NormFn: FnMut(&V) -> T,
                Sp : ExponentialSplit<T, S, V>,
                T: RealField ,
                S: Copy + From<T>,
                V: Clone + for <'b> SubAssign<&'b V>{
    fn ode_adapt_data(&self) -> &ODEAdaptiveData<T, V> {
        &self.adaptive_dat
    }

    fn ode_adapt_data_mut(&mut self) -> &mut ODEAdaptiveData<T, V> {
        &mut self.adaptive_dat
    }
    fn norm(&mut self) -> T{
        (self.norm)(&self.adaptive_dat.dx)
    }
    fn validate_adaptive(&self) -> Result<(), ()>{
        if self.alph_err.is_some(){
            Ok(())
        } else {
            Err(())
        }
    }
}