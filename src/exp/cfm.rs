
use alga::general::{Ring, SupersetOf, RealField};
use ndarray::{ArrayView1, ArrayView2, Array2};

use crate::base::{LinearCombination, ODEData, ODEStep, ODEState,
                  ODESolverBase, ODEError, ODEAdaptiveData};
use crate::exp::{ExponentialSplit, NormedExponentialSplit};
use std::marker::PhantomData;
use itertools::Itertools;
use std::mem::swap;
use crate::dat::cfqm::{CFM_R4_J2_GL, CFM_R2_J1_GL};
use crate::dat::quad::C_GAUSS_LEGENDRE_4;
use std::ops::SubAssign;
use crate::ODESolver;

///
/// Evaluates the linear combination of operators k := a.m dt
/// Then evaluates the exponential action x1 := exp(k) x0 with the splitting sp
///
pub fn cfm_exp<'a, Sp, T, S, V>(
    x0: &V, x1: &mut V, dt: T, m: &Vec<Sp::L>, k: &mut Sp::L,
    a: ArrayView1<'a, S>, sp: &mut Sp
)
    where   Sp: ExponentialSplit<T, S, V>,
            Sp::L : LinearCombination<S>,// + MulAssign<S> + for <'b> AddAssign<&'b Sp::L>,
            T: Ring + Copy + SupersetOf<f64>,
            S: Ring + Copy + From<T>,
            V: Clone
{

    m[0].scalar_multiply_to(a[0].clone(), k);
    for (ai, mi) in a.iter().skip(1)
        .zip(m.iter().skip(1)){
        k.add_scalar_mul(ai.clone(), mi);
    }
    k.scale(S::from(dt));
    let u = sp.exp(&*k);
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
        Fun: FnMut(&[T]) -> Vec<Sp::L>,
        Sp :ExponentialSplit<T, S, V>,
        Sp::L : LinearCombination<S>,
        T: Ring + Copy + SupersetOf<f64>,
        S: Ring + Copy + From<T>,
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

    let va = (*f)(&t_arr);

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

pub struct ExpCFMSolver<Sp, Fun, S, V, T>
    where
        Fun: FnMut(&[T]) -> Vec<Sp::L>,
        Sp :ExponentialSplit<T, S, V>,
        T: RealField,
        S: Ring + Copy + From<T>,
        V: Clone
{
    f: Fun,
    sp: Sp,
    dat: ODEData<T, V>,
    adaptive_dat: ODEAdaptiveData<T, V>,
    h: T,
    K: Vec<V>,
    KA: Vec<Sp::L>,
    c: Vec<T>,
    alpha: Array2<S>,
    alph_err: Array2<S>
}

impl<Sp, Fun, S, V, T> ExpCFMSolver<Sp, Fun, S, V, T>
where
    Fun: FnMut(&[T]) -> Vec<Sp::L>,
    Sp : NormedExponentialSplit<T, S, V>,
    T: RealField + SupersetOf<f64>,
    S: Ring + Copy + From<T>,
    V: Clone + for <'b> SubAssign<&'b V>{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T, sp: Sp) -> Self{
        let mut K: Vec<V> = Vec::new();
        let mut KA: Vec<Sp::L> = Vec::new();
        K.resize(3, x0.clone());
        KA.resize_with(1, | | sp.lin_zero());
        let c = C_GAUSS_LEGENDRE_4.iter()
            .map(|x| T::from_subset(x))
            .collect_vec();
        let alpha = Array2::from_shape_vec((2,2),
            CFM_R4_J2_GL.iter()
                .map(|x|S::from(T::from_subset(x)))
                .collect_vec()
            ).unwrap();
        let alph_err = Array2::from_shape_vec((2,1),
             CFM_R2_J1_GL.iter()
                 .map(|x|S::from(T::from_subset(x)))
                 .collect_vec()
            ).unwrap();
        let dat = ODEData::new(t0, tf, x0.clone());
        let adaptive_dat = ODEAdaptiveData::new_with_defaults(
            x0, T::from_subset(&3.0));
        Self{f, sp, dat, adaptive_dat, h, K, KA, c, alpha, alph_err}
    }
}


impl<Sp, Fun, S, V, T> ODESolverBase for ExpCFMSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : NormedExponentialSplit<T, S, V>,
                T: RealField + SupersetOf<f64>,
                S: Ring + Copy + From<T>,
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

    fn step_size(&self) -> ODEStep<T>{
        self.dat.step_size(self.h.clone())
    }

    fn try_step(&mut self, dt: T) -> Result<(), ODEError> {
        let dat = &mut self.dat;
        dat.next_dt = dt;
        cfm_general(&mut self.f, dat.t,&dat.x, &mut dat.next_x, self.h, &self.c, self.alpha.view(),
        &mut self.K, &mut self.KA, &mut self.sp, Some(&mut self.adaptive_dat.dx),
                    Some(self.alph_err.view()))
    }

    fn reject_step(&mut self) -> ODEState<T>{
        ODEState::Ok(ODEStep::Reject)
    }
}

impl<Sp, Fun, S, V, T> ODESolver for ExpCFMSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp :NormedExponentialSplit<T, S, V>,
                T: RealField + SupersetOf<f64>,
                S: Ring + Copy + From<T>,
                V: Clone + for <'b> SubAssign<&'b V>
{

    fn handle_try_step(&mut self, step: ODEStep<T>)-> ODEStep<T>{
        let step = step.map_dt(|dt| {
            self.ode_data_mut().next_dt = dt.clone();
            self.try_step(dt)});
        let ad = &mut self.adaptive_dat;
        if let ODEStep::Step(_) = step.clone(){
            ad.dx_norm = self.sp.norm(&ad.dx);
            let f = ad.rtol / ad.dx_norm;
            self.h = ad.step_size_mul(f) * self.h;
            if f <= T::one(){
                return ODEStep::Reject;
            }
        }

        step
    }
}