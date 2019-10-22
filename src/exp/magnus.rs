use super::split_exp::{ExponentialSplit, Commutator};
use alga::general::{Ring, SupersetOf, ClosedAdd, RealField};
use std::ops::{MulAssign, AddAssign};
use crate::{ODEData, ODESolverBase, ODEStep, ODEError, ODESolver};
use std::marker::PhantomData;

fn midpoint<Fun, T, S, V, Sp>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T, sp: &mut Sp) -> Result<(), ODEError>
where   Fun: FnMut(T) -> Sp::L,
        Sp : ExponentialSplit<T, S, V>,
        Sp::L : MulAssign<S>,
        T: RealField,
        S: Ring + Copy + From<T>,
        V: Clone
{
    let t_mid = t + dt * T::from_subset(&0.5);
    let mut l = (*f)(t_mid);
    l *= S::from(dt);
    let u = sp.exp(&l);
    *xf = sp.map_exp(&u, x0);

    Ok(())
}

fn magnus_42<Fun, T, S, V, Sp>(
    f: &mut Fun, t: T, x0: &V, xf: &mut V, dt: T,
    x_err: &mut V, sp: &mut Sp,
) -> Result<(), ODEError>
    where   Fun: FnMut(&[T]) -> Vec<Sp::L>,
            Sp : Commutator<T, S, V>,
            Sp::L : Clone + MulAssign<S>
                + for <'b> AddAssign<&'b Sp::L>,
            T: RealField,
            S: Ring + Copy + From<T>,
            V: Clone
{

    let c_mid = T::from_subset(&0.288675134594812882254574390251);
    let b1 : T = dt * T::from_subset(&0.5);
    let b2_flt = -0.144337567297406441127287195125;
    let b2: T = dt * dt * T::from_subset(&b2_flt);

    let mid_t : T = t + b1;
    let t_sl = [t - c_mid*dt, mid_t, t + c_mid*dt];

    let mut l_vec = (*f)(&t_sl);

    // ME 2
    let mut w2 : Sp::L= sp.commutator(&l_vec[0], &l_vec[2]);
    w2 *= S::from(b2);

    // ME 1
    let mut w1: Sp::L = l_vec[0].clone();
    w1 += &l_vec[2];
    w1 *= S::from(b1);

    // 4th order ME
    let mut w = w1;
    w += &w2;

    //Midpoint rule
    let mut err_w1: Sp::L = l_vec[1].clone();
    err_w1 *= S::from(b1);

    let u = sp.exp(&w);
    let u_err = sp.exp(&err_w1);

    *xf = sp.map_exp(&u, x0);
    *x_err = sp.map_exp(&u_err, x0);

    Ok(())
}

pub struct MidpointExpLinearSolver<Sp, Fun, S, V, T>
where   Fun: FnMut(T) -> Sp::L,
        Sp : ExponentialSplit<T, S, V>,
        Sp::L : MulAssign<S>,
        T: RealField,
        S: Ring + Copy + From<T>,
        V: Clone
{
    f: Fun,
    sp: Sp,
    dat: ODEData<T, V>,
    h: T,
    _phantom: PhantomData<S>
}


impl<Sp, Fun, S, V, T> MidpointExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(T) -> Sp::L,
                Sp : ExponentialSplit<T, S, V>,
                Sp::L : MulAssign<S>,
                T: RealField,
                S: Ring + Copy + From<T>,
                V: Clone
{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T, sp: Sp) -> Self{
        let mut K: Vec<V> = Vec::new();
        K.resize(3, x0.clone());
        let dat = ODEData::new(t0, tf, x0);

        Self{f, sp, dat, h, _phantom: PhantomData}
    }
}

impl<Sp, Fun, S, V, T> ODESolverBase for MidpointExpLinearSolver<Sp, Fun, S, V, T>
where       Fun: FnMut(T) -> Sp::L,
            Sp : ExponentialSplit<T, S, V>,
            Sp::L : MulAssign<S>,
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

    fn try_step(&mut self, dt: T) -> Result<(), ODEError> {
        let dat = &mut self.dat;
        dat.next_dt = dt;
        midpoint(&mut self.f, dat.t.clone(), &dat.x,
                 &mut dat.next_x, dat.next_dt.clone(), &mut self.sp)
    }
}

impl<Sp, Fun, S, V, T> ODESolver for MidpointExpLinearSolver<Sp, Fun, S, V, T>
where  Fun: FnMut(T) -> Sp::L,
       Sp : ExponentialSplit<T, S, V>,
       Sp::L : MulAssign<S>,
       T: RealField,
       S: Ring + Copy + From<T>,
       V: Clone
{

}


pub struct MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where   Fun: FnMut(&[T]) -> Vec<Sp::L>,
            Sp : Commutator<T, S, V>,
            Sp::L : MulAssign<S>,
            T: RealField,
            S: Ring + Copy + From<T>,
            V: Clone
{
    f: Fun,
    sp: Sp,
    dat: ODEData<T, V>,
    x_err: V,
    h: T,
    _phantom: PhantomData<S>
}

impl<Sp, Fun, S, V, T> MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V>,
                Sp::L : MulAssign<S>,
                T: RealField,
                S: Ring + Copy + From<T>,
                V: Clone
{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, h: T, sp: Sp) -> Self{
        let mut K: Vec<V> = Vec::new();
        K.resize(3, x0.clone());
        let x_err = x0.clone();
        let dat = ODEData::new(t0, tf, x0);

        Self{f, sp, dat, x_err, h, _phantom: PhantomData}
    }
}

impl<Sp, Fun, S, V, T> ODESolverBase for MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V>,
                Sp::L : MulAssign<S>
                    + for <'b> AddAssign<&'b Sp::L>,
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

    fn try_step(&mut self, dt: T) -> Result<(), ODEError> {
        let dat = &mut self.dat;
        dat.next_dt = dt;
        magnus_42(&mut self.f, dat.t.clone(), &dat.x, &mut dat.next_x,
                  dat.next_dt.clone(), &mut self.x_err, &mut self.sp)
    }
}

impl<Sp, Fun, S, V, T> ODESolver for MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V>,
                Sp::L : MulAssign<S>
                + for <'b> AddAssign<&'b Sp::L>,
                T: RealField,
                S: Ring + Copy + From<T>,
                V: Clone
{

}