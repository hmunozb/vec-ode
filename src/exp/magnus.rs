use super::split_exp::{ExponentialSplit, Commutator};
use alga::general::{Ring, SupersetOf, ClosedAdd, RealField, SubsetOf};
use std::ops::{MulAssign, AddAssign, SubAssign};
use crate::{ODEData, ODESolverBase, ODEStep, ODEError, ODESolver, ODEState};
use std::marker::PhantomData;
use num_traits::Float;
use crate::exp::NormedExponentialSplit;

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
            V: Clone + for <'b> SubAssign<&'b V>
{

    let c_mid = T::from_subset(&0.288675134594812882254574390251);
    let b1 : T = dt * T::from_subset(&0.5);
    let b2_flt = -0.144337567297406441127287195125;
    let b2: T = dt * dt * T::from_subset(&b2_flt);

    let mid_t : T = t + b1;
    let t_sl = [   mid_t - c_mid*dt,
                            //mid_t,
                            mid_t + c_mid*dt];

    let mut l_vec = (*f)(&t_sl);

    // ME 2
    let mut w2 : Sp::L= sp.commutator(&l_vec[0], &l_vec[1]);
    w2 *= S::from(b2);

    // ME 1
    let mut w1: Sp::L = l_vec[0].clone();
    w1 += &l_vec[1];
    w1 *= S::from(b1);

    let u1 = sp.exp(&w1);
    // 4th order ME
    let mut w = w1;
    w += &w2;

    //Midpoint rule
    //let mut err_w1: Sp::L = l_vec[1].clone();
    //err_w1 *= S::from(b1);

    let u = sp.exp(&w);
    //let u_err = sp.exp(&err_w1);

    *xf = sp.map_exp(&u, x0);
    *x_err = sp.map_exp(&u1, x0);
    *x_err -= xf;

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
            Sp : Commutator<T, S, V> + NormedExponentialSplit<T, S, V>,
            Sp::L : MulAssign<S>,
            T: RealField,
            S: Ring + Copy + From<T>,
            V: Clone
{
    f: Fun,
    sp: Sp,
    dat: ODEData<T, V>,
    x_err: V,
    dt_range: (T, T),
    h: T,
    atol:T,
    rtol:T,
    err: T,
    _phantom: PhantomData<S>
}

impl<Sp, Fun, S, V, T> MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V> + NormedExponentialSplit<T, S, V>,
                Sp::L : MulAssign<S>,
                T: RealField,
                S: Ring + Copy + From<T>,
                V: Clone
{
    pub fn new(f: Fun, t0: T, tf: T, x0: V, sp: Sp) -> Self{
        let x_err = x0.clone();
        let h = T::from_subset(&1.0e-3);
        let dat = ODEData::new(t0, tf, x0);
        let dt_range = (T::from_subset(&1.0e-6), T::from_subset(&1.0));
        let atol = T::from_subset(&1.0e-6);
        let rtol = T::from_subset(&1.0e-6);
        f64::epsilon();
        Self{f, sp, dat, x_err, dt_range, h, atol, rtol, err: T::zero(), _phantom: PhantomData}
    }

    pub fn with_step_range(self, dt_min: T, dt_max: T) -> Self{
        if dt_min <= T::zero() || dt_max <= T::zero() || dt_max <= dt_min {
            panic!("Invalid step range: ({}, {})", dt_min, dt_max);
        }

        let dt_range = (dt_min, dt_max);
        let h = T::sqrt(dt_min*dt_max);
        Self{dt_range, h, ..self}
    }

    pub fn with_init_step(self, h: T) -> Self{
        if h < self.dt_range.0 || h > self.dt_range.1{
            panic!("Step {} is not inside the range ({}, {})",
                   h, self.dt_range.0, self.dt_range.0);
        }
        Self{h, ..self}
    }

    pub fn with_tolerance(self, atol: T, rtol: T) -> Self {
        if atol <= T::zero() || rtol <= T::zero(){
            panic!("Invalid tolerances: atol={}, rtol={}", atol, rtol);
        }
        Self{atol, rtol, ..self}
    }
}

impl<Sp, Fun, S, V, T> ODESolverBase for MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V> + NormedExponentialSplit<T, S, V>,
                Sp::L : MulAssign<S>
                    + for <'b> AddAssign<&'b Sp::L>,
                T: RealField,
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
//        dat.next_dt = dt;
        let res = magnus_42(&mut self.f, dat.t.clone(), &dat.x, &mut dat.next_x,
                  dt, &mut self.x_err, &mut self.sp);
        res

    }

    fn reject_step(&mut self) -> ODEState{
        ODEState::Ok
    }
}

impl<Sp, Fun, S, V, T> ODESolver for MagnusExpLinearSolver<Sp, Fun, S, V, T>
    where       Fun: FnMut(&[T]) -> Vec<Sp::L>,
                Sp : Commutator<T, S, V> + NormedExponentialSplit<T, S, V>,
                Sp::L : MulAssign<S>
                + for <'b> AddAssign<&'b Sp::L>,
                T: RealField,
                S: Ring + Copy + From<T>,
                V: Clone + for <'b> SubAssign<&'b V>
{
    fn handle_try_step(&mut self, step: ODEStep<T>)-> ODEStep<T>{
        let step = step.map_dt(|dt| {
            self.ode_data_mut().next_dt = dt.clone();
            self.try_step(dt)});

        if let ODEStep::Step(_) = step.clone(){
            self.err = self.sp.norm(&self.x_err);
            let f = self.rtol / self.err;
            self.h = T::from_subset(&0.9) * T::powf(f, T::from_subset(&(1.0/3.0))) * self.h;
            if f <= T::one(){
                return ODEStep::Reject;
            }
        }

        step
    }
}