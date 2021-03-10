//use alga::general::{RealField};
use crate::RealField;

use num_traits::{FromPrimitive};
use std::mem::swap;
use std::fmt;
use crate::from_f64;

pub trait Normed<R: Copy, V> {
    fn norm(v: &V) -> R;
}

#[derive(Debug, Clone)]
pub struct ODEError{
    pub msg: String
}

impl ODEError{
    pub fn new(s: &str) -> Self{
        Self{msg: String::from(s)}
    }
}

impl fmt::Display for ODEError{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.msg)
    }
}

impl std::error::Error for ODEError{ }

/// Marks the state of the ODE
#[derive(Clone)]
pub enum ODEState<T:Clone+Copy>{
    Ok(ODEStep<T>),
    Done,
    Err(ODEError)
}

/// Marks the current ODE stepping instructions
#[derive(Clone)]
pub enum ODEStep<T:Clone+Copy>{
    Step(T),    //Ordinary step by dt
    Chkpt,      //Perform a checkpoint update
    Reject,     //This step is rejected
    End,        //Reached end of integration
    Err(ODEError)
}

impl<T> ODEStep<T>
where T: Clone + Copy
{   /// Map an ODEError into ODEStep::Err
    pub fn map_dt<F>(self, mut f:  F)  -> ODEStep<T>
    where F: FnMut(T)-> Result<(), ODEError> {
        match self.clone(){
            ODEStep::Step(dt)  =>
                match f(dt) {   Ok(_) => ODEStep::Step(dt),
                                Err(e) => ODEStep::Err(e) },
            _ => self
        }
    }

    pub fn unwrap_dt(&self) -> T{
        match self{
            ODEStep::Step(dt) => *dt,
            _ => panic!("ODEStep::unwrap_dt expected Step(T) in enum")
        }
    }

    pub fn unwrap_dt_or(&self, dt2 : T) -> T{
        match self{
            ODEStep::Step(dt) => *dt,
            _ => dt2
        }
    }
}

/// Generic utility struct to group together ODE state variables
pub struct ODEData<T, V>
where V: Clone, T: Copy
{
    pub t0: T,
    pub tf: T,
    pub x0: V,

    pub t:  T,
    pub x: V,

    pub t_list: Vec<T>,
    pub tgt_t: usize,
    pub next_x: V,
    pub next_dt: T,
    pub h: T,
    pub prev_h: T
}

/// Group together data needed by adaptive solvers
pub struct ODEAdaptiveData<T, V>
    where V: Clone, T: Copy
{
    pub dx: V,
    pub atol: T,
    pub rtol: T,
    pub dx_norm: T,

    pub alpha: T,
    pub min_dt: T,
    pub max_dt: T,
    pub pow: T
}

impl<T, V> ODEAdaptiveData<T, V>
where V: Clone, T: RealField{
    pub fn new(init_dx: V, order: T, min_dt: T, max_dt: T) -> Self{
        //let atol = T::from_f64(1.0e-6).unwrap();
        let atol = from_f64!(1.0e-6);
        let rtol = T::from_f64(1.0e-4).unwrap();
        let dx_norm = T::zero();
        let alpha = T::from_f64(0.9).unwrap();
        let pow = order.recip();

        Self{dx: init_dx, atol, rtol, dx_norm, alpha, min_dt, max_dt, pow}
    }
    pub fn new_with_defaults(init_dx: V, order: T) -> Self{
        let min_dt = T::from_f64(1.0e-6).unwrap();
        let max_dt = T::from_f64(1.0).unwrap();
        Self::new(init_dx, order, min_dt, max_dt)
    }
    pub fn with_alpha(self, alpha: T) -> Self{
        Self{alpha, ..self}
    }

    pub fn step_size_mul(&self, f: T) -> T{
        self.alpha * T::powf(f, self.pow)
    }

}

impl<T, V> ODEData<T, V>
where V: Clone, T: RealField {
    pub fn new(t0: T, tf: T, x0: V, h: T) -> Self{
        let x = x0.clone();
        let t = t0.clone();
        let t_list = vec![t0, tf];
        let tgt_t = 0;
        let next_x = x0.clone();
        let next_dt = h;

        Self{t0, tf, x0, t, x, t_list, tgt_t, next_x, next_dt, h, prev_h: h}
    }

    pub fn current(&self) -> (T, &V){
        (self.t, &self.x)
    }

    pub fn into_current(self) -> (T, V){
        (self.t, self.x)
    }

    /// Returns a step instructions given a default step size dt_max
    /// If stepping by dt_max does not exceed the next checkpoint time or end time,
    /// it is simply returned as the next step size. Otherwise, the largest possible step
    /// is returned. The Checkpt instruction is then emitted until the checkpoint is updated
    /// The End instruction is perpetually emitted if the integration reaches the final time.
    pub fn step_size_of(&self, dt_max: T) -> ODEStep<T>{
        let checkpt_t = match self.t_list.get(self.tgt_t){
            None => return ODEStep::End,
            Some(t) => t.clone()
        };
        let dt_opt = check_step(self.t.clone(), checkpt_t, dt_max);
        match dt_opt{
            Some(dt) => ODEStep::Step(dt),
            None if self.tgt_t >= self.t_list.len() - 1 => ODEStep::End,
            None => ODEStep::Chkpt
        }
    }

    pub fn step_size(&self) -> ODEStep<T>{
        let step = self.step_size_of(self.h);
        step
    }

    /// Updates the current (t, x) to (next_x, t + next_dt)
    pub fn advance(&mut self) {
        swap(&mut self.x, &mut self.next_x);
        //self.x.clone_from(&self.next_x);
        self.t += self.next_dt.clone();
    }

    /// Updates the checkpoint index.
    /// If the solver is adaptive, also backtrack to the last accepted step size
    pub fn checkpoint_update(&mut self, _end: bool){
        self.tgt_t += 1;
        self.h = self.prev_h;
    }

    pub fn reset_step_size(&mut self, h: T){
        self.h = h;
        self.prev_h = h;
    }

    pub fn update_step_size(&mut self, h: T){
        self.prev_h = self.h;
        self.h = h;
    }
}

pub trait ODESolverBase: Sized{
    type TField: RealField;
    type RangeType: Clone;

    fn ode_data(&self) -> & ODEData<Self::TField, Self::RangeType>;
    fn ode_data_mut(&mut self) -> &mut ODEData<Self::TField, Self::RangeType>;
    fn into_ode_data(self) -> ODEData<Self::TField, Self::RangeType>;
    //fn time_range(&self) -> (Self::TField, Self::TField);
    fn current(&self) -> (Self::TField, & Self::RangeType){
        self.ode_data().current()
    }
    fn into_current(self) ->  (Self::TField, Self::RangeType){
        self.into_ode_data().into_current()
    }
    ///The next step size to attempt. Return None if integration has reached the end of the interval
    /// Return Checkpt if a checkpoint time is reached
    fn step_size(&self) -> ODEStep<Self::TField>{
        self.ode_data().step_size()
    }
    /// Attempt a step with the given step size
    fn try_step(&mut self, dt: Self::TField) -> Result<(), ODEError>;
    /// Accept the previously attempted step
    fn accept_step(&mut self){
        self.ode_data_mut().advance();
    }

    /// Any handling to be done when a checkpoint time is reached
    fn checkpoint(&mut self, end: bool){
        self.ode_data_mut().checkpoint_update(end);
    }
}

pub trait ODESolver : ODESolverBase{

    /// If stepping, update the ode_data with the step size and attempt the step
    /// For adaptive solvers, this should be overwritten to determine whether to reject
    /// the attempted step and update the next step size.
    fn handle_try_step(&mut self, step: ODEStep<Self::TField>) -> ODEStep<Self::TField>{
        step.map_dt(|dt| {
            self.ode_data_mut().next_dt = dt.clone();
            self.try_step(dt)})
    }

    /// Perform a single iteration of the ODE
    fn step(&mut self) -> ODEState<Self::TField>{
        let dt_opt = self.step_size();
        let res = self.handle_try_step(dt_opt);
        apply_step(self, res, false)
    }

}

pub trait AdaptiveODESolver<T: RealField>: ODESolver<TField=T>{

    fn ode_adapt_data(&self) -> & ODEAdaptiveData<T, Self::RangeType>;
    fn ode_adapt_data_mut(&mut self) -> &mut ODEAdaptiveData<T, Self::RangeType>;
    /// Return the norm of the most recent ODE x error (may be a mutable calculation)
    fn norm(&mut self) -> Self::TField;
    fn validate_adaptive(&self) -> Result<(), ()>{
        Ok(())
    }

    fn with_step_range(mut self, dt_min: T, dt_max: T) -> Self{
        if dt_min <= T::zero() || dt_max <= T::zero() || dt_max <= dt_min {
            panic!("Invalid step range: ({}, {})", dt_min, dt_max);
        }

        //let dt_range = (dt_min, dt_max);
        let h = T::sqrt(dt_min*dt_max);
        {
            let ad = self.ode_adapt_data_mut();
            ad.min_dt = dt_min;
            ad.max_dt = dt_max;
        }
        {
            let dat = self.ode_data_mut();
            dat.reset_step_size(h)
        }

        self
    }

    fn with_init_step(mut self, h: T) -> Self{
        let ad = self.ode_adapt_data_mut();
        if h < ad.min_dt || h > ad.max_dt{
            panic!("Step {} is not inside the range ({}, {})",
                   h, ad.min_dt, ad.max_dt);
        }
        let dat = self.ode_data_mut();
        dat.reset_step_size(h);
        self
    }

    fn with_tolerance(mut self, atol: T, rtol: T) -> Self {
        if atol <= T::zero() || rtol <= T::zero(){
            panic!("Invalid tolerances: atol={}, rtol={}", atol, rtol);
        }
        let ad = self.ode_adapt_data_mut();
        ad.atol = atol;
        ad.rtol = rtol;
        self
    }


    /// Try the step and adaptively change the step size from
    /// the resulting error estimate. Reject if error is too large
    fn handle_step_adaptive(&mut self, step: ODEStep<T>) -> ODEStep<T>{
        self.validate_adaptive().expect("adaptive step validation failed");

        let h = self.ode_data().h.clone();
        let step = self.handle_try_step(step);
        if let ODEStep::Step(_) = step.clone(){
            let dx_norm = self.norm();
            let ad = self.ode_adapt_data_mut();
            ad.dx_norm = dx_norm;
            let f = ad.rtol / ad.dx_norm;
            let fp_lim =T::min( T::max(ad.step_size_mul(f) ,
                                       from_f64!(0.3) ),
                                from_f64!(2.0));
            let new_h = T::min(T::max(fp_lim * h, ad.min_dt), ad.max_dt);

            self.ode_data_mut().update_step_size(new_h);

            if f <= from_f64!(1.0){
                return ODEStep::Reject;
            }
        }

        step
    }

    /// Perform a single iteration of the adaptive ODE
    fn step_adaptive(&mut self) -> ODEState<Self::TField>{
        let dt_opt = self.step_size();
        let res = self.handle_step_adaptive(dt_opt);
        apply_step(self, res, true)
    }


}

// pub struct AdaptiveODEController<S: ODESolver>{
//     pub inner_solver: S,
//     adaptive_dat: ODEAdaptiveData<S::TField, S::RangeType>,
//     //norm: Normed<S::TField, S::RangeType>
// }
//
// impl<S: ODESolver> ODESolverBase for AdaptiveODEController<S>{
//     type TField = S::TField;
//     type RangeType = S::RangeType;
//
//     fn ode_data(&self) -> &ODEData<Self::TField, Self::RangeType> {
//         self.inner_solver.ode_data()
//     }
//
//     fn ode_data_mut(&mut self) -> &mut ODEData<Self::TField, Self::RangeType> {
//         self.inner_solver.ode_data_mut()
//     }
//
//     fn into_ode_data(self) -> ODEData<Self::TField, Self::RangeType> {
//         self.inner_solver.into_ode_data()
//     }
//
//     fn try_step(&mut self, dt: Self::TField) -> Result<(), ODEError> {
//
//     }
// }
//
// impl<S: ODESolver> ODESolver for AdaptiveODEController<S>{ }
//
// impl<S: ODESolver> AdaptiveODESolver<S::TField> for AdaptiveODEController<S>{
//     fn ode_adapt_data(&self) -> &ODEAdaptiveData<Self::TField, Self::RangeType> {
//         &self.adaptive_dat
//     }
//
//     fn ode_adapt_data_mut(&mut self) -> &mut ODEAdaptiveData<<S as ODESolverBase>::TField, Self::RangeType> {
//         &mut self.adaptive_dat
//     }
//
//     fn norm(&mut self) -> Self::TField {
//         self.norm::
//     }
// }

pub fn check_step<T : RealField>(t0: T, tf: T, dt: T) -> Option<T>{
    let rem_t: T = tf.clone() - t0.clone();
    if rem_t.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative()){
        return None;
    }
    if rem_t.clone() < dt.clone() {
        Some(rem_t)
    } else {
        Some(dt)
    }
}

/// Return the state corresponding to the step taken by the solver
pub fn apply_step<D: ODESolver>(d: &mut D, step: ODEStep<D::TField>, adaptive: bool) -> ODEState<D::TField>{
    match step.clone(){
        ODEStep::Step(_) =>{
            d.accept_step();
            ODEState::Ok(step)
        },
        ODEStep::Chkpt =>{
            d.checkpoint(false);
            ODEState::Ok(step)
        },
        ODEStep::Reject => {
            if !adaptive{
                 let (t, _v) = d.current();
                 ODEState::Err(ODEError{msg: format!("Rejected step at time {}", t)})
            } else{
                ODEState::Ok(step)
            }
        }
        ODEStep::End =>{
            d.checkpoint(true);
            ODEState::Done
        },
        ODEStep::Err(e) =>{
            ODEState::Err(e)
        }
    }
}