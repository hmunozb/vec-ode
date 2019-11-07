use alga::general::{RealField};
use approx::RelativeEq;
use num_traits::Zero;

#[derive(Clone)]
pub struct ODEError{
    pub msg: String
}

impl ODEError{
    pub fn new(s: &str) -> Self{
        Self{msg: String::from(s)}
    }
}

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
{
    pub fn map_dt<F>(self, mut f:  F)  -> ODEStep<T>
    where F: FnMut(T)-> Result<(), ODEError> {
        match self.clone(){
            ODEStep::Step(dt)  =>
                match f(dt) {   Ok(_) => ODEStep::Step(dt),
                                Err(e) => ODEStep::Err(e) },
            _ => self
        }
    }
}

/// Generic utility struct to group together ODE state variables
pub struct ODEData<T, V>
where V: Clone, T: Clone
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
}

/// Group together data needed by adaptive solvers
pub struct ODEAdaptiveData<T, V>
    where V: Clone, T: Clone
{
    pub dx: V,
    pub atol: T,
    pub rtol: T,
    pub dx_norm: T,

    pub alpha: T,
    pub min_dx: T,
    pub max_dx: T,
    pub pow: T
}

impl<V> ODEAdaptiveData<f64, V>
where V: Clone{
    pub fn new(init_dx: V, order: f64, min_dx: f64, max_dx: f64) -> Self{
        let atol = 1.0e-6;
        let rtol = 1.0e-4;
        let dx_norm = 0.0;
        let alpha = 0.9;
        let pow = order.recip();

        Self{dx: init_dx, atol, rtol, dx_norm, alpha, min_dx, max_dx, pow}
    }
    pub fn with_alpha(self, alpha: f64) -> Self{
        Self{alpha, ..self}
    }
}

impl<T, V> ODEData<T, V>
where V: Clone, T: Clone + RealField {
    pub fn new(t0: T, tf: T, x0: V) -> Self{
        let x = x0.clone();
        let t = t0.clone();
        let mut t_list = vec![t0, tf];
        let tgt_t = 0;
        let next_x = x0.clone();
        let next_dt = t0.clone();

        Self{t0, tf, x0, t, x, t_list, tgt_t, next_x, next_dt}
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
    pub fn step_size(&self, dt_max: T) -> ODEStep<T>{
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

    /// Updates the current (t, x) to (next_x, t + next_dt)
    pub fn step_update(&mut self) {
        self.x.clone_from(&self.next_x);
        self.t += self.next_dt.clone();
    }

    /// Updates the checkpoint index.
    pub fn checkpoint_update(&mut self, end: bool){
        self.tgt_t += 1;
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
    fn step_size(&self) -> ODEStep<Self::TField>;
    /// Attempt a step with the given step size
    fn try_step(&mut self, dt: Self::TField) -> Result<(), ODEError>;
    /// Accept the previously attempted step
    fn accept_step(&mut self){
        self.ode_data_mut().step_update();
    }

    /// Any handling to be done when a checkpoint time is reached
    fn checkpoint(&mut self, end: bool){
        self.ode_data_mut().checkpoint_update(end);
    }
    /// Handle a step rejection. Adaptive solvers should adjust their step size and continue
    /// if possible. Rejection is an error for non-adaptive solvers
    fn reject_step(&mut self) -> ODEState<Self::TField>{
        let (t, v) = self.current();
        ODEState::Err(ODEError{msg: format!("Rejected step at time {}", t)})
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
        match res.clone(){
            ODEStep::Step(_) =>{
                self.accept_step();
                ODEState::Ok(res)
            },
            ODEStep::Chkpt =>{
                self.checkpoint(false);
                ODEState::Ok(res)
            },
            ODEStep::Reject => {
                self.reject_step()
            }
            ODEStep::End =>{
                self.checkpoint(true);
                ODEState::Done
            },
            ODEStep::Err(e) =>{
                ODEState::Err(e)
            }
        }
    }

}

pub trait AdaptiveODESolver: ODESolverBase{

}

pub fn check_step<T : RealField>(t0: T, tf: T, dt: T) -> Option<T>{
    let rem_t: T = tf.clone() - t0.clone();
    if rem_t.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative()){
        return None;
    }
    if rem_t.clone() < dt.clone(){
        return Some(rem_t);
    } else {
        return Some(dt);
    }
}
