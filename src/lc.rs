use itertools::zip_eq;

/// Trait for implementing linear combination operations
/// on a vector space type V with respect to a scalar type S
/// This trait is implemented on a trivial struct to define
/// specific vector space operations without grief.
pub trait LinearCombination<S: Copy, V>  {

    /// Mutably scale v to kv
    fn scale(v: &mut V, k: S);
    /// scale v to kv and store in target
    fn scalar_multiply_to(v: &V, k: S, target: &mut V);
    /// scale u to ku and add to v
    fn add_scalar_mul(v: &mut V, k: S, u: &V);
    /// add u to v
    fn add_assign_ref(v: &mut V, u: &V);
    /// subtract y from v
    fn delta(v: &mut V, y: &V);

    fn linear_combination<S2: Copy + Into<S>>(v: &mut V, v_arr: &[V], k_arr: &[S2]){
        if v_arr.is_empty() || k_arr.is_empty(){
            panic!("linear_combination: slices cannot be empty")
        }
        if v_arr.len() != k_arr.len(){
            panic!("linear_combination: slices must be the same length")
        }

        let (v0, v_arr) = v_arr.split_at(1);
        let (k0, k_arr) = k_arr.split_at(1);

        Self::scalar_multiply_to(&v0[0], k0[0].clone().into(), v);
        for (vi, &k) in v_arr.iter().zip(k_arr.iter()){
            Self::add_scalar_mul(v, k.into(), vi);
        }
    }

    fn linear_combination_iter<'a, IV, IS, S2:'a + Copy + Into<S>>(
        v: &'a mut V, v_iter: IV, s_iter: IS
    ) -> Result<(), ()>
        where IV: IntoIterator<Item=&'a V>, IS: IntoIterator<Item=&'a S2>
    {
        let mut v_iter = v_iter.into_iter();
        let mut s_iter = s_iter.into_iter();
        if let (Some(vi), Some(&si)) = (v_iter.next(), s_iter.next()){
            Self::scalar_multiply_to(vi, si.into(), v);
        } else {
            return Err(());
        }
        for (vi, &si) in zip_eq(v_iter,s_iter){
            Self::add_scalar_mul(v, si.into(), vi);
        }

        Ok(())
    }
}

pub trait NormedLinearCombination<T: Copy, S: Copy, V> : LinearCombination<S, V>{
    fn norm(&self, x: &V) -> T;
}

/// Equivalent to LinearCombination
/// but implemented on the vector space type V
/// rather than a trivial struct
pub trait LinearCombinationSpace<S>: Sized
    where S:Clone
{
    fn scale(&mut self, k: S);
    fn scalar_multiply_to(&self, k: S, target: &mut Self);
    fn add_scalar_mul(&mut self, k: S, u: &Self);
    fn add_assign_ref(&mut self, u: &Self);
    /// Subtracts the vector y from self
    fn delta(&mut self, y: &Self);

    fn linear_combination(&mut self, v_arr: &[Self], k_arr: &[S]){
        if v_arr.is_empty() || k_arr.is_empty(){
            panic!("linear_combination: slices cannot be empty")
        }
        if v_arr.len() != k_arr.len(){
            panic!("linear_combination: slices must be the same length")
        }

        let (v0, v_arr) = v_arr.split_at(1);
        let (k0, k_arr) = k_arr.split_at(1);

        Self::scalar_multiply_to(&v0[0], k0[0].clone(), self);
        for (v, k) in v_arr.iter().zip(k_arr.iter()){
            self.add_scalar_mul(k.clone(), v);
        }
    }
}

#[derive(Copy, Clone, Default)]
pub struct VecODELinearCombination;


impl<S: Copy, V> LinearCombination<S, V> for VecODELinearCombination
    where V: LinearCombinationSpace<S>{
    fn scale(v: &mut V, k: S) {
        v.scale(k);
    }

    fn scalar_multiply_to(v: &V, k: S, target: &mut V) {
        v.scalar_multiply_to(k, target);
    }

    fn add_scalar_mul(v: &mut V, k: S, other: & V) {
        v.add_scalar_mul(k,  other);
    }

    fn add_assign_ref(v: &mut V, other: &V) {
        v.add_assign_ref(other);
    }

    fn delta(v: &mut V, y: &V) {
        v.delta(y);
    }

}
