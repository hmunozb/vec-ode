use crate::lc::NormedLinearCombination;

pub fn trapezoid(){

}

pub trait Quadrature{
    
    fn sum<T: Copy + Into<f64>, S: Copy, V, L: NormedLinearCombination<T, S, V> >(
        &mut self,
        x_arr: & [V],
        lc: L,
        res: &mut V
    );
}
struct TrapezoidQuad{

}