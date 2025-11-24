use ark_ff::Field;

pub fn mul_0_optimized<T: Field>(left: T, right: T) -> T {
    if left.is_zero() || right.is_zero() {
        T::zero()
    } else {
        left * right
    }
}

#[inline(always)]
pub fn mul_1_optimized<T: Field>(left: T, right: T) -> T {
    if left.is_one() {
        right
    } else if right.is_one() {
        left
    } else {
        left * right
    }
}

#[inline(always)]
pub fn mul_01_optimized<T: Field>(left: T, right: T) -> T {
    if left.is_zero() || right.is_zero() {
        T::zero()
    } else {
        mul_1_optimized(left, right)
    }
}
