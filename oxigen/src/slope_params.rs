//! This module contains the definition of SlopeParams, useful to define a
//! polynomial function fastly.

/// This struct easies the definition of a simple polynomial.
pub struct SlopeParams {
    /// Value in which the slope starts.
    pub start: f64,
    /// Minimum (or maximum if the slope is positive) possible value of the slope.
    pub bound: f64,
    /// Coefficient of the slope (usually negative to decease rate along iterations).
    pub coefficient: f64,
}

impl SlopeParams {
    /// Returns the expected number: the bound or the result of the slope.
    pub fn check_bound(&self, y: f64) -> f64 {
        if self.coefficient >= 0.0f64 {
            if y < self.bound {
                y
            } else {
                self.bound
            }
        } else if y > self.bound {
            y
        } else {
            self.bound
        }
    }
}
