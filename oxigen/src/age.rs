//! This module contains the definition of the Age trait and the provided age functions.

use age::AgeFunctions::*;

/// This trait defines the manner in which old individuals are fitness decreased.
pub trait Age: Send + Sync {
    /// Returns the minimum number of generations required to start the age unfitness.
    fn age_threshold(&self) -> u64;

    /// Returns the fitness after decrease it by age using the
    /// number of `age_generations` that exceeds the age threshold.
    /// The generation when the individual is created is the age 0.
    /// This function is called when `age_generations` is bigger or equal than 0.
    /// E.g.:
    /// ```ignore
    /// individual.age=2, age_threshold=2 -> age_generations=0
    /// ```
    /// ```ignore
    /// individual.age=1, age_threshold=2 -> age_generations=-1 (not called)
    /// ```
    /// ```ignore
    /// individual.age=4, age_threshold=2 -> age_generations=2
    /// ```
    fn age_unfitness(&self, age_generations: u64, fitness: f64) -> f64;
}

/// Defines the age threshold parameter.
pub struct AgeThreshold(pub u64);
/// Defines the age slope parameter.
pub struct AgeSlope(pub f64);

/// Provided age functions.
pub enum AgeFunctions {
    /// No age unfitness.
    None,
    /// Linear function of iterations.
    Linear(AgeThreshold, AgeSlope),
    /// Quadratic function of iterations.
    Quadratic(AgeThreshold, AgeSlope),
}

impl Age for AgeFunctions {
    fn age_threshold(&self) -> u64 {
        match self {
            None => 0,
            Linear(threshold, _) => threshold.0,
            Quadratic(threshold, _) => threshold.0,
        }
    }

    fn age_unfitness(&self, age_generations: u64, fitness: f64) -> f64 {
        match self {
            None => fitness,
            Linear(_, sp) => fitness - sp.0 * age_generations as f64,
            Quadratic(_, sp) => fitness - sp.0 * age_generations as f64 * age_generations as f64,
        }
    }
}
