//! This module contains the trait definition of the mutation rate evolution.

use mutation_rate::MutationRates::*;
use slope_params::SlopeParams;

/// This trait defines the mutation rate function used to modify the mutation rate.
pub trait MutationRate: Send + Sync {
    /// Returns the mutation rate according to the generation, the progress in the
    /// last generations, the fitnesses of the population and the number of solutions found.
    fn rate(
        &self,
        generation: u64,
        progress: f64,
        n_solutions: usize,
        population_fitness: &[f64],
    ) -> f64;
}

/// Provided MutationRate implementations.
pub enum MutationRates {
    /// Constant value for the whole algorithm.
    Constant(f64),
    /// Linear function of generations.
    Linear(SlopeParams),
    /// Quadratic function of generations.
    Quadratic(SlopeParams),
}

impl MutationRate for MutationRates {
    fn rate(
        &self,
        generation: u64,
        _progress: f64,
        _n_solutions: usize,
        _population_fitness: &[f64],
    ) -> f64 {
        match self {
            Constant(c) => *c,
            Linear(sp) => sp.check_bound(sp.coefficient * generation as f64 + sp.start),
            Quadratic(sp) => {
                sp.check_bound(sp.coefficient * generation as f64 * generation as f64 + sp.start)
            }
        }
    }
}
