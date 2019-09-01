//! This module contains the trait definition of the selection rate evolution.

use selection_rate::SelectionRates::*;
use slope_params::SlopeParams;

/// This trait defines the selection rate function used to modify the selection rate.
pub trait SelectionRate: Send + Sync {
    /// Returns the selection rate according to the generation, the progress in the
    /// last generations, the fitnesses of the population and the number of solutions found.
    fn rate(
        &self,
        generation: u64,
        progress: f64,
        n_solutions: usize,
        population_fitness: &[f64],
    ) -> usize;
}

/// Provided SelectionRate implementations.
pub enum SelectionRates {
    /// Constant value for the whole algorithm.
    Constant(usize),
    /// Linear function of iterations.
    Linear(SlopeParams),
    /// Quadratic function of iterations.
    Quadratic(SlopeParams),
}

impl SelectionRate for SelectionRates {
    fn rate(
        &self,
        generation: u64,
        _progress: f64,
        _n_solutions: usize,
        _population_fitness: &[f64],
    ) -> usize {
        match self {
            Constant(c) => *c,
            Linear(sp) => sp
                .check_bound(sp.coefficient * generation as f64 + sp.start)
                .ceil() as usize,
            Quadratic(sp) => sp
                .check_bound(sp.coefficient * generation as f64 * generation as f64 + sp.start)
                .ceil() as usize,
        }
    }
}
