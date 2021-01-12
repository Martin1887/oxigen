//! This module contains the trait definition of the selection rate evolution.

use selection_rate::SelectionRates::*;
use slope_params::SlopeParams;
use OxigenStatsValues;

/// This trait defines the selection rate function used to modify the selection rate.
pub trait SelectionRate: Send + Sync {
    /// Return the selection rate according to the generation, the statistics in the
    /// last generations, the fitnesses of the population and the number of solutions found.
    ///
    /// The `stats_values` parameter is mutable because it contains cached values that
    /// can be modified by `::stats` module, but it should not be manually modified.
    fn rate(
        &self,
        generation: u64,
        stats_values: &mut OxigenStatsValues,
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
        _stats_values: &mut OxigenStatsValues,
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
