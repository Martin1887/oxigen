//! This module contains the trait definition of the mutation rate evolution.

use mutation_rate::MutationRates::*;
use slope_params::SlopeParams;
use OxigenStatsValues;

/// This trait defines the mutation rate function used to modify the mutation rate.
pub trait MutationRate: Send + Sync {
    /// Return the mutation rate according to the generation, the statistics in the
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
        _stats_values: &mut OxigenStatsValues,
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
