//! This module contains the trait definition of the niches beta rate used in the
//! niches population refitness.

use niches_beta_rate::NichesBetaRates::*;
use slope_params::SlopeParams;
use OxigenStatsValues;

/// This trait defines the niches beta rate function used to compute the beta value.
pub trait NichesBetaRate: Send + Sync {
    /// Return the niches beta rate according to the generation, the statistics in the
    /// last generations, and the number of solutions found.
    ///
    /// The `stats_values` parameter is not mutable because this function is
    /// executed inside  parallel loop, so many statistics functions calls may
    /// slow down the execution.
    fn rate(&self, generation: u64, stats_values: &OxigenStatsValues, n_solutions: usize) -> f64;
}

/// Provided NichesBetaRate implementations.
pub enum NichesBetaRates {
    /// Constant value for the whole algorithm.
    Constant(f64),
    /// Linear function of iterations.
    Linear(SlopeParams),
    /// Quadratic function of iterations.
    Quadratic(SlopeParams),
}

impl NichesBetaRate for NichesBetaRates {
    fn rate(&self, generation: u64, _stats_values: &OxigenStatsValues, _n_solutions: usize) -> f64 {
        match self {
            Constant(c) => *c,
            Linear(sp) => sp.check_bound(sp.coefficient * generation as f64 + sp.start),
            Quadratic(sp) => {
                sp.check_bound(sp.coefficient * generation as f64 * generation as f64 + sp.start)
            }
        }
    }
}
