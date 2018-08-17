//! This module contains the definition of the SurvivalPressure trait and the provided
//! survival_pressure functions.

use genotype::Genotype;

/// This trait defines the kill function used to remove individuals at the end of a generation.
pub trait SurvivalPressure<T, G: Genotype<T>>: Send + Sync {
    /// Returns the indexes of the individuals to be deleted according to the population size,
    /// the population and the fitness of the population. Population and fitness are sorted
    /// from bigger to lower fitness.
    fn kill(&self, population_size: usize, population: &[Box<G>], fitnesses: &[f64]) -> Vec<usize>;
}

/// Provided survival pressure functions.
pub enum SurvivalPressureFunctions {
    /// Kill worst individuals until reach the population size.
    Worst,
}

impl<T, G: Genotype<T>> SurvivalPressure<T, G> for SurvivalPressureFunctions {
    fn kill(
        &self,
        population_size: usize,
        population: &[Box<G>],
        _fitnesses: &[f64],
    ) -> Vec<usize> {
        let mut killed = Vec::with_capacity(population_size / 2);
        let mut i = population.len() - 1;
        while i >= population_size / 2 {
            killed.push(i);
            i -= 1;
        }
        killed
    }
}
