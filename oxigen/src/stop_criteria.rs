//! This module contains the trait definition of StopCriterion and the provided stop criteria.

/// This trait can be implemented to provide a stop criterion of the genetic algorithm
/// using the generation, the progress of the last executions, the number of solutions
/// found and the fitness of all individuals of the population.
pub trait StopCriterion: Send + Sync {
    /// Returns if the genetic algorithm has finished according to the generation,
    /// the progress of the last generations, the number of solutions found and the
    /// fitness of the population.
    fn stop(
        &self,
        generation: u64,
        progress: f64,
        n_solutions: usize,
        population_fitness: &[f64],
    ) -> bool;
}

/// Provided stop criteria.
pub enum StopCriteria {
    /// Stop when a solution has been found.
    SolutionFound,
    /// Stop when this number of solutions have been found.
    SolutionsFound(usize),
    /// Stop in a specific generation.
    Generation(u64),
    /// Stop when the mean progress in the last generations is lower than a specific threshold.
    Progress(f64),
}

impl StopCriterion for StopCriteria {
    fn stop(
        &self,
        generation: u64,
        progress: f64,
        n_solutions: usize,
        _population_fitness: &[f64],
    ) -> bool {
        match self {
            StopCriteria::SolutionFound => n_solutions > 0,
            StopCriteria::SolutionsFound(i) => n_solutions >= *i,
            StopCriteria::Generation(i) => generation >= *i,
            StopCriteria::Progress(p) => progress <= *p,
        }
    }
}
