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
    /// Stop when the specified solutions have been found or a specific generation,
    /// what happens before
    SolutionsFoundOrGeneration(usize, u64),
    /// Stop in a specific generation.
    Generation(u64),
    /// Stop when the mean progress in the last generations is lower than a specific threshold.
    Progress(f64),
    /// Stop when the generation is bigger than the first value and the mean progress in the last
    /// generations is lower than the specific threshold specified as the second value.
    GenerationAndProgress(u64, f64),
    /// Stop when the max fitness is bigger or equal than a specific threshold.
    MaxFitness(f64),
    /// Stop when the min fitness is bigger or equal than a specific threshold.
    MinFitness(f64),
    /// Stop when the average fitness is bigger or equal than a specific threshold.
    AvgFitness(f64),
}

impl StopCriterion for StopCriteria {
    fn stop(
        &self,
        generation: u64,
        progress: f64,
        n_solutions: usize,
        population_fitness: &[f64],
    ) -> bool {
        match self {
            StopCriteria::SolutionFound => n_solutions > 0,
            StopCriteria::SolutionsFound(i) => n_solutions >= *i,
            StopCriteria::SolutionsFoundOrGeneration(i, g) => n_solutions >= *i || generation >= *g,
            StopCriteria::Generation(g) => generation >= *g,
            StopCriteria::Progress(p) => progress <= *p,
            StopCriteria::GenerationAndProgress(g, p) => generation >= *g && progress <= *p,
            StopCriteria::MaxFitness(f) => {
                *population_fitness
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
                    >= *f
            }
            StopCriteria::MinFitness(f) => {
                *population_fitness
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
                    >= *f
            }
            StopCriteria::AvgFitness(f) => {
                population_fitness.iter().sum::<f64>() / population_fitness.len() as f64 >= *f
            }
        }
    }
}
