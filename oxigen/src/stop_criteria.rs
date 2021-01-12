//! This module contains the trait definition of StopCriterion and the provided stop criteria.

use OxigenStatsFieldFunction;
use OxigenStatsFields;
use OxigenStatsValues;

/// This trait can be implemented to provide a stop criterion of the genetic algorithm
/// using the generation, the statistics of the last executions, the number of solutions
/// found and the fitness of all individuals of the population.
pub trait StopCriterion: Send + Sync {
    /// Returns if the genetic algorithm has finished according to the generation,
    /// the statistics of the last generations, the number of solutions found
    /// and the fitness of the population.
    ///
    /// The `stats_values` parameter is mutable because it contains cached values that
    /// can be modified by `::stats` module, but it should not be manually modified.
    fn stop(
        &self,
        generation: u64,
        stats_values: &mut OxigenStatsValues,
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
    /// Stop when the progress of the best individual in the last generations is
    /// lower than a specific threshold.
    BestProgress(f64),
    /// Stop when the average progress in the last generations is lower than
    /// a specific threshold.
    AvgProgress(f64),
    /// Stop when the generation is bigger than the first value and the progress
    /// of the best individual in the last generations is lower than the
    /// specific threshold specified as the second value.
    GenerationAndBestProgress(u64, f64),
    /// Stop when the generation is bigger than the first value and the average
    /// progress in the last generations is lower than the specific threshold
    /// specified as the second value.
    GenerationAndAvgProgress(u64, f64),
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
        mut stats_values: &mut OxigenStatsValues,
        n_solutions: usize,
        population_fitness: &[f64],
    ) -> bool {
        match self {
            StopCriteria::SolutionFound => n_solutions > 0,
            StopCriteria::SolutionsFound(i) => n_solutions >= *i,
            StopCriteria::SolutionsFoundOrGeneration(i, g) => n_solutions >= *i || generation >= *g,
            StopCriteria::Generation(g) => generation >= *g,
            StopCriteria::BestProgress(p) => {
                let best_progress_avg =
                    OxigenStatsFields::BestProgressAvg.function()(&mut stats_values);
                best_progress_avg < *p
            }
            StopCriteria::AvgProgress(p) => {
                let avg_progress = OxigenStatsFields::AvgProgressAvg.function()(&mut stats_values);
                avg_progress < *p
            }
            StopCriteria::GenerationAndBestProgress(g, p) => {
                let best_progress_avg =
                    OxigenStatsFields::BestProgressAvg.function()(&mut stats_values);
                generation >= *g && best_progress_avg <= *p
            }
            StopCriteria::GenerationAndAvgProgress(g, p) => {
                let avg_progress = OxigenStatsFields::AvgProgressAvg.function()(&mut stats_values);
                generation >= *g && avg_progress <= *p
            }
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
