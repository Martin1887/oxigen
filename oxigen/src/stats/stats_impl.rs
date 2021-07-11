//! This module provides the definition of the `OxigenStatsFieldFunction` trait
//! to implement statistics functions and contains the implementation fo the
//! built-in functions.

use stats_aux::*;
use OxigenStatsFields;
use OxigenStatsValues;

/// Trait that defines the implementation of statistics function in a generation.
pub trait OxigenStatsFieldFunction: Send + Sync {
    /// The function that computes the statistics value from the last generations
    /// fitnesses and progresses.
    fn function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64;
}

impl OxigenStatsFieldFunction for OxigenStatsFields {
    fn function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsFields::BestLastProgress => {
                &|stats: &OxigenStatsValues| *(get_best_progresses(&stats).last().unwrap())
            }
            OxigenStatsFields::BestProgressAvg => &|stats: &OxigenStatsValues| {
                let progresses = get_best_progresses(&stats);
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::BestProgressStd => &|stats: &OxigenStatsValues| {
                let avg = OxigenStatsFields::BestProgressAvg.function()(&stats);
                stddev(&get_best_progresses(&stats), avg)
            },
            OxigenStatsFields::BestProgressMax => &|stats: &OxigenStatsValues| {
                *get_best_progresses(&stats)
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressMin => &|stats: &OxigenStatsValues| {
                *get_best_progresses(&stats)
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressP10 => {
                &|stats: &OxigenStatsValues| percentile(&get_best_progress_histogram(&stats), 0.1)
            }
            OxigenStatsFields::BestProgressP25 => {
                &|stats: &OxigenStatsValues| percentile(&get_best_progress_histogram(&stats), 0.25)
            }
            OxigenStatsFields::BestProgressMedian => {
                &|stats: &OxigenStatsValues| percentile(&get_best_progress_histogram(&stats), 0.5)
            }
            OxigenStatsFields::BestProgressP75 => {
                &|stats: &OxigenStatsValues| percentile(&get_best_progress_histogram(&stats), 0.75)
            }
            OxigenStatsFields::BestProgressP90 => {
                &|stats: &OxigenStatsValues| percentile(&get_best_progress_histogram(&stats), 0.9)
            }
            OxigenStatsFields::FitnessAvg => &|stats: &OxigenStatsValues| {
                // If the avg fitnesses are not cached it is faster to compute only the last one
                if stats.cache.read().unwrap().avg_fitnesses.is_none() {
                    let fit = &stats.last_generations.back().unwrap().fitnesses;
                    fit.iter().sum::<f64>() / fit.len() as f64
                } else {
                    *get_avg_fitnesses(&stats).last().unwrap()
                }
            },
            OxigenStatsFields::FitnessStd => &|stats: &OxigenStatsValues| {
                let avg = OxigenStatsFields::FitnessAvg.function()(&stats);
                stddev(&stats.last_generations.back().unwrap().fitnesses, avg)
            },
            OxigenStatsFields::FitnessMax => &|stats: &OxigenStatsValues| {
                *stats
                    .last_generations
                    .back()
                    .unwrap()
                    .fitnesses
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::FitnessMin => &|stats: &OxigenStatsValues| {
                *stats
                    .last_generations
                    .back()
                    .unwrap()
                    .fitnesses
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::FitnessP10 => {
                &|stats: &OxigenStatsValues| percentile(&get_fitness_histogram(&stats), 0.1)
            }
            OxigenStatsFields::FitnessP25 => {
                &|stats: &OxigenStatsValues| percentile(&get_fitness_histogram(&stats), 0.25)
            }
            OxigenStatsFields::FitnessMedian => {
                &|stats: &OxigenStatsValues| percentile(&get_fitness_histogram(&stats), 0.5)
            }
            OxigenStatsFields::FitnessP75 => {
                &|stats: &OxigenStatsValues| percentile(&get_fitness_histogram(&stats), 0.75)
            }
            OxigenStatsFields::FitnessP90 => {
                &|stats: &OxigenStatsValues| percentile(&get_fitness_histogram(&stats), 0.9)
            }
            OxigenStatsFields::AvgLastProgress => {
                &|stats: &OxigenStatsValues| *get_avg_progresses(&stats).last().unwrap()
            }
            OxigenStatsFields::AvgProgressAvg => &|stats: &OxigenStatsValues| {
                let progresses = get_avg_progresses(&stats);
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::AvgProgressStd => &|stats: &OxigenStatsValues| {
                let avg = OxigenStatsFields::AvgProgressAvg.function()(&stats);
                stddev(&get_avg_progresses(&stats), avg)
            },
            OxigenStatsFields::AvgProgressMax => &|stats: &OxigenStatsValues| {
                *get_avg_progresses(&stats)
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressMin => &|stats: &OxigenStatsValues| {
                *get_avg_progresses(&stats)
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressP10 => {
                &|stats: &OxigenStatsValues| percentile(&get_avg_progress_histogram(&stats), 0.1)
            }
            OxigenStatsFields::AvgProgressP25 => {
                &|stats: &OxigenStatsValues| percentile(&get_avg_progress_histogram(&stats), 0.25)
            }
            OxigenStatsFields::AvgProgressMedian => {
                &|stats: &OxigenStatsValues| percentile(&get_avg_progress_histogram(&stats), 0.5)
            }
            OxigenStatsFields::AvgProgressP75 => {
                &|stats: &OxigenStatsValues| percentile(&get_avg_progress_histogram(&stats), 0.75)
            }
            OxigenStatsFields::AvgProgressP90 => {
                &|stats: &OxigenStatsValues| percentile(&get_avg_progress_histogram(&stats), 0.9)
            }
        }
    }
}
