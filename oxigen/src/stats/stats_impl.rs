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
    fn function(&self) -> &dyn Fn(&mut OxigenStatsValues) -> f64;

    /// Variant without cache that should be used only when a mutable reference
    /// is not possible (e.g. multiple threads).
    fn uncached_function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64;
}

impl OxigenStatsFieldFunction for OxigenStatsFields {
    fn function(&self) -> &dyn Fn(&mut OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsFields::BestLastProgress => &|mut stats: &mut OxigenStatsValues| {
                *(get_best_progresses(&mut stats).last().unwrap())
            },
            OxigenStatsFields::BestProgressAvg => &|mut stats: &mut OxigenStatsValues| {
                let progresses = get_best_progresses(&mut stats);
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::BestProgressStd => &|mut stats: &mut OxigenStatsValues| {
                let avg = OxigenStatsFields::BestProgressAvg.function()(&mut stats);
                stddev(get_best_progresses(&mut stats), avg)
            },
            OxigenStatsFields::BestProgressMax => &|mut stats: &mut OxigenStatsValues| {
                *get_best_progresses(&mut stats)
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressMin => &|mut stats: &mut OxigenStatsValues| {
                *get_best_progresses(&mut stats)
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressP10 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_best_progress_histogram(&mut stats), 0.1)
            },
            OxigenStatsFields::BestProgressP25 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_best_progress_histogram(&mut stats), 0.25)
            },
            OxigenStatsFields::BestProgressMedian => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_best_progress_histogram(&mut stats), 0.5)
            },
            OxigenStatsFields::BestProgressP75 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_best_progress_histogram(&mut stats), 0.75)
            },
            OxigenStatsFields::BestProgressP90 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_best_progress_histogram(&mut stats), 0.9)
            },
            OxigenStatsFields::FitnessAvg => &|mut stats: &mut OxigenStatsValues| {
                // If the avg fitnesses are not cached it is faster to compute only the last one
                if stats.cache.avg_fitnesses.is_none() {
                    let fit = &stats.last_generations.back().unwrap().fitnesses;
                    fit.iter().sum::<f64>() / fit.len() as f64
                } else {
                    *get_avg_fitnesses(&mut stats).last().unwrap()
                }
            },
            OxigenStatsFields::FitnessStd => &|mut stats: &mut OxigenStatsValues| {
                let avg = OxigenStatsFields::FitnessAvg.function()(&mut stats);
                stddev(&stats.last_generations.back().unwrap().fitnesses, avg)
            },
            OxigenStatsFields::FitnessMax => &|stats: &mut OxigenStatsValues| {
                *stats
                    .last_generations
                    .back()
                    .unwrap()
                    .fitnesses
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::FitnessMin => &|stats: &mut OxigenStatsValues| {
                *stats
                    .last_generations
                    .back()
                    .unwrap()
                    .fitnesses
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::FitnessP10 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_fitness_histogram(&mut stats), 0.1)
            },
            OxigenStatsFields::FitnessP25 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_fitness_histogram(&mut stats), 0.25)
            },
            OxigenStatsFields::FitnessMedian => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_fitness_histogram(&mut stats), 0.5)
            },
            OxigenStatsFields::FitnessP75 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_fitness_histogram(&mut stats), 0.75)
            },
            OxigenStatsFields::FitnessP90 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_fitness_histogram(&mut stats), 0.9)
            },
            OxigenStatsFields::AvgLastProgress => {
                &|mut stats: &mut OxigenStatsValues| *get_avg_progresses(&mut stats).last().unwrap()
            }
            OxigenStatsFields::AvgProgressAvg => &|mut stats: &mut OxigenStatsValues| {
                let progresses = get_avg_progresses(&mut stats);
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::AvgProgressStd => &|mut stats: &mut OxigenStatsValues| {
                let avg = OxigenStatsFields::AvgProgressAvg.function()(&mut stats);
                stddev(get_avg_progresses(&mut stats), avg)
            },
            OxigenStatsFields::AvgProgressMax => &|mut stats: &mut OxigenStatsValues| {
                *get_avg_progresses(&mut stats)
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressMin => &|mut stats: &mut OxigenStatsValues| {
                *get_avg_progresses(&mut stats)
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressP10 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_avg_progress_histogram(&mut stats), 0.1)
            },
            OxigenStatsFields::AvgProgressP25 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_avg_progress_histogram(&mut stats), 0.25)
            },
            OxigenStatsFields::AvgProgressMedian => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_avg_progress_histogram(&mut stats), 0.5)
            },
            OxigenStatsFields::AvgProgressP75 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_avg_progress_histogram(&mut stats), 0.75)
            },
            OxigenStatsFields::AvgProgressP90 => &|mut stats: &mut OxigenStatsValues| {
                percentile(&get_avg_progress_histogram(&mut stats), 0.9)
            },
        }
    }

    fn uncached_function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsFields::BestLastProgress => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_best_fitnesses(&stats))
                    .last()
                    .unwrap()
            },
            OxigenStatsFields::BestProgressAvg => &|stats: &OxigenStatsValues| {
                let progresses = compute_progresses(&compute_best_fitnesses(&stats));
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::BestProgressStd => &|stats: &OxigenStatsValues| {
                let best_progresses = compute_progresses(&compute_best_fitnesses(&stats));
                let avg = best_progresses.iter().sum::<f64>() / best_progresses.len() as f64;
                stddev(&best_progresses, avg)
            },
            OxigenStatsFields::BestProgressMax => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_best_fitnesses(&stats))
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressMin => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_best_fitnesses(&stats))
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::BestProgressP10 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_best_fitnesses(&stats))),
                    0.1,
                )
            },
            OxigenStatsFields::BestProgressP25 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_best_fitnesses(&stats))),
                    0.25,
                )
            },
            OxigenStatsFields::BestProgressMedian => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_best_fitnesses(&stats))),
                    0.5,
                )
            },
            OxigenStatsFields::BestProgressP75 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_best_fitnesses(&stats))),
                    0.75,
                )
            },
            OxigenStatsFields::BestProgressP90 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_best_fitnesses(&stats))),
                    0.9,
                )
            },
            OxigenStatsFields::FitnessAvg => &|stats: &OxigenStatsValues| {
                let fit = &stats.last_generations.back().unwrap().fitnesses;
                fit.iter().sum::<f64>() / fit.len() as f64
            },
            OxigenStatsFields::FitnessStd => &|stats: &OxigenStatsValues| {
                let avg = OxigenStatsFields::FitnessAvg.uncached_function()(&stats);
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
            OxigenStatsFields::FitnessP10 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&stats.last_generations.back().unwrap().fitnesses),
                    0.1,
                )
            },
            OxigenStatsFields::FitnessP25 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&stats.last_generations.back().unwrap().fitnesses),
                    0.25,
                )
            },
            OxigenStatsFields::FitnessMedian => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&stats.last_generations.back().unwrap().fitnesses),
                    0.5,
                )
            },
            OxigenStatsFields::FitnessP75 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&stats.last_generations.back().unwrap().fitnesses),
                    0.75,
                )
            },
            OxigenStatsFields::FitnessP90 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&stats.last_generations.back().unwrap().fitnesses),
                    0.9,
                )
            },
            OxigenStatsFields::AvgLastProgress => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_avg_fitnesses(&stats))
                    .last()
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressAvg => &|stats: &OxigenStatsValues| {
                let progresses = compute_progresses(&compute_avg_fitnesses(&stats));
                progresses.iter().sum::<f64>() / progresses.len() as f64
            },
            OxigenStatsFields::AvgProgressStd => &|stats: &OxigenStatsValues| {
                let progresses = compute_progresses(&compute_avg_fitnesses(&stats));
                let avg = progresses.iter().sum::<f64>() / progresses.len() as f64;
                stddev(&progresses, avg)
            },
            OxigenStatsFields::AvgProgressMax => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_avg_fitnesses(&stats))
                    .iter()
                    .max_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressMin => &|stats: &OxigenStatsValues| {
                *compute_progresses(&compute_avg_fitnesses(&stats))
                    .iter()
                    .min_by(|x, y| x.partial_cmp(&y).unwrap())
                    .unwrap()
            },
            OxigenStatsFields::AvgProgressP10 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_avg_fitnesses(&stats))),
                    0.1,
                )
            },
            OxigenStatsFields::AvgProgressP25 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_avg_fitnesses(&stats))),
                    0.25,
                )
            },
            OxigenStatsFields::AvgProgressMedian => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_avg_fitnesses(&stats))),
                    0.5,
                )
            },
            OxigenStatsFields::AvgProgressP75 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_avg_fitnesses(&stats))),
                    0.75,
                )
            },
            OxigenStatsFields::AvgProgressP90 => &|stats: &OxigenStatsValues| {
                percentile(
                    &compute_histogram(&compute_progresses(&compute_avg_fitnesses(&stats))),
                    0.9,
                )
            },
        }
    }
}
