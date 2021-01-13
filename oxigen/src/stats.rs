//! This module provides the structs and functions used to get fitness
//! and progress statistics for stop criteria and outputs.

use rayon::prelude::*;
use std::collections::vec_deque::VecDeque;

/// Compute the standard deviation of a vector.
pub fn stddev(vector: &[f64], avg: f64) -> f64 {
    ((1_f64 / vector.len() as f64) * vector.iter().map(|x| (*x - avg).powi(2)).sum::<f64>()).sqrt()
}

/// Compute the percentile of a histogram, where `decimal_perc` is the
/// percentile divided by 100.
pub fn percentile(histogram: &[f64], decimal_perc: f64) -> f64 {
    let x = decimal_perc * histogram.len() as f64;
    if x as usize >= histogram.len() - 1 {
        histogram[histogram.len() - 1]
    } else if x.fract() < f64::EPSILON {
        (histogram[x as usize] + histogram[x as usize + 1]) / 2.0
    } else {
        histogram[x as usize + 1]
    }
}

/// Get the best fitnesses from the cached value or caching them.
pub fn get_best_fitnesses(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.best_fitnesses.is_none() {
        stats.cache.best_fitnesses = Some(compute_best_fitnesses(&stats));
    }
    stats.cache.best_fitnesses.as_ref().unwrap()
}

/// Get the best fitnesses from the cache or computing them without caching
/// (not mutable variant).
fn get_best_fitnesses_not_mutable(stats: &OxigenStatsValues) -> Vec<f64> {
    if stats.cache.best_fitnesses.is_none() {
        compute_best_fitnesses(&stats)
    } else {
        stats.cache.best_fitnesses.as_ref().unwrap().to_vec()
    }
}

/// Compute the best fitnesses from immutable stats. Use `get_best_fitnesses`
/// instead when a mutable reference of `OxigenStatsValues` is possible.
pub fn compute_best_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    (*stats
        .last_generations
        .par_iter()
        .map(|x| {
            *x.fitnesses
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        })
        .collect::<Vec<f64>>())
    .to_vec()
}

/// Get the worst fitnesses from the cached value or caching them.
pub fn get_worst_fitnesses(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.worst_fitnesses.is_none() {
        stats.cache.worst_fitnesses = Some(compute_worst_fitnesses(&stats));
    }
    stats.cache.worst_fitnesses.as_ref().unwrap()
}

/// Compute the worst fitnesses from immutable stats. Use `get_worst_fitnesses`
/// instead when a mutable reference of `OxigenStatsValues` is possible.
pub fn compute_worst_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    (*stats
        .last_generations
        .par_iter()
        .map(|x| {
            *x.fitnesses
                .iter()
                .min_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        })
        .collect::<Vec<f64>>())
    .to_vec()
}

/// Get the progress of the best individual from the cached value or caching them.
pub fn get_best_progresses(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.best_progresses.is_none() {
        let best = get_best_fitnesses_not_mutable(stats);
        stats.cache.best_progresses = Some(compute_progresses(&best));
    }
    stats.cache.best_progresses.as_ref().unwrap()
}

/// Compute the progress of the elements in `vector`. The resulting vector has
/// one element less because the progress cannot be computed in the first element.
pub fn compute_progresses(vector: &[f64]) -> Vec<f64> {
    let n = vector.len();
    // avoid the first element because the previous one does not exist (or has been deleted)
    let mut progresses: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 1..n {
        progresses.push(vector[i] - vector[i - 1]);
    }

    progresses
}

/// Get the avg fitness for the last generation from the cached value or caching them.
pub fn get_avg_fitnesses(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.avg_fitnesses.is_none() {
        stats.cache.avg_fitnesses = Some(compute_avg_fitnesses(&stats));
    }
    stats.cache.avg_fitnesses.as_ref().unwrap()
}

/// Get the avg fitnesses from the cache or computing them without caching
/// (not mutable variant).
fn get_avg_fitnesses_not_mutable(stats: &OxigenStatsValues) -> Vec<f64> {
    if stats.cache.avg_fitnesses.is_none() {
        compute_avg_fitnesses(&stats)
    } else {
        stats.cache.avg_fitnesses.as_ref().unwrap().to_vec()
    }
}

/// Compute the average fitnesses from immutable stats. Use `get_avg_fitnesses`
/// instead when a mutable reference of `OxigenStatsValues` is possible.
pub fn compute_avg_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    (*stats
        .last_generations
        .par_iter()
        .map(|x| x.fitnesses.iter().sum::<f64>() / x.fitnesses.len() as f64)
        .collect::<Vec<f64>>())
    .to_vec()
}

/// Get the avg progresses from the cached value or caching them.
pub fn get_avg_progresses(mut stats: &mut OxigenStatsValues) -> &[f64] {
    let avg = get_avg_fitnesses_not_mutable(&stats);
    stats.cache.avg_progresses = Some(compute_progresses(&avg));
    stats.cache.avg_progresses.as_ref().unwrap()
}

/// Get the best progress histogram for the last generations from the cached value or caching them.
pub fn get_best_progress_histogram(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.best_progress_histogram.is_none() {
        stats.cache.best_progress_histogram =
            Some(compute_histogram(get_best_progresses(&mut stats)));
    }
    stats.cache.best_progress_histogram.as_ref().unwrap()
}

/// Get the avg progress histogram for the last generations from the cached value or caching them.
pub fn get_avg_progress_histogram(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.avg_progress_histogram.is_none() {
        stats.cache.avg_progress_histogram =
            Some(compute_histogram(get_avg_progresses(&mut stats)));
    }
    stats.cache.avg_progress_histogram.as_ref().unwrap()
}

/// Get the fitness histogram for the last generation from the cached value or caching them.
pub fn get_fitness_histogram(mut stats: &mut OxigenStatsValues) -> &[f64] {
    if stats.cache.fitness_histogram.is_none() {
        stats.cache.fitness_histogram = Some(compute_histogram(
            &stats.last_generations.back().unwrap().fitnesses,
        ));
    }
    stats.cache.fitness_histogram.as_ref().unwrap()
}

/// Compute the histogram from unsorted values. Use a get_*_histogram function
/// instead when a mutable reference of `OxigenStatsValues` is possible.
pub fn compute_histogram(unsorted_values: &[f64]) -> Vec<f64> {
    let mut histogram = unsorted_values.to_vec();
    histogram.par_sort_unstable_by(|el1, el2| el1.partial_cmp(el2).unwrap());
    histogram
}

/// All statistics options that can be selected for statistics output.
pub enum OxigenStatsFields {
    /// The progress of the best individual in the last generation.
    BestLastProgress,
    /// The average progress of the best individual in the last generations.
    BestProgressAvg,
    /// The standard deviation of progress of the best individual in the last generations.
    BestProgressStd,
    /// The max progress of the best individual in the last generations.
    BestProgressMax,
    /// The min progress of the best individual in the last generations.
    BestProgressMin,
    /// The percentile 10 of progress of the best individual in the last generations.
    BestProgressP10,
    /// The percentile 25 of progress of the best individual in the last generations.
    BestProgressP25,
    /// The median of progress of the best individual in the last generations.
    BestProgressMedian,
    /// The percentile 75 of progress of the best individual in the last generations.
    BestProgressP75,
    /// The percentile 90 of progress of the best individual in the last generations.
    BestProgressP90,
    /// The average fitness in the last generation.
    FitnessAvg,
    /// The standard deviation of fitness in the last generation.
    FitnessStd,
    /// The maximum fitness in the last generation.
    FitnessMax,
    /// The minimum fitness in the last generation.
    FitnessMin,
    /// The percentile 10 of fitness in the last generation.
    FitnessP10,
    /// The percentile 25 of fitness in the last generation.
    FitnessP25,
    /// The median of fitness in the last generation.
    FitnessMedian,
    /// The percentile 75 of fitness in the last generation.
    FitnessP75,
    /// The percentile 90 of fitness in the last generation.
    FitnessP90,
    /// The generation average progress in the last generations.
    AvgLastProgress,
    /// The average of generation average progress in the last generations.
    AvgProgressAvg,
    /// The standard deviation of generation average progress in the last generations.
    AvgProgressStd,
    /// The maximum generation average progress in the last generations.
    AvgProgressMax,
    /// The minimum generation average progress in the last generations.
    AvgProgressMin,
    /// The percentile 10 of generation average progress in the last generations.
    AvgProgressP10,
    /// The percentile 25 of generation average progress in the last generations.
    AvgProgressP25,
    /// The median of generation average progress in the last generations.
    AvgProgressMedian,
    /// The percentile 75 of generation average progress in the last generations.
    AvgProgressP75,
    /// The percentile 90 of generation average progress in the last generations.
    AvgProgressP90,
}

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

pub struct OxigenStatsCache {
    /// The best fitness in the last generations.
    best_fitnesses: Option<Vec<f64>>,
    /// The worst fitness in the last generations.
    worst_fitnesses: Option<Vec<f64>>,
    /// The progress in fitness between the best individual in each generation
    /// and the best individual in the previous generation for the last generations.
    best_progresses: Option<Vec<f64>>,
    /// The average fitness in the last generations.
    avg_fitnesses: Option<Vec<f64>>,
    /// The progress between the average fitness in each generation and the
    /// average fitness in the previous generation for the last generations.
    avg_progresses: Option<Vec<f64>>,
    /// The sorted best progress in the last generations.
    best_progress_histogram: Option<Vec<f64>>,
    /// The sorted avg progress in the last generations.
    avg_progress_histogram: Option<Vec<f64>>,
    /// The sorted fitness in the last generation.
    fitness_histogram: Option<Vec<f64>>,
}

impl Default for OxigenStatsCache {
    fn default() -> Self {
        Self::new()
    }
}

impl OxigenStatsCache {
    pub fn new() -> Self {
        OxigenStatsCache {
            best_fitnesses: None,
            worst_fitnesses: None,
            best_progresses: None,
            avg_fitnesses: None,
            avg_progresses: None,
            best_progress_histogram: None,
            avg_progress_histogram: None,
            fitness_histogram: None,
        }
    }
}

/// Fitness values of each generation that can be used to extract statistics.
#[derive(Debug, PartialEq)]
pub struct OxigenStatsGenerationValues {
    /// The fitness of each individual in the generation.
    pub(crate) fitnesses: Vec<f64>,
}

/// A `VecDeque` where each element is a generation and the number of generations
/// is the defined in `GeneticExecution.stats_generations` field. Each generation
/// consists of a `OxigenStatsGenerationValues` object.
/// Generations are in order: the first element in `last_generations` is the
/// oldest recorded generation and the last element is the newest one.
pub struct OxigenStatsValues {
    /// The fitness and progress values in the last generations.
    pub(crate) last_generations: VecDeque<OxigenStatsGenerationValues>,
    /// The number of last generations to keep for statistics. Needed because
    /// VecDeque can put a bigger capacity in the creation.
    /// TODO: check when the exact capacity is stabilized or think in using another struct.
    pub(crate) capacity: usize,
    /// Cache statistics that are cleared in each update.
    pub(crate) cache: OxigenStatsCache,
}

impl OxigenStatsValues {
    /// Initialize the `VecDeque` with the `generations_stats` capacity.
    fn new(generations_stats: usize) -> Self {
        OxigenStatsValues {
            last_generations: VecDeque::with_capacity(generations_stats),
            capacity: generations_stats,
            cache: OxigenStatsCache::new(),
        }
    }
    /// Update the fitness and progress values in each generation.
    pub(crate) fn update(&mut self, gen_fitnesses: &[f64]) {
        // stats are computed after create the object to avoid fitnesses cloning
        let values = OxigenStatsGenerationValues {
            fitnesses: gen_fitnesses.to_vec(),
        };

        // if the queue is full move all elements one place to left and
        // replace the last one, insert at the end otherwise
        if self.last_generations.len() == self.capacity {
            self.last_generations.rotate_left(1);
            if let Some(x) = self.last_generations.back_mut() {
                *x = values
            }
        } else {
            self.last_generations.push_back(values);
        }

        // clear cache
        self.cache = OxigenStatsCache::new();
    }
}

/// Contain the actual values of the last generations, the schema to use to
/// print the statistics and the delimiter used in the CSV output stats file.
/// The stats hierarchy is:
/// OxigenStats
///     OxigenStatsValues
///         OxigenStatsGenerationValues
///         OxigenStatsCache
///     OxigenStatsSchema
///         OxigenStatsInstantiatedField
///             OxigenStatsAllFields
///                 OxigenStatsFields
pub(crate) struct OxigenStats {
    pub(crate) values: OxigenStatsValues,
    schema: OxigenStatsSchema,
    fields_delimiter: String,
}

impl OxigenStats {
    pub(crate) fn new(generations_stats: usize, delimiter: &str) -> Self {
        OxigenStats {
            values: OxigenStatsValues::new(generations_stats),
            schema: OxigenStatsSchema::new(),
            fields_delimiter: delimiter.to_string(),
        }
    }

    pub(crate) fn add_field(
        &mut self,
        name: &str,
        field: Box<dyn OxigenStatsFieldFunction>,
    ) -> &mut Self {
        self.schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from(name),
            enabled: true,
            field,
        });
        self
    }
    pub(crate) fn enable_field(&mut self, field: OxigenStatsFields) -> &mut Self {
        self.schema.fields[field as usize].enabled = true;
        self
    }
    pub(crate) fn disable_field(&mut self, field: OxigenStatsFields) -> &mut Self {
        self.schema.fields[field as usize].enabled = false;
        self
    }
    pub(crate) fn set_delimiter(&mut self, delimiter: &str) -> &mut Self {
        self.fields_delimiter = delimiter.to_string();
        self
    }

    /// Header String without ending line break.
    pub(crate) fn header(&self) -> String {
        let mut header = String::from("");
        let mut empty = true;
        for field in &self.schema.fields {
            if field.enabled {
                if empty {
                    header.push_str(&field.name);
                } else {
                    header.push_str(&format!("{}{}", self.fields_delimiter, field.name));
                }
                empty = false;
            }
        }

        header
    }

    /// The value of the current generation stats without ending line break.
    pub(crate) fn stats_line(&mut self, generation: u64, solutions_found: usize) -> String {
        let mut line = String::from("");
        // generation and solutions found are special and provided by parameters
        line.push_str(&format!(
            "{}{}{}",
            generation, self.fields_delimiter, solutions_found
        ));
        // the rest of enabled fields (statistics)
        for field in &self.schema.fields[2..] {
            line.push_str(&format!(
                "{}{}",
                self.fields_delimiter,
                field.field.function()(&mut self.values),
            ));
        }

        line
    }
}

/// Schema for the statistics CSV file.
struct OxigenStatsSchema {
    fields: Vec<OxigenStatsInstantiatedField>,
}

impl OxigenStatsSchema {
    fn new() -> Self {
        let mut schema = OxigenStatsSchema {
            fields: Vec::with_capacity(OxigenStatsAllFields::count()),
        };
        // Probably strum or a custom macro could be used for this
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Generation"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::Generation),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Solutions"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::Solutions),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best last progress"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestLastProgress,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress average"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP90,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness avg"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP90,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg last progress"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgLastProgress,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress average"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP90,
            )),
        });

        schema
    }
}

/// Definition of each field of the statistics file specifying name, enabling
/// status and the field.
struct OxigenStatsInstantiatedField {
    name: String,
    enabled: bool,
    field: Box<dyn OxigenStatsFieldFunction>,
}

/// Private struct containing the stats fields plus generation and solutions.
enum OxigenStatsAllFields {
    /// Current generation.
    Generation,
    /// Number of found solutions.
    Solutions,
    /// Statistics fields.
    StatsField(OxigenStatsFields),
}
impl OxigenStatsAllFields {
    /// The number of all fields. Note that this function must be changed when
    /// new fields are added. This can be done with a macro but it is too much
    /// cumbersome. Probably the `strum` crate could be used, but it seems a
    /// unnecessary dependency.
    fn count() -> usize {
        31
    }
}
impl OxigenStatsFieldFunction for OxigenStatsAllFields {
    fn function(&self) -> &dyn Fn(&mut OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsAllFields::StatsField(field) => field.function(),
            _ => panic!("The function cannot be applied over generation or solutions!"),
        }
    }
    fn uncached_function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsAllFields::StatsField(field) => field.uncached_function(),
            _ => panic!("The function cannot be applied over generation or solutions!"),
        }
    }
}
