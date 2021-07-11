//! This module provides auxiliary functions to compute statistics.

use rayon::prelude::*;
use OxigenStatsValues;

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
pub fn get_best_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut best_fitnesses = c.best_fitnesses.clone();
    // release the lock
    drop(c);

    if best_fitnesses.is_none() {
        best_fitnesses = Some(compute_best_fitnesses(stats));
        // poisoning cannot happen
        stats.cache.write().unwrap().best_fitnesses = Some(best_fitnesses.clone().unwrap());
    }
    best_fitnesses.unwrap()
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
pub fn get_worst_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut worst_fitnesses = c.worst_fitnesses.clone();
    // release the lock
    drop(c);

    if worst_fitnesses.is_none() {
        worst_fitnesses = Some(compute_worst_fitnesses(stats));
        // poisoning cannot happen
        stats.cache.write().unwrap().worst_fitnesses = Some(worst_fitnesses.clone().unwrap());
    }
    worst_fitnesses.unwrap()
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
pub fn get_best_progresses(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut best_progresses = c.best_progresses.clone();
    let mut best_fitnesses = c.best_fitnesses.clone();
    // release the lock
    drop(c);

    if best_progresses.is_none() {
        if best_fitnesses.is_none() {
            best_fitnesses = Some(get_best_fitnesses(stats));
        }
        best_progresses = Some(compute_progresses(best_fitnesses.as_ref().unwrap()));
        // poisoning cannot happen
        let mut c = stats.cache.write().unwrap();
        c.best_progresses = Some(best_progresses.clone().unwrap());
        c.best_fitnesses = best_fitnesses;
    }
    best_progresses.unwrap()
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
pub fn get_avg_fitnesses(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut avg_fitnesses = c.avg_fitnesses.clone();
    // release the lock
    drop(c);

    if avg_fitnesses.is_none() {
        avg_fitnesses = Some(compute_avg_fitnesses(stats));
        // poisoning cannot happen
        stats.cache.write().unwrap().avg_fitnesses = Some(avg_fitnesses.clone().unwrap());
    }
    avg_fitnesses.unwrap()
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
pub fn get_avg_progresses(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut avg_progresses = c.avg_progresses.clone();
    let mut avg_fitnesses = c.avg_fitnesses.clone();
    // release the lock
    drop(c);

    if avg_progresses.is_none() {
        if avg_fitnesses.is_none() {
            avg_fitnesses = Some(get_avg_fitnesses(stats));
        }
        avg_progresses = Some(compute_progresses(avg_fitnesses.as_ref().unwrap()).to_vec());
        // poisoning cannot happen
        let mut c = stats.cache.write().unwrap();
        c.avg_progresses = Some(avg_progresses.clone().unwrap());
        c.avg_fitnesses = avg_fitnesses;
    }
    avg_progresses.unwrap()
}

/// Get the best progress histogram for the last generations from the cached value or caching them.
pub fn get_best_progress_histogram(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut best_progress_histogram = c.best_progress_histogram.clone();
    let mut best_progresses = c.best_progresses.clone();
    // release the lock
    drop(c);

    if best_progress_histogram.is_none() {
        if best_progresses.is_none() {
            best_progresses = Some(get_best_progresses(stats));
        }
        best_progress_histogram = Some(compute_histogram(best_progresses.as_ref().unwrap()));
        // poisoning cannot happen
        let mut c = stats.cache.write().unwrap();
        c.best_progress_histogram = Some(best_progress_histogram.clone().unwrap());
        c.best_progresses = best_progresses;
    }
    best_progress_histogram.unwrap()
}

/// Get the avg progress histogram for the last generations from the cached value or caching them.
pub fn get_avg_progress_histogram(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut avg_progress_histogram = c.avg_progress_histogram.clone();
    let mut avg_progresses = c.avg_progresses.clone();
    // release the lock
    drop(c);

    if avg_progress_histogram.is_none() {
        if avg_progresses.is_none() {
            avg_progresses = Some(get_avg_progresses(stats));
        }
        avg_progress_histogram = Some(compute_histogram(avg_progresses.as_ref().unwrap()));
        // poisoning cannot happen
        let mut c = stats.cache.write().unwrap();
        c.avg_progress_histogram = Some(avg_progress_histogram.clone().unwrap());
        c.avg_progresses = avg_progresses;
    }
    avg_progress_histogram.unwrap()
}

/// Get the fitness histogram for the last generation from the cached value or caching them.
pub fn get_fitness_histogram(stats: &OxigenStatsValues) -> Vec<f64> {
    // poisoning cannot happen
    let c = stats.cache.read().unwrap();
    let mut fitness_histogram = c.fitness_histogram.clone();
    // release the lock
    drop(c);

    if fitness_histogram.is_none() {
        fitness_histogram = Some(compute_histogram(
            &stats.last_generations.back().unwrap().fitnesses,
        ));

        // poisoning cannot happen
        stats.cache.write().unwrap().fitness_histogram = Some(fitness_histogram.clone().unwrap());
    }
    fitness_histogram.unwrap()
}

/// Compute the histogram from unsorted values. Use a get_*_histogram function
/// instead when a mutable reference of `OxigenStatsValues` is possible.
pub fn compute_histogram(unsorted_values: &[f64]) -> Vec<f64> {
    let mut histogram = unsorted_values.to_vec();
    histogram.par_sort_unstable_by(|el1, el2| el1.partial_cmp(el2).unwrap());
    histogram
}
