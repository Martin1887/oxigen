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
