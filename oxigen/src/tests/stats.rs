use std::collections::VecDeque;
use OxigenStats;
use OxigenStatsFieldFunction;
use OxigenStatsFields;
use OxigenStatsGenerationValues;

/// Get the stats object in which test all functions.
fn get_stats() -> OxigenStats {
    let mut stats = OxigenStats::new(16, ",");

    // Put only 8 values to test that it works without being full
    stats.values.update(&[1.0, 2.0, 4.0, 8.0]);
    stats.values.update(&[2.0, 2.0, 2.0, 2.0]);
    stats.values.update(&[4.0, 4.0, 4.0, 4.0]);
    stats.values.update(&[0.0, 2.0, 0.0, 8.0]);
    stats.values.update(&[4.0, 0.0, 0.0, 0.0]);
    stats.values.update(&[-1.0, -2.0, -4.0, -8.0]);
    stats.values.update(&[-1.0, 2.0, 4.0, -8.0]);
    stats.values.update(&[500.0, 2.0, 4.0, 8.0]);

    stats
}

/// Compute the standard deviation of a vector.
fn stddev(vector: &[f64], avg: f64) -> f64 {
    ((1_f64 / vector.len() as f64) * vector.iter().map(|x| (*x - avg).powi(2)).sum::<f64>()).sqrt()
}

/// Compute the percentile of a histogram, where `decimal_perc` is the
/// percentile divided by 100.
fn percentile(histogram: &[f64], decimal_perc: f64) -> f64 {
    let x = decimal_perc * histogram.len() as f64;
    if x as usize >= histogram.len() - 1 {
        histogram[histogram.len() - 1]
    } else if x.fract() < f64::EPSILON {
        (histogram[x as usize] + histogram[x as usize + 1]) / 2.0
    } else {
        histogram[x as usize + 1]
    }
}

/// Get the progresses from a vector.
fn get_progresses(vector: &[f64]) -> Vec<f64> {
    let n = vector.len();
    // avoid the first element because the previous one does not exist (or has been deleted)
    let mut progresses: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 1..n {
        progresses.push(vector[i] - vector[i - 1]);
    }

    progresses
}

#[test]
fn test_stats_update() {
    let mut stats = OxigenStats::new(2, ",");

    let generations = vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
    ];
    let mut compared_vecdeque = VecDeque::new();

    for (i, gen) in generations.iter().enumerate() {
        stats.values.update(gen);

        if compared_vecdeque.len() < 2 {
            compared_vecdeque.push_back(OxigenStatsGenerationValues {
                fitnesses: gen.clone(),
            });
        } else {
            let mut new_vecdeque = VecDeque::new();
            new_vecdeque.push_back(compared_vecdeque.pop_back().unwrap());
            new_vecdeque.push_back(OxigenStatsGenerationValues {
                fitnesses: gen.clone(),
            });
            compared_vecdeque = new_vecdeque;
        }
        assert_eq!(
            stats.values.last_generations,
            compared_vecdeque,
            "{}",
            format!("Failed in update {}", i + 1)
        );
    }
}

// -------------------------------------------------------------------------- //
// BestProgress functions
// -------------------------------------------------------------------------- //

#[test]
fn test_best_last_progress() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::BestLastProgress.function()(&mut stats.values),
        500.0 - 4.0,
    );
}

#[test]
fn test_best_progress_avg() {
    let mut stats = get_stats();
    let bests: Vec<f64> = vec![8.0, 2.0, 4.0, 8.0, 4.0, -1.0, 4.0, 500.0];
    let progresses = get_progresses(&bests);
    let avg = progresses.iter().sum::<f64>() / progresses.len() as f64;
    assert_eq!(
        OxigenStatsFields::BestProgressAvg.function()(&mut stats.values),
        avg,
    );
}

#[test]
fn test_best_progress_std() {
    let mut stats = get_stats();
    let bests: Vec<f64> = vec![8.0, 2.0, 4.0, 8.0, 4.0, -1.0, 4.0, 500.0];
    let progresses = get_progresses(&bests);
    let avg = progresses.iter().sum::<f64>() / progresses.len() as f64;
    let std = stddev(&progresses, avg);
    assert_eq!(
        OxigenStatsFields::BestProgressStd.function()(&mut stats.values),
        std,
    );
}

#[test]
fn test_best_progress_max() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::BestProgressMax.function()(&mut stats.values),
        496.0,
    );
}

#[test]
fn test_best_progress_min() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::BestProgressMin.function()(&mut stats.values),
        -6.0,
    );
}

#[test]
fn test_best_progress_p10() {
    let mut stats = get_stats();
    let sorted_progresses = [-6.0, -5.0, -4.0, 2.0, 4.0, 5.0, 496.0];
    let perc = percentile(&sorted_progresses, 0.1);
    assert_eq!(
        OxigenStatsFields::BestProgressP10.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_best_progress_p25() {
    let mut stats = get_stats();
    let sorted_progresses = [-6.0, -5.0, -4.0, 2.0, 4.0, 5.0, 496.0];
    let perc = percentile(&sorted_progresses, 0.25);
    assert_eq!(
        OxigenStatsFields::BestProgressP25.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_best_progress_median() {
    let mut stats = get_stats();
    let sorted_progresses = [-6.0, -5.0, -4.0, 2.0, 4.0, 5.0, 496.0];
    let perc = percentile(&sorted_progresses, 0.5);
    assert_eq!(
        OxigenStatsFields::BestProgressMedian.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_best_progress_p75() {
    let mut stats = get_stats();
    let sorted_progresses = [-6.0, -5.0, -4.0, 2.0, 4.0, 5.0, 496.0];
    let perc = percentile(&sorted_progresses, 0.75);
    assert_eq!(
        OxigenStatsFields::BestProgressP75.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_best_progress_p90() {
    let mut stats = get_stats();
    let sorted_progresses = [-6.0, -5.0, -4.0, 2.0, 4.0, 5.0, 496.0];
    let perc = percentile(&sorted_progresses, 0.9);
    assert_eq!(
        OxigenStatsFields::BestProgressP90.function()(&mut stats.values),
        perc,
    );
}

// -------------------------------------------------------------------------- //
// AvgProgress functions
// -------------------------------------------------------------------------- //

#[test]
fn test_avg_last_progress() {
    let mut stats = get_stats();
    let avg: Vec<f64> = vec![3.75, 2.0, 4.0, 2.5, 1.0, -3.75, -0.75, 128.5];
    let progresses = get_progresses(&avg);
    assert_eq!(
        OxigenStatsFields::AvgLastProgress.function()(&mut stats.values),
        *progresses.last().unwrap(),
    );
}

#[test]
fn test_avg_progress_avg() {
    let mut stats = get_stats();
    let avg: Vec<f64> = vec![3.75, 2.0, 4.0, 2.5, 1.0, -3.75, -0.75, 128.5];
    let progresses = get_progresses(&avg);
    let avg = progresses.iter().sum::<f64>() / progresses.len() as f64;
    assert_eq!(
        OxigenStatsFields::AvgProgressAvg.function()(&mut stats.values),
        avg,
    );
}

#[test]
fn test_avg_progress_std() {
    let mut stats = get_stats();
    let avg: Vec<f64> = vec![3.75, 2.0, 4.0, 2.5, 1.0, -3.75, -0.75, 128.5];
    let progresses = get_progresses(&avg);
    let avg = progresses.iter().sum::<f64>() / progresses.len() as f64;
    let std = stddev(&progresses, avg);
    assert_eq!(
        OxigenStatsFields::AvgProgressStd.function()(&mut stats.values),
        std,
    );
}

#[test]
fn test_avg_progress_max() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::AvgProgressMax.function()(&mut stats.values),
        129.25,
    );
}

#[test]
fn test_avg_progress_min() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::AvgProgressMin.function()(&mut stats.values),
        -4.75,
    );
}

#[test]
fn test_avg_progress_p10() {
    let mut stats = get_stats();
    let sorted_progresses = [-4.75, -1.75, -1.5, -1.5, 2.0, 3.0, 129.25];
    let perc = percentile(&sorted_progresses, 0.1);
    assert_eq!(
        OxigenStatsFields::AvgProgressP10.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_avg_progress_p25() {
    let mut stats = get_stats();
    let sorted_progresses = [-4.75, -1.75, -1.5, -1.5, 2.0, 3.0, 129.25];
    let perc = percentile(&sorted_progresses, 0.25);
    assert_eq!(
        OxigenStatsFields::AvgProgressP25.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_avg_progress_median() {
    let mut stats = get_stats();
    let sorted_progresses = [-4.75, -1.75, -1.5, -1.5, 2.0, 3.0, 129.25];
    let perc = percentile(&sorted_progresses, 0.5);
    assert_eq!(
        OxigenStatsFields::AvgProgressMedian.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_avg_progress_p75() {
    let mut stats = get_stats();
    let sorted_progresses = [-4.75, -1.75, -1.5, -1.5, 2.0, 3.0, 129.25];
    let perc = percentile(&sorted_progresses, 0.75);
    assert_eq!(
        OxigenStatsFields::AvgProgressP75.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_avg_progress_p90() {
    let mut stats = get_stats();
    let sorted_progresses = [-4.75, -1.75, -1.5, -1.5, 2.0, 3.0, 129.25];
    let perc = percentile(&sorted_progresses, 0.9);
    assert_eq!(
        OxigenStatsFields::AvgProgressP90.function()(&mut stats.values),
        perc,
    );
}

// -------------------------------------------------------------------------- //
// AvgFitness functions
// -------------------------------------------------------------------------- //

#[test]
fn test_fitness_avg() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::FitnessAvg.function()(&mut stats.values),
        128.5,
    );
}

#[test]
fn test_fitness_std() {
    let mut stats = get_stats();
    let last: Vec<f64> = vec![500.0, 2.0, 4.0, 8.0];
    let avg: f64 = 128.5;
    let std = stddev(&last, avg);
    assert_eq!(
        OxigenStatsFields::FitnessStd.function()(&mut stats.values),
        std,
    );
}

#[test]
fn test_fitness_max() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::FitnessMax.function()(&mut stats.values),
        500.0,
    );
}

#[test]
fn test_fitness_min() {
    let mut stats = get_stats();
    assert_eq!(
        OxigenStatsFields::FitnessMin.function()(&mut stats.values),
        2.0,
    );
}

#[test]
fn test_fitness_p10() {
    let mut stats = get_stats();
    let sorted_fitnesses = [2.0, 4.0, 8.0, 500.0];
    let perc = percentile(&sorted_fitnesses, 0.1);
    assert_eq!(
        OxigenStatsFields::FitnessP10.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_fitness_p25() {
    let mut stats = get_stats();
    let sorted_fitnesses = [2.0, 4.0, 8.0, 500.0];
    let perc = percentile(&sorted_fitnesses, 0.25);
    assert_eq!(
        OxigenStatsFields::FitnessP25.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_fitness_median() {
    let mut stats = get_stats();
    let sorted_fitnesses = [2.0, 4.0, 8.0, 500.0];
    let perc = percentile(&sorted_fitnesses, 0.5);
    assert_eq!(
        OxigenStatsFields::FitnessMedian.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_fitness_p75() {
    let mut stats = get_stats();
    let sorted_fitnesses = [2.0, 4.0, 8.0, 500.0];
    let perc = percentile(&sorted_fitnesses, 0.75);
    assert_eq!(
        OxigenStatsFields::FitnessP75.function()(&mut stats.values),
        perc,
    );
}

#[test]
fn test_fitness_p90() {
    let mut stats = get_stats();
    let sorted_fitnesses = [2.0, 4.0, 8.0, 500.0];
    let perc = percentile(&sorted_fitnesses, 0.90);
    assert_eq!(
        OxigenStatsFields::FitnessP90.function()(&mut stats.values),
        perc,
    );
}
