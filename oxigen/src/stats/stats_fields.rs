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
