//! This module contains the definition of the PopulationRefitness trait and the provided
//! population refitness functions

use genotype::Genotype;
use niches_beta_rate::*;
use rayon::prelude::*;
use std::cmp::PartialEq;
use IndWithFitness;
use PopulationRefitnessFunctions::*;

/// This trait defines the population_refitness function that permits to modify the
/// individuals fitness comparing them to the others individuals in the population.
/// This function is called when all individuals fitness have been computed and just
/// before the survival pressure kill.
pub trait PopulationRefitness<T: PartialEq + Send + Sync, G: Genotype<T>>: Send + Sync {
    /// Modify the individual fitness comparing it with the other individuals in the
    /// population. Called just before survival pressure kill.
    ///
    /// # Parameters:
    /// - `individual_index`: The individual index inside the population.
    /// - `population`: The full population. It includes the individual that is being
    /// evaluated (`individual_index`) that is probably wanted to be excluded of the
    /// comparison with population individuals. The `Option<Fitness>` is always
    /// `Some(Fitness)` when this function is called.
    /// - `generation`: The current generation.
    /// - `progress`: The progress in the last iterations.
    /// - `n_solutions`: The number of found solutions in the last generation.
    ///
    /// # Returns
    /// The new fitness of the individual.
    fn population_refitness(
        &self,
        individual_index: usize,
        population: &[IndWithFitness<T, G>],
        generation: u64,
        progress: f64,
        n_solutions: usize,
    ) -> f64;
}

/// Importance given to distance among individuals. It is usually a number between 0 and 1.
#[derive(Debug)]
pub struct NichesAlpha(pub f64);

/// Threshold used in niches to divide distance by it if the distance is lower than sigma.
#[derive(Debug)]
pub struct NichesSigma(pub f64);

/// Provided PopulationRefitness functions.
pub enum PopulationRefitnessFunctions {
    /// Remains the fitness unmodified.
    None,
    /// Scales the fitness using the formula `f' = f^beta / m`, where `m` is
    /// the sum of (1 - distance / sigma) for each individual in the population.
    Niches(NichesAlpha, Box<NichesBetaRates>, NichesSigma),
}

impl<T: PartialEq + Send + Sync, G: Genotype<T>> PopulationRefitness<T, G>
    for PopulationRefitnessFunctions
{
    fn population_refitness(
        &self,
        individual_index: usize,
        population: &[IndWithFitness<T, G>],
        generation: u64,
        progress: f64,
        n_solutions: usize,
    ) -> f64 {
        match self {
            None => {
                population[individual_index]
                    .fitness
                    .unwrap()
                    .original_fitness
            }
            Niches(alfa, beta, sigma) => {
                let current_ind = &population[individual_index].ind;
                let current_fitness = &population[individual_index].fitness;
                let mut current_fitness = current_fitness.unwrap().original_fitness;
                /*
                let avg_sim: f64 = population
                    .par_iter()
                    .enumerate()
                    .filter(|(i, _ind)| *i != individual_index)
                    .map(|(_i, ind)| current_ind.similarity(&ind.0))
                    .sum::<f64>()
                    / (population.len() as f64 - 1.0);
                */
                let m = population
                    .par_iter()
                    .enumerate()
                    .filter(|(i, _ind)| *i != individual_index)
                    .map(|(_i, ind)| current_ind.distance(&ind.ind))
                    .map(|d| {
                        if d >= sigma.0 {
                            0.0
                        } else {
                            (1.0 - (d / sigma.0)).powf(alfa.0)
                        }
                    })
                    .sum::<f64>();
                if current_fitness > 0.0 {
                    current_fitness =
                        current_fitness.powf(beta.rate(generation, progress, n_solutions));
                    if m > 0.0 {
                        current_fitness /= m;
                    }
                    current_fitness
                } else {
                    current_fitness
                }
            }
        }
    }
}
