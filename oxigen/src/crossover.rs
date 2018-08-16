//! This module contains the Crossover trait and the provided crossover functions.

use genotype::Genotype;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::cmp::min;
use CrossoverFunctions::*;

/// This trait defines the cross function.
pub trait Crossover<T, G: Genotype<T>>: Send + Sync {
    /// Generates two children combining the two selected individuals.
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G);
}

/// Provided crossover functions.
pub enum CrossoverFunctions {
    /// Single point Crossover.
    SingleCrossPoint,
}

impl<T, G: Genotype<T>> Crossover<T, G> for CrossoverFunctions {
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G) {
        match self {
            SingleCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                let cross_point = SmallRng::from_entropy().sample(Uniform::from(0..ind_size));

                (
                    ind1.clone()
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _gen)| *i < cross_point)
                        .chain(
                            ind2.clone()
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _gen)| *i >= cross_point),
                        )
                        .map(|(_i, gen)| gen)
                        .collect(),
                    ind2.clone()
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _gen)| *i < cross_point)
                        .chain(
                            ind1.clone()
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _gen)| *i >= cross_point),
                        )
                        .map(|(_i, gen)| gen)
                        .collect(),
                )
            }
        }
    }
}
