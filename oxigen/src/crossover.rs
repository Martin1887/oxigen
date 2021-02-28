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
    /// Multi point Crossover.
    MultiCrossPoint,
    /// Uniform Crossover.
    UniformCross,
}

impl<T, G: Genotype<T>> Crossover<T, G> for CrossoverFunctions {
    #[allow(clippy::comparison_chain)]
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G) {
        match self {
            SingleCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                if ind_size == 0 {
                    panic!("The size of the smallest individual is 0");
                } else if ind_size == 1 {
                    return crosspoint_cross_single_genes(&ind1, &ind2);
                }
                let cross_point = SmallRng::from_entropy().sample(Uniform::from(1..ind_size));

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
            MultiCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                if ind_size == 0 {
                    panic!("The size of the smallest individual is 0");
                } else if ind_size == 1 {
                    return crosspoint_cross_single_genes(&ind1, &ind2);
                }
                let mut cross_points = Vec::new();
                let mut point_maximum = ind_size / 2;
                if point_maximum <= 2 {
                    point_maximum = 3.min(ind_size);
                }
                let mut i = SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                while i < ind_size {
                    cross_points.push(i);
                    i += SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                }
                cross_points.push(ind_size);

                (
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen2;
                                    } else {
                                        return gen1;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen1
                        })
                        .collect(),
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen1;
                                    } else {
                                        return gen2;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen2
                        })
                        .collect(),
                )
            }
            UniformCross => {
                let mut child1: G = ind1
                    .clone()
                    .into_iter()
                    .zip(ind2.clone().into_iter())
                    .enumerate()
                    .map(|(i, (gen1, gen2))| if i % 2 == 0 { gen1 } else { gen2 })
                    .collect();
                let mut child2: G = ind1
                    .clone()
                    .into_iter()
                    .zip(ind2.clone().into_iter())
                    .enumerate()
                    .map(|(i, (gen1, gen2))| if i % 2 != 0 { gen1 } else { gen2 })
                    .collect();

                // If one individual is shorter than other the zipped iterator ends in the smallest
                // one, so the rest of the original individual should be appended to the child
                let len1 = ind1.iter().len();
                let len2 = ind2.iter().len();
                if len1 > len2 {
                    child1 = child1
                        .clone()
                        .into_iter()
                        .chain(
                            ind1.clone()
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _el)| *i >= len2)
                                .map(|(_i, el)| el),
                        )
                        .collect();
                } else if len1 < len2 {
                    child2 = child2
                        .clone()
                        .into_iter()
                        .chain(
                            ind2.clone()
                                .into_iter()
                                .enumerate()
                                .filter(|(i, _el)| *i >= len1)
                                .map(|(_i, el)| el),
                        )
                        .collect();
                }

                (child1, child2)
            }
        }
    }
}

/// Crosspoint crossover when one or both individuals have length 1
fn crosspoint_cross_single_genes<T, G: Genotype<T>>(ind1: &G, ind2: &G) -> (G, G) {
    let len1 = ind1.iter().len();
    let len2 = ind2.iter().len();

    if len1 > 1 {
        // interchange ind2 gene with a random gene in ind1
        interchange_gene(&ind2, &ind1, len1)
    } else if len2 > 1 {
        // interchange ind2 gene with a random gene in ind1
        interchange_gene(&ind1, &ind2, len2)
    } else {
        // children equal to parents, since both have length 1
        (ind2.clone(), ind1.clone())
    }
}

/// Interchange len1_ind gene into a random position of the another individual
fn interchange_gene<T, G: Genotype<T>>(len1_ind: &G, bigger_ind: &G, bigger_len: usize) -> (G, G) {
    let interchanged = SmallRng::from_entropy().sample(Uniform::from(0..bigger_len));
    // return the interchanged gene of bigger_ind as child1 and the bigger_ind
    // with the len1_ind gene in the interchanged position as child2
    (
        bigger_ind
            .clone()
            .into_iter()
            .enumerate()
            .filter(|(i, _gen)| *i == interchanged)
            .map(|(_i, gen)| gen)
            .collect(),
        bigger_ind
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, gen)| {
                if i == interchanged {
                    len1_ind.clone().into_iter().next().unwrap()
                } else {
                    gen
                }
            })
            .collect(),
    )
}
