//! This module contains the definition of genotypes.

use rand::prelude::SmallRng;
use std::fmt::Display;
use std::slice::Iter;
use std::vec::IntoIter;

/// This trait defines an individual of a population in the genetic algorithm.
/// It defines the fitness and mutation functions and the type of the
/// individual representation.
pub trait Genotype<T: PartialEq>: Display + Clone + Send + Sync {
    /// The type that represents the problem size of the genotype. For example,
    /// in the N Queens problem the size of the `ProblemSize` is a numeric type
    /// (the number of queens).
    type ProblemSize: Default + Send + Sync;

    /// The type that is used for hashing, by default the self struct.
    #[cfg(feature = "global_cache")]
    type GenotypeHash: Eq + std::hash::Hash + Send + Sync;

    /// Returns an iterator over the genes of the individual.
    fn iter(&self) -> Iter<T>;

    /// Consumes the individual into an iterator over its genes.
    fn into_iter(self) -> IntoIter<T>;

    /// Set the genes of the individual from an iterator.
    fn from_iter<I: Iterator<Item = T>>(&mut self, I);

    /// Randomly initializes an individual.
    fn generate(size: &Self::ProblemSize) -> Self;

    /// Returns a fitness value for an individual.
    fn fitness(&self) -> f64;

    /// Defines the manner in which an individual is mutated when
    /// an element of the individual is selected to mutate.
    fn mutate(&mut self, rgen: &mut SmallRng, index: usize);

    /// Defines if an individual is a valid solution to the problem.
    fn is_solution(&self, fitness: f64) -> bool;

    /// Fix the individual to satisfy the problem restrictions. The default
    /// implementation is to remain the individual unmodified always.
    ///
    /// # Returns
    ///
    /// true if the individual has changed and false otherwise. If this function
    /// returns true the fitness is recomputed.
    fn fix(&mut self) -> bool {
        false
    }

    /// A function to define how different is the individual from another one.
    /// The default implementation sums the number of different genes and divides
    /// it by the total number of genes. If one individual is shorter than another,
    /// to this value the difference in length is added, and the sum is divided by
    /// the length of the longer individual. This
    /// function is used to determine if solutions are different and in some
    /// survival pressure functions.
    fn distance(&self, other: &Self) -> f64 {
        if self.iter().len() == other.iter().len() {
            self.iter()
                .zip(other.iter())
                .filter(|(gen, gen_other)| gen != gen_other)
                .count() as f64
                / self.iter().len() as f64
        } else {
            let max_length = if other.iter().len() > self.iter().len() {
                other.iter().len()
            } else {
                self.iter().len()
            };
            (((other.iter().len() - self.iter().len()) as f64).abs()
                + self
                    .iter()
                    .zip(other.iter())
                    .filter(|(gen, gen_other)| gen != gen_other)
                    .count() as f64)
                / max_length as f64
        }
    }

    /// Function to quickly hash the individual for global cache.
    /// The default implementation is the `to_string()` function but
    /// another faster function can be implemented if the `Display`
    /// implementation is slow.
    #[cfg(feature = "global_cache")]
    fn hash(&self) -> Self::GenotypeHash;
}
