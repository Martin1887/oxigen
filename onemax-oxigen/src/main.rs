extern crate oxigen;
extern crate rand;

use oxigen::prelude::*;

use rand::distributions::Standard;
use rand::prelude::*;

use std::fmt::Display;
use std::iter::FromIterator;
#[derive(Clone)]
struct OneMax(Vec<bool>);

impl Display for OneMax {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", self.0)
    }
}

impl FromIterator<bool> for OneMax {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = bool>,
    {
        OneMax {
            0: iter.into_iter().collect(),
        }
    }
}

impl Genotype<bool> for OneMax {
    type ProblemSize = usize;

    fn iter(&self) -> std::slice::Iter<bool> {
        self.0.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<bool> {
        self.0.into_iter()
    }

    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual = Vec::with_capacity(*size as usize);
        let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.push(rgen.sample(Standard));
        }
        OneMax(individual)
    }

    fn fitness(&self) -> f64 {
        (self.0.iter().filter(|el| **el).count()) as f64
    }

    fn mutate(&mut self, _rgen: &mut SmallRng, index: usize) {
        self.0[index] = !self.0[index];
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness as usize == self.0.len()
    }
}

fn main() {
    let problem_size: usize = std::env::args()
        .nth(1)
        .expect("Enter a number bigger than 1")
        .parse()
        .expect("Enter a number bigger than 1");
    let population_size = problem_size * 8;
    let log2 = (f64::from(problem_size as u32) * 4_f64).log2().ceil();
    let (solutions, generation, _progress, _population) = GeneticExecution::<bool, OneMax>::new()
        .population_size(population_size)
        .genotype_size(problem_size)
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: f64::from(problem_size as u32) / (8_f64 + 2_f64 * log2) / 100_f64,
            bound: 0.005,
            coefficient: -0.0002,
        }))).selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: log2 - 2_f64,
            bound: log2 / 1.5,
            coefficient: -0.0005,
        }))).select_function(Box::new(SelectionFunctions::Cup))
        .run();

    println!("Finished in the generation {}", generation);
    for sol in &solutions {
        println!("{}", sol);
    }
}
