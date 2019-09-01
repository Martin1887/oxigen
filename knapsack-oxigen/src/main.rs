extern crate clap;
extern crate oxigen;
extern crate rand;

use clap::{App, Arg};
use oxigen::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::fmt::Debug;
use std::fmt::Display;
use std::fs::File;

#[derive(Clone, Default)]
struct Item {
    weight: f64,
    value: f64,
}
impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "Item(Weight: {}, value: {})", self.weight, self.value)
    }
}

#[derive(Clone)]
struct Knapsack {
    capacity: f64,
    items: Vec<bool>,
    available_items: Vec<Item>,
}
impl Display for Knapsack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mut s = format!("Capacity: {}, Items: ", self.capacity);
        let mut first = true;
        for i in self.items.iter() {
            if first {
                first = false;
                s.push_str(&format!("[{}", i));
            } else {
                s.push_str(&format!(", {}", i));
            }
        }
        write!(f, "{}]", s)
    }
}
impl Debug for Knapsack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self)
    }
}

impl Genotype<bool> for Knapsack {
    type ProblemSize = (f64, Vec<Item>);

    fn iter(&self) -> std::slice::Iter<bool> {
        self.items.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<bool> {
        self.items.into_iter()
    }
    fn from_iter<I: Iterator<Item = bool>>(&mut self, genes: I) {
        self.items = genes.collect();
    }

    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual = Knapsack {
            capacity: size.0,
            items: Vec::with_capacity(size.1.len()),
            available_items: size.1.clone(),
        };
        let mut rgen = SmallRng::from_entropy();

        for _i in 0..size.1.len() {
            individual.items.push(rgen.gen_bool(0.5));
        }
        individual
    }

    // This function returns the mximum punctuaction possible (n-1, since in the
    // worst case n-1 queens must be moved to get a solution) minus the number of
    // queens that collide with others
    fn fitness(&self) -> f64 {
        self.items
            .iter()
            .enumerate()
            .map(|(i, ind)| {
                if *ind {
                    self.available_items[i].value
                } else {
                    0.0f64
                }
            })
            .sum()
    }

    fn mutate(&mut self, _rgen: &mut SmallRng, index: usize) {
        self.items[index] = !self.items[index];
    }

    fn is_solution(&self, fitness: f64) -> bool {
        (fitness - self.available_items.iter().map(|i| i.value).sum::<f64>()).abs() < 0.00001
    }

    fn fix(&mut self) -> bool {
        let mut total_weight: f64 = self
            .items
            .iter()
            .enumerate()
            .map(|(i, ind)| {
                if *ind {
                    self.available_items[i].weight
                } else {
                    0.0f64
                }
            })
            .sum();
        let mut changed = false;
        while total_weight > self.capacity {
            changed = true;
            let min_value = self
                .available_items
                .iter()
                .enumerate()
                .map(|(i, ind)| (i, ind.value))
                .filter(|(i, _ind)| self.items[*i])
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
            if let Some(min_value) = min_value {
                self.items[min_value.0] = false;
                total_weight -= self.available_items[min_value.0].weight;
            }
        }
        changed
    }
}

fn main() {
    let matches = App::new("Knapsack problem resolution using oxigen")
        .version("1.0")
        .author("Martin1887")
        .arg(
            Arg::with_name("capacity")
                .help("Sets the knapsack capacity in kg")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name("n_items")
                .help("The number of items to be randomly generated")
                .index(2)
                .conflicts_with("items"),
        )
        .get_matches();

    let progress_log = File::create("progress.csv").expect("Error creating progress log file");
    let population_log =
        File::create("population.txt").expect("Error creating population log file");
    let population_size = 64 as usize;
    let capacity = matches.value_of("capacity").unwrap().parse().unwrap();
    let mut items = Vec::new();
    if matches.is_present("n_items") {
        let mut rgen = SmallRng::from_entropy();
        let n_items = matches.value_of("n_items").unwrap().parse().unwrap();
        for _i in 0..n_items {
            items.push(Item {
                weight: rgen.sample(Uniform::from(0.0..10.0)),
                value: rgen.sample(Uniform::from(0.0..87.0)),
            });
        }
    } else {
        items = vec![
            Item {
                weight: 4.0,
                value: 8.26,
            },
            Item {
                weight: 87.15,
                value: 15.48,
            },
            Item {
                weight: 2.15,
                value: 40.87,
            },
            Item {
                weight: 15.21,
                value: 4.87,
            },
        ];
    }
    let (solutions, generation, progress, _population) = GeneticExecution::<bool, Knapsack>::new()
        .population_size(population_size)
        .genotype_size((capacity, items))
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: 15.0,
            bound: 0.005,
            coefficient: -0.005,
        })))
        .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: 4.0,
            bound: 2.0,
            coefficient: -0.0005,
        })))
        .select_function(Box::new(SelectionFunctions::Cup))
        .crossover_function(Box::new(CrossoverFunctions::UniformCross))
        .age_function(Box::new(AgeFunctions::Quadratic(
            AgeThreshold(2),
            AgeSlope(4_f64),
        )))
        .population_refitness_function(Box::new(PopulationRefitnessFunctions::Niches(
            NichesAlpha(1.0),
            Box::new(NichesBetaRates::Constant(1.0)),
            NichesSigma(0.2),
        )))
        .survival_pressure_function(Box::new(
            SurvivalPressureFunctions::DeterministicOverpopulation,
        ))
        .progress_log(100, progress_log)
        .population_log(500, population_log)
        .stop_criterion(Box::new(StopCriteria::GenerationAndProgress(1000, 0.0001)))
        .run();

    println!(
        "Finished in the generation {} with a progress of {}",
        generation, progress
    );
    for sol in &solutions {
        println!("{}", sol);
    }
}
