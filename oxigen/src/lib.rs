//! This crate provides functions for parallel genetic algorithm execution.

// #![feature(test)]

extern crate rand;
extern crate rayon;

// mod benchmarks;
#[cfg(test)]
mod tests;

pub mod age;
pub mod crossover;
pub mod genotype;
pub mod mutation_rate;
pub mod niches_beta_rate;
pub mod population_refitness;
pub mod prelude;
pub mod selection;
pub mod selection_rate;
pub mod slope_params;
pub mod stats;
pub mod stop_criteria;
pub mod survival_pressure;

pub use age::*;
pub use crossover::*;
pub use genotype::Genotype;
pub use mutation_rate::*;
pub use niches_beta_rate::*;
pub use population_refitness::*;
pub use selection::*;
pub use selection_rate::*;
pub use slope_params::*;
pub use stats::stats_aux::*;
pub use stats::stats_fields::*;
pub use stats::stats_impl::*;
use stats::stats_schema::*;
pub use stats::*;
use std::fmt::Display;
use std::marker::PhantomData;
pub use stop_criteria::*;
pub use survival_pressure::*;

use rand::distributions::{Standard, Uniform};
use rand::prelude::*;
use rayon::prelude::*;
#[cfg(feature = "global_cache")]
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::sync::mpsc::channel;

const POPULATION_SEPARATOR: &[u8] = b"\n\n\n\n---------------------------------\n\n\n\n";
const POPULATION_ERR_MSG: &str = "Error writing on population log file";
const PROGRESS_ERR_MSG: &str = "Error writing in progress log file";

/// Struct that defines the fitness of each individual and the related information.
#[derive(Copy, Clone, Debug)]
pub struct Fitness {
    /// Age of the individual.
    age: u64,
    /// Actual fitness.
    fitness: f64,
    /// Original fitness of the individual before being unfitnessed by age.
    original_fitness: f64,
    /// Age effect over the original fitness (usually negative).
    age_effect: f64,
    /// Refitness effect over the original fitness (usually negative).
    refitness_effect: f64,
}

/// Struct that defines a pair of individual-fitness
#[derive(Debug)]
pub struct IndWithFitness<T: PartialEq + Send + Sync, Ind: Genotype<T>> {
    /// Individual
    pub ind: Ind,
    /// Fitness (can be not computed yet)
    pub fitness: Option<Fitness>,
    _phantom: PhantomData<T>,
}

impl<T: PartialEq + Send + Sync, Ind: Genotype<T>> IndWithFitness<T, Ind> {
    pub fn new(ind: Ind, fitness: Option<Fitness>) -> IndWithFitness<T, Ind> {
        IndWithFitness {
            ind,
            fitness,
            _phantom: PhantomData,
        }
    }
}

impl<T: PartialEq + Send + Sync, Ind: Genotype<T>> Display for IndWithFitness<T, Ind> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "ind: {}, fitness: {:?}", self.ind, self.fitness)
    }
}

/// Struct that defines a genetic algorithm execution.
pub struct GeneticExecution<T: PartialEq + Send + Sync, Ind: Genotype<T>> {
    /// The number of individuals in the population.
    population_size: usize,
    /// Population with all individuals and their respective fitnesses.
    population: Vec<IndWithFitness<T, Ind>>,
    /// Environment associated with the problem.
    environment: Ind::Environment,
    /// The mutation rate variation along iterations and progress.
    mutation_rate: Box<dyn MutationRate>,
    /// The number of stages in the cup whose individuals are selected to crossover.
    selection_rate: Box<dyn SelectionRate>,
    /// The selection function.
    selection: Box<dyn Selection>,
    /// The age fitness decrease function.
    age: Box<dyn Age>,
    /// The crossover function.
    crossover: Box<dyn Crossover<T, Ind>>,
    /// The function used to recompute the fitness using the full population just
    /// before survival pressure.
    population_refitness: Box<dyn PopulationRefitness<T, Ind>>,
    /// The function used to replace individuals in the population.
    survival_pressure: Box<dyn SurvivalPressure<T, Ind>>,
    /// The stop criterion to finish execution.
    stop_criterion: Box<dyn StopCriterion>,
    /// Cache fitness value of individuals or compute it in each iteration.
    cache_fitness: bool,
    /// Global cache for all evaluated individuals, i.e., if a new individual is
    /// equal to a previous individual its fitness is not recomputed. Only valid
    /// if cache_fitness is true.
    global_cache: bool,
    /// HashMap used for the global cache.
    #[cfg(feature = "global_cache")]
    cache_map: HashMap<Ind::GenotypeHash, f64>,
    /// Stats log, writes statistics of the population every certain number of generations.
    stats_log: (u64, Option<File>),
    /// Number of last generations to be taken into account in stats (64 by default).
    stats_generations: usize,
    /// Delimiter of the statistics CSV file ("\t" by default).
    stats_delimiter: String,
    /// `OxigenStats` object with the statistics options. By default all fields are enabled.
    stats: OxigenStats,
    /// Population log, writes the population and fitnesses every certain number of generations.
    population_log: (u64, Option<File>),
}

impl<T: PartialEq + Send + Sync, Ind: Genotype<T>> Default for GeneticExecution<T, Ind> {
    fn default() -> Self {
        GeneticExecution {
            population_size: 64,
            population: Vec::new(),
            environment: Ind::Environment::default(),
            mutation_rate: Box::new(MutationRates::Constant(0.1)),
            selection_rate: Box::new(SelectionRates::Constant(2)),
            selection: Box::new(SelectionFunctions::Cup),
            age: Box::new(AgeFunctions::None),
            crossover: Box::new(CrossoverFunctions::SingleCrossPoint),
            population_refitness: Box::new(PopulationRefitnessFunctions::None),
            survival_pressure: Box::new(SurvivalPressureFunctions::Worst),
            stop_criterion: Box::new(StopCriteria::SolutionFound),
            cache_fitness: true,
            global_cache: cfg!(feature = "global_cache"),
            #[cfg(feature = "global_cache")]
            cache_map: HashMap::new(),
            stats_log: (0, None),
            stats_generations: 64,
            stats_delimiter: String::from("\t"),
            stats: OxigenStats::new(64, "\t"),
            population_log: (0, None),
        }
    }
}

impl<T: PartialEq + Send + Sync, Ind: Genotype<T>> GeneticExecution<T, Ind> {
    /// Create a new default genetic algorithm execution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the population size.
    pub fn population_size(mut self, new_pop_size: usize) -> Self {
        self.population_size = new_pop_size;
        self
    }

    /// Set the initial population individuals. If lower individuals
    /// than population_size are received, the rest of population will be
    /// generated randomly.
    pub fn population(mut self, new_pop: Vec<IndWithFitness<T, Ind>>) -> Self {
        self.population = new_pop;
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: Ind::Environment) -> Self {
        self.environment = env;
        self
    }

    /// Set the mutation rate.
    pub fn mutation_rate(mut self, new_mut: Box<dyn MutationRate>) -> Self {
        self.mutation_rate = new_mut;
        self
    }

    /// Set the number of tournament stages whose individuals are selected for crossover.
    pub fn selection_rate(mut self, new_sel_rate: Box<dyn SelectionRate>) -> Self {
        self.selection_rate = new_sel_rate;
        self
    }

    /// Set the selection function of the genetic algorithm.
    pub fn select_function(mut self, new_sel: Box<dyn Selection>) -> Self {
        self.selection = new_sel;
        self
    }

    /// Set the age function of the genetic algorithm.
    pub fn age_function(mut self, new_age: Box<dyn Age>) -> Self {
        self.age = new_age;
        self
    }

    /// Set the crossover function of the genetic algorithm.
    pub fn crossover_function(mut self, new_cross: Box<dyn Crossover<T, Ind>>) -> Self {
        self.crossover = new_cross;
        self
    }

    /// Set the population refitness function of the genetic algorithm.
    pub fn population_refitness_function(
        mut self,
        new_refit: Box<dyn PopulationRefitness<T, Ind>>,
    ) -> Self {
        self.population_refitness = new_refit;
        self
    }

    /// Set the survival pressure function of the genetic algorithm.
    pub fn survival_pressure_function(
        mut self,
        new_surv: Box<dyn SurvivalPressure<T, Ind>>,
    ) -> Self {
        self.survival_pressure = new_surv;
        self
    }

    /// Set the stop criterion of the genetic algorithm.
    pub fn stop_criterion(mut self, new_crit: Box<dyn StopCriterion>) -> Self {
        self.stop_criterion = new_crit;
        self
    }

    /// Set the cache fitness flag.
    pub fn cache_fitness(mut self, new_cache: bool) -> Self {
        self.cache_fitness = new_cache;
        self
    }

    /// Set the global cache flag.
    ///
    /// # Panics: when trying to put it true without `global_cache` feature enabled.
    pub fn global_cache(mut self, new_global_cache: bool) -> Self {
        if cfg!(feature = "global_cache") {
            self.global_cache = new_global_cache;
        } else if new_global_cache {
            panic!("global_cache feature must been enabled to enable global_cache");
        } else {
            self.global_cache = false;
        }
        self
    }

    /// Set the stats log.
    pub fn stats_log(mut self, generations: u64, log_file: File) -> Self {
        self.stats_log = (generations, Some(log_file));
        self
    }

    /// Set the number of generations taken into account for the statistics.
    pub fn stats_generations(mut self, generations: usize) -> Self {
        self.stats_generations = generations;
        self.stats = OxigenStats::new(self.stats_generations, &self.stats_delimiter);
        self
    }

    /// Add a custom statistics field to print in the statistics log.
    pub fn stats_add_custom_field(
        mut self,
        name: &str,
        field: Box<dyn OxigenStatsFieldFunction>,
    ) -> Self {
        self.stats.add_field(name, field);
        self
    }

    /// Enable a statistics field to print in the statistics log.
    pub fn stats_enable_field(mut self, field: OxigenStatsFields) -> Self {
        self.stats.enable_field(field);
        self
    }

    /// Disable a statistics field (not printed in the statistics log).
    pub fn stats_disable_field(mut self, field: OxigenStatsFields) -> Self {
        self.stats.disable_field(field);
        self
    }

    /// Set the delimiter for the stats CSV file.
    pub fn stats_delimiter(mut self, delimiter: String) -> Self {
        self.stats_delimiter = delimiter;
        self.stats.set_delimiter(&self.stats_delimiter);
        self
    }

    /// Set the population log.
    pub fn population_log(mut self, generations: u64, log_file: File) -> Self {
        self.population_log = (generations, Some(log_file));
        self
    }

    /// Run the genetic algorithm execution until the `stop_criterion` is satisfied.
    ///
    /// # Returns
    ///
    /// - A vector with the individuals of the population that are solution of the problem.
    /// - The number of generations run.
    /// - The statistics in the last generations.
    /// - The entire population in the last generation (useful for resuming the execution).
    pub fn run(
        mut self,
    ) -> (
        Vec<Ind>,
        u64,
        OxigenStatsValues,
        Vec<IndWithFitness<T, Ind>>,
    ) {
        // Initialize randomly the population
        while self.population.len() < self.population_size {
            self.population
                .push(IndWithFitness::new(Ind::generate(&self.environment), None));
        }
        self.fix();
        self.compute_fitnesses(true);

        if self.stats_log.0 > 0 {
            self.print_progress_header();
        }

        let (generation, solutions) = self.run_loop();

        (solutions, generation, self.stats.values, self.population)
    }

    fn run_loop(&mut self) -> (u64, Vec<Ind>) {
        let mut generation: u64 = 0;
        // A HashSet is not used to not force to implement Hash in Genotype
        let mut solutions: Vec<Ind> = Vec::new();
        let mut mutation_rate;
        let mut selection_rate;

        let mut current_fitnesses = self.get_fitnesses();

        while !self.stop_criterion.stop(
            generation,
            &mut self.stats.values,
            solutions.len(),
            &current_fitnesses,
        ) {
            generation += 1;

            mutation_rate = self.mutation_rate.rate(
                generation,
                &mut self.stats.values,
                solutions.len(),
                &current_fitnesses,
            );
            selection_rate = self.selection_rate.rate(
                generation,
                &mut self.stats.values,
                solutions.len(),
                &current_fitnesses,
            );

            self.compute_fitnesses(true);
            self.refitness(generation, solutions.len());
            current_fitnesses = self.get_fitnesses();
            let selected = self.selection.select(&current_fitnesses, selection_rate);
            let parents_children = self.cross(&selected);
            // fitnesses must be computed for the new individuals to get solutions
            self.compute_fitnesses(false);
            self.get_solutions(&mut solutions);
            self.mutate(mutation_rate);
            self.fix();
            self.compute_fitnesses(false);
            self.refitness(generation, solutions.len());
            self.age_unfitness();
            // get solutions before kill
            self.get_solutions(&mut solutions);
            self.survival_pressure_kill(&parents_children);

            current_fitnesses = self.get_fitnesses();
            self.stats.values.update(&current_fitnesses);

            if self.stats_log.0 > 0 && generation % self.stats_log.0 == 0 {
                self.print_progress(generation, solutions.len());
            }
            if self.population_log.0 > 0 && generation % self.population_log.0 == 0 {
                self.print_population(generation);
            }

            self.update_age();
        }

        (generation, solutions)
    }

    fn print_population(&mut self, generation: u64) {
        if let Some(ref mut f) = self.population_log.1 {
            f.write_all(format!("Generation {}\n", generation).as_bytes())
                .expect(POPULATION_ERR_MSG);
            for (i, ind) in self.population.iter().enumerate() {
                f.write_all(
                    format!(
                        "Individual: {}; fitness: {}, age: {}, original_fitness: {}\n",
                        i,
                        ind.fitness.unwrap().fitness,
                        ind.fitness.unwrap().age,
                        ind.fitness.unwrap().original_fitness
                    )
                    .as_bytes(),
                )
                .expect(POPULATION_ERR_MSG);
                f.write_all(format!("{}\n\n", ind.ind).as_bytes())
                    .expect(POPULATION_ERR_MSG);
            }
            f.write_all(POPULATION_SEPARATOR).expect(POPULATION_ERR_MSG);
        }
    }

    fn print_progress_header(&mut self) {
        if let Some(ref mut f) = self.stats_log.1 {
            f.write_all(format!("{}\n", self.stats.header()).as_bytes())
                .expect(PROGRESS_ERR_MSG);
        }
    }

    fn print_progress(&mut self, generation: u64, n_solutions: usize) {
        if let Some(ref mut f) = self.stats_log.1 {
            f.write_all(format!("{}\n", self.stats.stats_line(generation, n_solutions)).as_bytes())
                .expect(PROGRESS_ERR_MSG);
        }
    }

    fn fix(&mut self) {
        self.population.par_iter_mut().for_each(|indwf| {
            if indwf.ind.fix() {
                indwf.fitness = None
            }
        });
    }

    fn update_age(&mut self) {
        self.population.par_iter_mut().for_each(|indwf| {
            if let Some(mut fit) = indwf.fitness {
                fit.age += 1;
                indwf.fitness = Some(fit);
            }
        });
    }

    fn get_fitnesses(&self) -> Vec<f64> {
        self.population
            .iter()
            .map(|indwf| indwf.fitness.unwrap().fitness)
            .collect::<Vec<f64>>()
    }

    #[cfg(feature = "global_cache")]
    fn compute_fitnesses(&mut self, refresh_on_nocache: bool) {
        if cfg!(feature = "global_cache") && self.cache_fitness && self.global_cache {
            let (sender, receiver) = channel();
            self.population
                .par_iter()
                .enumerate()
                .filter(|(_i, indwf)| indwf.fitness.is_none())
                .for_each_with(sender, |s, (i, indwf)| {
                    let hashed_ind = self.population[i].ind.hash();
                    let new_fit_value = {
                        match self.cache_map.get(&hashed_ind) {
                            Some(val) => *val,
                            None => indwf.ind.fitness(),
                        }
                    };
                    s.send((i, new_fit_value, hashed_ind)).unwrap();
                });
            for (i, new_fit_value, hashed_ind) in receiver {
                // only none fitness individuals are sent to the receiver
                self.population[i].fitness = Some(Fitness {
                    age: 0,
                    fitness: new_fit_value,
                    original_fitness: new_fit_value,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                });
                // insert in cache if it is not already in it
                self.cache_map.entry(hashed_ind).or_insert(new_fit_value);
            }
        } else {
            self.compute_fitnesses_without_global_cache(refresh_on_nocache);
        }
    }

    #[cfg(not(feature = "global_cache"))]
    fn compute_fitnesses(&mut self, refresh_on_nocache: bool) {
        self.compute_fitnesses_without_global_cache(refresh_on_nocache);
    }

    fn compute_fitnesses_without_global_cache(&mut self, refresh_on_nocache: bool) {
        self.population
            .par_iter_mut()
            .filter(|indwf| {
                if refresh_on_nocache {
                    true
                } else {
                    indwf.fitness.is_none()
                }
            })
            .for_each(|indwf| {
                let new_fit_value = indwf.ind.fitness();
                // this match is only to keep the age
                match indwf.fitness {
                    Some(fit) => {
                        indwf.fitness = Some(Fitness {
                            fitness: new_fit_value,
                            ..fit
                        })
                    }
                    None => {
                        indwf.fitness = Some(Fitness {
                            age: 0,
                            fitness: new_fit_value,
                            original_fitness: new_fit_value,
                            age_effect: 0.0,
                            refitness_effect: 0.0,
                        })
                    }
                }
            });
    }

    fn age_unfitness(&mut self) {
        let age_function = &self.age;
        self.population.par_iter_mut().for_each(|indwf| {
            if let Some(fit) = indwf.fitness {
                let age_exceed: i64 = fit.age as i64 - age_function.age_threshold() as i64;
                if age_exceed >= 0 {
                    let new_fit = age_function.age_unfitness(age_exceed as u64, fit.fitness);
                    indwf.fitness = Some(Fitness {
                        fitness: new_fit,
                        age: fit.age,
                        original_fitness: fit.original_fitness,
                        age_effect: fit.age_effect + (new_fit - fit.fitness),
                        refitness_effect: fit.refitness_effect,
                    });
                }
            }
        });
    }

    fn get_solutions(&self, solutions: &mut Vec<Ind>) {
        for indwf in &self.population {
            if indwf
                .ind
                .is_solution(indwf.fitness.unwrap().original_fitness)
                && Self::not_found_yet_solution(&solutions, &indwf.ind)
            {
                solutions.push(indwf.ind.clone());
            }
        }
    }

    #[allow(clippy::never_loop)]
    fn not_found_yet_solution(solutions: &[Ind], other: &Ind) -> bool {
        for ind in solutions {
            if other.distance(ind) == 0.0 {
                return false;
            }
        }

        true
    }

    fn cross(&mut self, selected: &[usize]) -> Vec<Reproduction> {
        let reprs_number = (selected.len() + 1) / 2;
        let mut reprs = Vec::with_capacity(reprs_number);
        let (sender, receiver) = channel();

        std::ops::Range {
            start: 0,
            end: reprs_number,
        }
        .into_par_iter()
        .for_each_with(sender, |s, i| {
            let ind1 = i * 2;
            let mut ind2 = ind1 + 1;
            // If the number of selected individuals is odd, the last crossover is done
            // using a random one among the selected individuals
            if ind2 >= selected.len() {
                ind2 = SmallRng::from_entropy().sample(Uniform::from(0..selected.len()));
            }
            let (crossed1, crossed2) = self.crossover.cross(
                &self.population[selected[ind1]].ind,
                &self.population[selected[ind2]].ind,
            );
            s.send((selected[ind1], selected[ind2], crossed1, crossed2))
                .unwrap();
        });
        for (parent1, parent2, child1, child2) in receiver {
            self.population.push(IndWithFitness {
                ind: child1,
                fitness: None,
                _phantom: PhantomData,
            });
            self.population.push(IndWithFitness {
                ind: child2,
                fitness: None,
                _phantom: PhantomData,
            });
            reprs.push(Reproduction {
                parents: (parent1, parent2),
                children: (self.population.len() - 2, self.population.len() - 1),
            });
        }
        reprs
    }

    fn mutate(&mut self, mutation_rate: f64) {
        self.population.par_iter_mut().for_each(|indwf| {
            let mut rgen = SmallRng::from_entropy();
            for gen in 0..indwf.ind.iter().len() {
                let random: f64 = rgen.sample(Standard);
                if random < mutation_rate {
                    indwf.ind.mutate(&mut rgen, gen);
                    indwf.fitness = None;
                }
            }
        });
    }

    fn refitness(&mut self, generation: u64, n_solutions: usize) {
        if !self.population_refitness.skip_refitness() {
            let (sender, receiver) = channel();

            self.population
                .par_iter()
                .enumerate()
                .for_each_with(sender, |s, (i, indwf)| {
                    if let Some(fit) = indwf.fitness {
                        if self.population_refitness.individual_is_refitnessed(
                            i,
                            &self.population,
                            generation,
                            &self.stats.values,
                            n_solutions,
                        ) {
                            let new_fit = self.population_refitness.population_refitness(
                                i,
                                &self.population,
                                generation,
                                &self.stats.values,
                                n_solutions,
                            );
                            if (new_fit - fit.fitness).abs() > 0.000_001 {
                                s.send((
                                    i,
                                    Some(Fitness {
                                        fitness: new_fit,
                                        ..fit
                                    }),
                                ))
                                .unwrap()
                            }
                        }
                    }
                });
            for (i, fit) in receiver {
                self.population[i].fitness = fit;
            }
        }
    }

    fn survival_pressure_kill(&mut self, parents_children: &[Reproduction]) {
        self.survival_pressure.kill(
            self.population_size,
            &mut self.population,
            &parents_children,
        );
    }
}
