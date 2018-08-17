//! This crate provides functions for parallel genetic algorithm execution.

extern crate historian;
extern crate rand;
extern crate rayon;

pub mod age;
pub mod crossover;
pub mod genotype;
pub mod mutation_rate;
pub mod prelude;
pub mod selection;
pub mod selection_rate;
pub mod slope_params;
pub mod stop_criteria;
pub mod survival_pressure;

pub use age::*;
pub use crossover::*;
pub use genotype::Genotype;
pub use mutation_rate::*;
pub use selection::*;
pub use selection_rate::*;
pub use slope_params::*;
pub use stop_criteria::*;
pub use survival_pressure::*;

use historian::Histo;
use rand::distributions::{Standard, Uniform};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::sync::mpsc::channel;

const POPULATION_SEPARATOR: &[u8] = b"\n\n\n\n---------------------------------\n\n\n\n";
const POPULATION_ERR_MSG: &str = "Error writing on population log file";
const PROGRESS_ERR_MSG: &str = "Error writing in progress log file";
const PROGRESS_HEADER: &[u8] = b"Generation\t\
    Solutions\t\
    Last progress\t\
    Progress avg\t\
    Progress std\t\
    Progress max\t\
    Progress min\t\
    Progress p10\t\
    Progress p25\t\
    Progress median\t\
    Progress p75\t\
    Progress p90\t\
    Fitness avg\t\
    Fitness std\t\
    Fitness max\t\
    Fitness min\t\
    Fitness p10\t\
    Fitness p25\t\
    Fitness median\t\
    Fitness p75\t\
    Fitness p90\n";

/// Struct that defines a genetic algorithm execution.
pub struct GeneticExecution<T, Ind: Genotype<T>> {
    /// The number of individuals in the population.
    population_size: usize,
    /// Population with all individuals.
    population: Vec<Box<Ind>>,
    /// Size of the genotype problem.
    genotype_size: Ind::ProblemSize,
    /// The mutation rate variation along iterations and progress.
    mutation_rate: Box<MutationRate>,
    /// The number of stages in the cup whose individuals are selected to crossover.
    selection_rate: Box<SelectionRate>,
    /// The selection function.
    selection: Box<Selection>,
    /// The age fitness decrease function.
    age: Box<Age>,
    /// The crossover function.
    crossover: Box<Crossover<T, Ind>>,
    /// The function used to replace individuals in the population.
    survival_pressure: Box<SurvivalPressure<T, Ind>>,
    /// The stop criterion to finish execution.
    stop_criterion: Box<StopCriterion>,
    /// Cache fitness value of individuals or compute it in each iteration.
    cache_fitness: bool,
    /// Progress log, writes statistics of the population every certain number of generations.
    progress_log: (u64, Option<File>),
    /// Population log, writes the population and fitnesses every certain number of generations.
    population_log: (u64, Option<File>),
    /// Vector with the fitness cache of all individuals.
    fitnesses: Vec<Option<Fitness>>,
}

#[derive(Copy, Clone)]
struct Fitness {
    age: u64,
    fitness: f64,
    original_fitness: f64,
}

impl<T, Ind: Genotype<T>> Default for GeneticExecution<T, Ind> {
    fn default() -> Self {
        GeneticExecution {
            population_size: 64,
            population: Vec::new(),
            genotype_size: Ind::ProblemSize::default(),
            mutation_rate: Box::new(MutationRates::Constant(0.1)),
            selection_rate: Box::new(SelectionRates::Constant(2)),
            selection: Box::new(SelectionFunctions::Cup),
            age: Box::new(AgeFunctions::None),
            crossover: Box::new(CrossoverFunctions::SingleCrossPoint),
            survival_pressure: Box::new(SurvivalPressureFunctions::Worst),
            stop_criterion: Box::new(StopCriteria::SolutionFound),
            cache_fitness: true,
            progress_log: (0, None),
            population_log: (0, None),
            fitnesses: vec![None; 64],
        }
    }
}

impl<T, Ind: Genotype<T>> GeneticExecution<T, Ind> {
    /// Creates a new default genetic algorithm execution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the population size.
    pub fn population_size(mut self, new_pop: usize) -> Self {
        self.population_size = new_pop;
        self.fitnesses = vec![None; new_pop];
        self
    }

    /// Sets the genotype size.
    pub fn genotype_size(mut self, new_size: Ind::ProblemSize) -> Self {
        self.genotype_size = new_size;
        self
    }

    /// Sets the mutation rate.
    pub fn mutation_rate(mut self, new_mut: Box<MutationRate>) -> Self {
        self.mutation_rate = new_mut;
        self
    }

    /// Sets the number of tournament stages whose individuals are selected for crossover.
    pub fn selection_rate(mut self, new_sel_rate: Box<SelectionRate>) -> Self {
        self.selection_rate = new_sel_rate;
        self
    }

    /// Sets the selection function of the genetic algorithm.
    pub fn select_function(mut self, new_sel: Box<Selection>) -> Self {
        self.selection = new_sel;
        self
    }

    /// Sets the age function of the genetic algorithm.
    pub fn age_function(mut self, new_age: Box<Age>) -> Self {
        self.age = new_age;
        self
    }

    /// Sets the crossover function of the genetic algorithm.
    pub fn crossover_function(mut self, new_cross: Box<Crossover<T, Ind>>) -> Self {
        self.crossover = new_cross;
        self
    }

    /// Sets the survival pressure function of the genetic algorithm.
    pub fn survival_pressure_function(mut self, new_surv: Box<SurvivalPressure<T, Ind>>) -> Self {
        self.survival_pressure = new_surv;
        self
    }

    /// Sets the stop criterion of the genetic algorithm.
    pub fn stop_criterion(mut self, new_crit: Box<StopCriterion>) -> Self {
        self.stop_criterion = new_crit;
        self
    }

    /// Sets the cache fitness flag.
    pub fn cache_fitness(mut self, new_cache: bool) -> Self {
        self.cache_fitness = new_cache;
        self
    }

    /// Sets the progress log.
    pub fn progress_log(mut self, generations: u64, log_file: File) -> Self {
        self.progress_log = (generations, Some(log_file));
        self
    }

    /// Sets the progress log.
    pub fn population_log(mut self, generations: u64, log_file: File) -> Self {
        self.population_log = (generations, Some(log_file));
        self
    }

    /// Run the genetic algorithm executiion until the `stop_criterion` is satisfied.
    ///
    /// # Returns
    ///
    /// - A vector with the individuals of the population that are solution of the problem.
    /// - The number of generations run.
    /// - The average progress in the last generations.
    pub fn run(mut self) -> (Vec<Box<Ind>>, u64, f64) {
        let mut generation: u64 = 0;
        let mut last_progresses: Vec<f64> = Vec::new();
        let mut progress: f64 = 0.0;
        let mut solutions: Vec<usize> = Vec::new();
        let mut mutation_rate;
        let mut selection_rate;
        let mut current_fitnesses = Vec::new();
        let mut last_best = 0f64;

        // Initialize randomly the population
        for _ind in 0..self.population_size {
            self.population
                .push(Box::new(Ind::generate(&self.genotype_size)));
        }

        if self.progress_log.0 > 0 {
            self.print_progress_header();
        }

        while !self.stop_criterion.stop(
            generation,
            progress,
            solutions.len() as u16,
            &current_fitnesses,
        ) {
            generation += 1;

            mutation_rate = self.mutation_rate.rate(
                generation,
                progress,
                solutions.len() as u16,
                &current_fitnesses,
            );
            selection_rate = self.selection_rate.rate(
                generation,
                progress,
                solutions.len() as u16,
                &current_fitnesses,
            );

            current_fitnesses = self.compute_fitnesses(true);
            let selected = self.selection.select(&current_fitnesses, selection_rate);
            self.cross(&selected);
            self.mutate(mutation_rate);
            self.compute_fitnesses(false);
            self.sort_population();
            current_fitnesses = self.get_fitnesses();
            self.survival_pressure_kill(&current_fitnesses);

            current_fitnesses = self.get_fitnesses();
            solutions = self.get_solutions(&current_fitnesses);
            let best = current_fitnesses[0];
            progress = Self::update_progress(last_best, best, &mut last_progresses);
            last_best = best;

            if self.progress_log.0 > 0 && generation % self.progress_log.0 == 0 {
                self.print_progress(
                    generation,
                    &current_fitnesses,
                    progress,
                    &last_progresses,
                    solutions.len(),
                );
            }
            if self.population_log.0 > 0 && generation % self.population_log.0 == 0 {
                self.print_population(generation);
            }
        }

        let mut final_solutions: Vec<Box<Ind>> = Vec::new();
        for i in solutions {
            final_solutions.push(self.population[i].clone());
        }
        (final_solutions, generation, progress)
    }

    fn print_population(&mut self, generation: u64) {
        if let Some(ref mut f) = self.population_log.1 {
            f.write_all(format!("Generation {}\n", generation).as_bytes())
                .expect(POPULATION_ERR_MSG);
            for (i, ind) in self.population.iter().enumerate() {
                f.write_all(
                    format!(
                        "Individual: {}; fitness: {}\n\n",
                        i,
                        self.fitnesses[i].unwrap().fitness
                    ).as_bytes(),
                ).expect(POPULATION_ERR_MSG);
                f.write_all(format!("{}", ind).as_bytes())
                    .expect(POPULATION_ERR_MSG);
            }
            f.write_all(POPULATION_SEPARATOR).expect(POPULATION_ERR_MSG);
        }
    }

    fn print_progress_header(&mut self) {
        if let Some(ref mut f) = self.progress_log.1 {
            f.write_all(PROGRESS_HEADER).expect(PROGRESS_ERR_MSG);
        }
    }

    fn print_progress(
        &mut self,
        generation: u64,
        current_fitnesses: &[f64],
        progress: f64,
        last_progresses: &[f64],
        n_solutions: usize,
    ) {
        if let Some(ref mut f) = self.progress_log.1 {
            let mut progress_hist = Histo::default();
            for prog in last_progresses.iter() {
                progress_hist.measure(*prog);
            }
            let mut fit_hist = Histo::default();
            for fit in current_fitnesses.iter() {
                fit_hist.measure(*fit);
            }
            let fitness_avg =
                current_fitnesses.iter().sum::<f64>() / current_fitnesses.len() as f64;
            f.write_all(
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    generation,
                    n_solutions,
                    last_progresses[last_progresses.len() - 1],
                    progress,
                    (1_f64 / (last_progresses.len() - 1) as f64)
                        * last_progresses
                            .iter()
                            .fold(0_f64, |acc, x| acc + (x - progress).powi(2))
                            .sqrt(),
                    last_progresses
                        .iter()
                        .max_by(|x, y| x.partial_cmp(&y).unwrap())
                        .unwrap(),
                    last_progresses
                        .iter()
                        .min_by(|x, y| x.partial_cmp(&y).unwrap())
                        .unwrap(),
                    progress_hist.percentile(10_f64),
                    progress_hist.percentile(25_f64),
                    progress_hist.percentile(50_f64),
                    progress_hist.percentile(75_f64),
                    progress_hist.percentile(90_f64),
                    fitness_avg,
                    (1_f64 / (current_fitnesses.len() - 1) as f64)
                        * current_fitnesses
                            .iter()
                            .fold(0_f64, |acc, x| acc + (x - fitness_avg).powi(2))
                            .sqrt(),
                    current_fitnesses
                        .iter()
                        .max_by(|x, y| x.partial_cmp(&y).unwrap())
                        .unwrap(),
                    current_fitnesses
                        .iter()
                        .min_by(|x, y| x.partial_cmp(&y).unwrap())
                        .unwrap(),
                    fit_hist.percentile(10_f64),
                    fit_hist.percentile(25_f64),
                    fit_hist.percentile(50_f64),
                    fit_hist.percentile(75_f64),
                    fit_hist.percentile(90_f64),
                ).as_bytes(),
            ).expect(PROGRESS_ERR_MSG);
        }
    }

    fn update_progress(last_best: f64, best: f64, last_progresses: &mut Vec<f64>) -> f64 {
        last_progresses.push(best - last_best);
        if last_progresses.len() == 16 {
            last_progresses.remove(0);
        }

        last_progresses.par_iter().sum::<f64>() / last_progresses.len() as f64
    }

    fn get_fitnesses(&self) -> Vec<f64> {
        self.fitnesses
            .par_iter()
            .map(|f| f.unwrap().fitness)
            .collect::<Vec<f64>>()
    }

    fn compute_fitnesses(&mut self, refresh_on_nocache: bool) -> Vec<f64> {
        let (sender_fit, receiver_fit) = channel();
        let (sender_age, receiver_age) = channel();
        if self.cache_fitness || !refresh_on_nocache {
            self.fitnesses
                .par_iter()
                .enumerate()
                .filter(|(_i, fit)| fit.is_none())
                .for_each_with(sender_fit, |s, (i, _fit)| {
                    let new_fit_value = self.population[i].fitness();
                    s.send((i, 0, new_fit_value, new_fit_value)).unwrap();
                });
            self.fitnesses
                .par_iter()
                .enumerate()
                .filter(|(_i, fit)| fit.is_some())
                .map(|(i, fit)| (i, fit.unwrap()))
                .for_each_with(sender_age, |s, (i, fit)| {
                    let mut new_fit_value = fit.original_fitness;
                    let age_exceed: i64 = fit.age as i64 - self.age.age_threshold() as i64;
                    if age_exceed > 0 {
                        new_fit_value = self.age.age_unfitness(age_exceed as u64, new_fit_value);
                    }
                    s.send((i, fit.age, new_fit_value, fit.original_fitness))
                        .unwrap();
                });
        } else {
            self.fitnesses
                .par_iter()
                .enumerate()
                .for_each_with(sender_fit, |s, (i, fit)| {
                    let mut new_fit_value = self.population[i].fitness();
                    let age: u64 = if fit.is_some() { fit.unwrap().age } else { 0 };
                    let age_exceed: i64 = age as i64 - self.age.age_threshold() as i64;
                    if age_exceed > 0 {
                        new_fit_value = self.age.age_unfitness(age_exceed as u64, new_fit_value);
                    }
                    s.send((i, age, new_fit_value, 0_f64)).unwrap();
                });
        }
        for (i, age, fit, orig_fit) in receiver_fit {
            self.fitnesses[i] = Some(Fitness {
                age: age + refresh_on_nocache as u64,
                fitness: fit,
                original_fitness: orig_fit,
            });
        }
        for (i, age, fit, orig_fit) in receiver_age {
            self.fitnesses[i] = Some(Fitness {
                age: age + refresh_on_nocache as u64,
                fitness: fit,
                original_fitness: orig_fit,
            });
        }

        self.get_fitnesses()
    }

    fn get_solutions(&self, current_fitnesses: &[f64]) -> Vec<usize> {
        let mut solutions = Vec::new();
        let (sender, receiver) = channel();

        self.population
            .par_iter()
            .enumerate()
            .for_each_with(sender, |s, (i, ind)| {
                if ind.is_solution(current_fitnesses[i]) {
                    s.send(i).unwrap();
                }
            });
        for child in receiver {
            solutions.push(child);
        }

        solutions
    }

    fn cross(&mut self, selected: &[usize]) {
        let (sender, receiver) = channel();

        std::ops::Range {
            start: 0,
            end: ((selected.len() + 1) / 2),
        }.into_par_iter()
            .for_each_with(sender, |s, i| {
                let ind1 = i * 2;
                let mut ind2 = ind1 + 1;
                // If the number of selected individuals is odd, the last crossover is done
                // using a random one among the selected individuals
                if ind2 >= selected.len() {
                    ind2 = SmallRng::from_entropy().sample(Uniform::from(0..selected.len()));
                }
                let (crossed1, crossed2) = self.crossover.cross(
                    &self.population[selected[ind1]],
                    &self.population[selected[ind2]],
                );
                s.send(crossed1).unwrap();
                s.send(crossed2).unwrap();
            });
        for child in receiver {
            self.population.push(Box::new(child));
            self.fitnesses.push(None);
        }
    }

    fn mutate(&mut self, mutation_rate: f64) {
        let (sender, receiver) = channel();
        self.population
            .par_iter_mut()
            .enumerate()
            .for_each_with(sender, |s, (i, individual)| {
                let mut rgen = SmallRng::from_entropy();
                for gen in 0..individual.iter().len() {
                    let random: f64 = rgen.sample(Standard);
                    if random < mutation_rate {
                        individual.mutate(&mut rgen, gen);
                        s.send(i).unwrap();
                    }
                }
            });
        for ind in receiver {
            self.fitnesses[ind] = None;
        }
    }

    fn survival_pressure_kill(&mut self, current_fitnesses: &[f64]) {
        for killed in
            self.survival_pressure
                .kill(self.population_size, &self.population, current_fitnesses)
        {
            self.population.remove(killed);
            self.fitnesses.remove(killed);
        }
    }

    fn sort_population(&mut self) {
        Self::quick_sort(&mut self.population, &mut self.fitnesses);
    }

    fn quick_sort<I: Send>(v: &mut [I], fitnesses: &mut [Option<Fitness>]) {
        let mid = Self::quick_sort_partition(v, fitnesses);
        let (lo, hi) = v.split_at_mut(mid);
        let (lo_f, hi_f) = fitnesses.split_at_mut(mid);
        if lo_f.len() > 1 && hi_f.len() > 1 {
            rayon::join(|| Self::quick_sort(lo, lo_f), || Self::quick_sort(hi, hi_f));
        } else if lo_f.len() > 1 {
            Self::quick_sort(lo, lo_f)
        } else if hi_f.len() > 1 {
            Self::quick_sort(hi, hi_f)
        }
    }

    // Partition rearranges all items `>=` to the pivot item (arbitrary selected to be
    // the last item in the slice) to the first half of the slice. It then returns the
    // "dividing point" where the pivot is placed.
    fn quick_sort_partition<I: Send>(v: &mut [I], fitnesses: &mut [Option<Fitness>]) -> usize {
        let pivot = fitnesses.len() - 1;
        let mut i = 0;
        for j in 0..pivot {
            if fitnesses[j].unwrap().fitness >= fitnesses[pivot].unwrap().fitness {
                v.swap(i, j);
                fitnesses.swap(i, j);
                i += 1;
            }
        }
        v.swap(i, pivot);
        fitnesses.swap(i, pivot);
        i
    }
}
