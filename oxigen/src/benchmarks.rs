extern crate test;

#[allow(unused_imports)]
use benchmarks::test::Bencher;

use super::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::Display;
use std::iter::FromIterator;

#[derive(Clone, PartialEq, Eq, Hash)]
struct QueensBoard(Vec<u8>);
impl Display for QueensBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mut s = String::new();
        for row in self.iter() {
            let mut rs = String::from("|");
            for i in 0..self.0.len() {
                if i == *row as usize {
                    rs.push_str("Q|");
                } else {
                    rs.push_str(" |")
                }
            }
            rs.push('\n');
            s.push_str(&rs);
        }
        write!(f, "{}", s)
    }
}

impl FromIterator<u8> for QueensBoard {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = u8>,
    {
        QueensBoard {
            0: iter.into_iter().collect(),
        }
    }
}

impl Genotype<u8> for QueensBoard {
    type ProblemSize = u8;
    #[cfg(feature = "global_cache")]
    type GenotypeHash = Self;

    fn iter(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<u8> {
        self.0.into_iter()
    }
    fn from_iter<I: Iterator<Item = u8>>(&mut self, genes: I) {
        self.0 = genes.collect();
    }

    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual = Vec::with_capacity(*size as usize);
        let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.push(rgen.sample(Uniform::from(0..*size)));
        }
        QueensBoard(individual)
    }

    // This function returns the maximum score possible (n-1, since in the
    // worst case n-1 queens must be moved to get a solution) minus the number of
    // queens that collide with others
    fn fitness(&self) -> f64 {
        let size = self.0.len();
        let diags_exceed = size as isize - 1isize;
        let mut collisions = 0;
        let mut verticals = Vec::with_capacity(size);
        let mut diagonals = Vec::with_capacity(size + diags_exceed as usize);
        let mut inv_diags = Vec::with_capacity(size + diags_exceed as usize);
        for _i in 0..size {
            verticals.push(false);
            diagonals.push(false);
            inv_diags.push(false);
        }
        for _i in 0..diags_exceed as usize {
            diagonals.push(false);
            inv_diags.push(false);
        }

        for (row, queen) in self.0.iter().enumerate() {
            let mut collision = if verticals[*queen as usize] { 1 } else { 0 };
            verticals[*queen as usize] = true;

            // A collision exists in the diagonal if col-row have the same value
            // for more than one queen
            let diag = ((*queen as isize - row as isize) + diags_exceed) as usize;
            if diagonals[diag] {
                collision = 1;
            }
            diagonals[diag] = true;

            // A collision exists in the inverse diagonal if n-1-col-row have the
            // same value for more than one queen
            let inv_diag =
                ((diags_exceed - *queen as isize - row as isize) + diags_exceed) as usize;
            if inv_diags[inv_diag] {
                collision = 1;
            }
            inv_diags[inv_diag] = true;

            collisions += collision;
        }

        (size - 1 - collisions) as f64
    }

    fn mutate(&mut self, rgen: &mut SmallRng, index: usize) {
        self.0[index] = rgen.sample(Uniform::from(0..self.0.len())) as u8;
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness as usize == self.0.len() - 1
    }

    #[cfg(feature = "global_cache")]
    fn hash(&self) -> Self {
        self.clone()
    }
}

#[bench]
fn bench_generation_run_tournaments_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .select_function(Box::new(SelectionFunctions::Tournaments(NTournaments(
                population_size / 4,
            ))))
            .stop_criterion(Box::new(StopCriteria::Generation(1))),
    );
    while gen_exec.population.len() < gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.fix();
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.run_loop();
    });
}

#[bench]
fn bench_refitness_none_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.refitness(1, 0.0, 0);
    });
}

#[bench]
fn bench_refitness_niches_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .population_refitness_function(Box::new(PopulationRefitnessFunctions::Niches(
                NichesAlpha(1.0),
                Box::new(NichesBetaRates::Constant(1.0)),
                NichesSigma(0.2),
            ))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.refitness(1, 0.0, 0);
    });
}

#[bench]
fn bench_mutation_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let mutation_rate = test::black_box(0.1);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.mutate(mutation_rate);
    });
}

#[bench]
fn bench_selection_cup_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .select_function(Box::new(SelectionFunctions::Cup)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    b.iter(|| {
        gen_exec.selection.select(&current_fitnesses, 8);
    });
}

#[bench]
fn bench_selection_tournaments_256inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .select_function(Box::new(SelectionFunctions::Tournaments(NTournaments(
                population_size / 4,
            )))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    b.iter(|| {
        gen_exec
            .selection
            .select(&current_fitnesses, population_size / 2);
    });
}

#[bench]
fn bench_selection_roulette_256inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .select_function(Box::new(SelectionFunctions::Roulette)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    b.iter(|| {
        gen_exec
            .selection
            .select(&current_fitnesses, population_size / 4);
    });
}

#[bench]
fn bench_cross_single_point_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .crossover_function(Box::new(CrossoverFunctions::SingleCrossPoint)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    b.iter(|| {
        gen_exec.cross(&selected);
    });
}

#[bench]
fn bench_cross_multi_point_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .crossover_function(Box::new(CrossoverFunctions::MultiCrossPoint)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    b.iter(|| {
        gen_exec.cross(&selected);
    });
}

#[bench]
fn bench_cross_uniform_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .crossover_function(Box::new(CrossoverFunctions::UniformCross)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    b.iter(|| {
        gen_exec.cross(&selected);
    });
}

#[bench]
fn bench_fitness_not_cached_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .cache_fitness(false),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
    });
}

#[bench]
fn bench_fitness_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .global_cache(false),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.compute_fitnesses(false);
    });
}

#[bench]
#[cfg(feature = "global_cache")]
fn bench_fitness_global_cache_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
        for ind in &mut gen_exec.population {
            ind.fitness = None;
        }
    });
}

#[bench]
fn bench_fitness_age_not_cached_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .age_function(Box::new(AgeFunctions::Quadratic(
                AgeThreshold(0),
                AgeSlope(1_f64),
            )))
            .cache_fitness(false),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
        gen_exec.age_unfitness();
    });
}

#[bench]
fn bench_fitness_age_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .global_cache(false)
            .age_function(Box::new(AgeFunctions::Quadratic(
                AgeThreshold(0),
                AgeSlope(1_f64),
            ))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.compute_fitnesses(false);
        gen_exec.age_unfitness();
    });
}

#[bench]
fn bench_get_fitnesses_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.get_fitnesses();
    });
}

#[bench]
fn bench_update_progress_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let mut last_progresses: Vec<f64> = Vec::new();
    let last_best = 0_f64;
    let best = current_fitnesses[0];
    b.iter(|| {
        GeneticExecution::<u8, QueensBoard>::update_progress(last_best, best, &mut last_progresses);
    });
}

#[bench]
fn bench_get_solutions_1024inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    // Put 128 random individuals as solutions (though they are not) to measure comparisons
    let mut solutions = test::black_box(Vec::with_capacity(128));
    for (i, ind) in gen_exec.population.iter().enumerate() {
        if i % 8 == 0 {
            solutions.push(ind.ind.clone());
        }
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.get_solutions(&mut solutions);
    });
}

#[bench]
fn bench_survival_pressure_worst_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(SurvivalPressureFunctions::Worst)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_replace_most_similar_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenReplaceMostSimilar,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_replace_parents_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenReplaceParents,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_fight_most_similar_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenFightMostSimilar,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_fight_parents_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(SurvivalPressureFunctions::ChildrenFightParents)),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_overpopulation_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(SurvivalPressureFunctions::Overpopulation(
                M::new(
                    population_size - population_size / 4,
                    population_size / 2,
                    population_size,
                ),
            ))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_competitive_overpopulation_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::CompetitiveOverpopulation(M::new(
                    population_size - population_size / 4,
                    population_size / 2,
                    population_size,
                )),
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_deterministic_overpopulation_255inds(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::DeterministicOverpopulation,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_replace_parents_and_the_rest_random_most_similar_255inds(
    b: &mut Bencher,
) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenReplaceParentsAndTheRestRandomMostSimilar,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
fn bench_survival_pressure_children_replace_parents_and_the_rest_most_similar_255inds(
    b: &mut Bencher,
) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenReplaceParentsAndTheRestMostSimilar,
            )),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    let current_fitnesses = gen_exec.get_fitnesses();
    let selected = gen_exec.selection.select(&current_fitnesses, 8);
    let mut parents_children = Vec::with_capacity(selected.len() / 2);
    let mut i = 0;
    while parents_children.len() < selected.len() / 2 {
        parents_children.push(Reproduction {
            parents: (selected[i * 2], selected[i * 2 + 1]),
            children: (selected[i * 2] + 1, selected[i * 2 + 1] + 1),
        });
        i += 1;
    }
    let population_children_size = population_size + 255;
    // add dummy individuals until reach the initial number of individuals
    while gen_exec.population.len() < population_children_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard { 0: vec![0; 255] },
            Some(Fitness {
                age: 0,
                fitness: 0.0,
                original_fitness: 0.0,
                age_effect: 0.0,
                refitness_effect: 0.0,
            }),
        ));
    }
    b.iter(move || {
        gen_exec.survival_pressure_kill(&parents_children);
        // add dummy individuals until reach the initial number of individuals
        while gen_exec.population.len() < population_children_size {
            gen_exec.population.push(IndWithFitness::new(
                QueensBoard { 0: vec![0; 255] },
                Some(Fitness {
                    age: 0,
                    fitness: 0.0,
                    original_fitness: 0.0,
                    age_effect: 0.0,
                    refitness_effect: 0.0,
                }),
            ));
        }
    });
}

#[bench]
#[allow(clippy::unit_arg)]
fn bench_distance_255(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let population_size = 255_usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .survival_pressure_function(Box::new(SurvivalPressureFunctions::Worst)),
    );
    gen_exec.population = test::black_box(Vec::new());
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push(IndWithFitness::new(
            QueensBoard::generate(&gen_exec.genotype_size),
            None,
        ));
    }
    b.iter(move || {
        test::black_box(gen_exec.population.iter().for_each(|ind| {
            let _dist = gen_exec.population[0].ind.distance(&ind.ind);
        }))
    });
}
