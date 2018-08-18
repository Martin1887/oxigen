extern crate test;

#[allow(unused_imports)]
use tests::test::Bencher;

use super::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::Display;
use std::iter::FromIterator;

#[derive(Clone)]
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
            rs.push_str("\n");
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

    fn iter(&self) -> std::slice::Iter<u8> {
        self.0.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<u8> {
        self.0.into_iter()
    }

    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual = Vec::with_capacity(*size as usize);
        let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.push(rgen.sample(Uniform::from(0..*size)));
        }
        QueensBoard(individual)
    }

    // This function returns the mximum punctuaction possible (n-1, since in the
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
            // println!("row: {}, queen: {}", row, queen);
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
}

#[bench]
fn bench_mutation(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8),
    );
    b.iter(|| {
        gen_exec.mutate(0.5);
    });
}

#[bench]
fn bench_cup(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.selection.select(&current_fitnesses, 5);
    });
}

#[bench]
fn bench_tournaments(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .select_function(Box::new(SelectionFunctions::Tournaments(NTournaments(
                population_size / 2,
            )))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec
            .selection
            .select(&current_fitnesses, population_size / 2);
    });
}

#[bench]
fn bench_roulette(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec
            .selection
            .select(&current_fitnesses, population_size / 2);
    });
}

#[bench]
fn bench_cross(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let selected = vec![
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
        48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92,
    ];
    b.iter(|| {
        gen_exec.cross(&selected);
    });
}

#[bench]
fn bench_not_cached_fitness(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
    });
}

#[bench]
fn bench_fitness(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
        for ind in &mut gen_exec.population {
            ind.1 = None;
        }
    });
}

#[bench]
fn bench_not_cached_fitness_age(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .age_function(Box::new(AgeFunctions::Cuadratic(
                AgeThreshold(0),
                AgeSlope(1_f64),
            )))
            .cache_fitness(false),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.compute_fitnesses(true);
    });
}

#[bench]
fn bench_fitness_age(b: &mut Bencher) {
    let n_queens: u8 = test::black_box(255);
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log2 as u32) as usize;
    let mut gen_exec = test::black_box(
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .genotype_size(n_queens as u8)
            .age_function(Box::new(AgeFunctions::Cuadratic(
                AgeThreshold(0),
                AgeSlope(1_f64),
            ))),
    );
    // Initialize randomly the population
    for _ind in 0..gen_exec.population_size {
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.compute_fitnesses(false);
    });
}

#[bench]
fn bench_get_fitness(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.get_fitnesses();
    });
}

#[bench]
fn bench_update_progress(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    let mut last_progresses: Vec<f64> = Vec::new();
    let last_best = 0_f64;
    let best = current_fitnesses[0];
    b.iter(|| {
        GeneticExecution::<u8, QueensBoard>::update_progress(last_best, best, &mut last_progresses);
    });
}

#[bench]
fn bench_get_solutions(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.get_solutions(&current_fitnesses);
    });
}

#[bench]
fn bench_survival_pressure(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    let current_fitnesses = gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.survival_pressure_kill(&current_fitnesses);
    });
}

#[bench]
fn bench_sort_population(b: &mut Bencher) {
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
        gen_exec.population.push((
            Box::new(QueensBoard::generate(&gen_exec.genotype_size)),
            None,
        ));
    }
    gen_exec.compute_fitnesses(true);
    b.iter(|| {
        gen_exec.sort_population();
    });
}
