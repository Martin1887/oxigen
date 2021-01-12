extern crate oxigen;
extern crate rand;

use oxigen::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::Display;
use std::fs::File;

#[derive(Clone, PartialEq, Eq, std::hash::Hash)]
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

impl Genotype<u8> for QueensBoard {
    type Environment = u8;
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

    fn generate(size: &Self::Environment) -> Self {
        let mut individual = Vec::with_capacity(*size as usize);
        let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.push(rgen.sample(Uniform::from(0..*size)));
        }
        QueensBoard(individual)
    }

    // This function returns the maximum score possible (n, since in the
    // worst case n queens collide) minus the number of queens that collide with others
    fn fitness(&self) -> f64 {
        let size = self.0.len();
        let diags_exceed = size as isize - 1_isize;
        let mut collisions = Vec::with_capacity(size);
        let mut verticals: Vec<isize> = Vec::with_capacity(size);
        let mut diagonals: Vec<isize> = Vec::with_capacity(size + diags_exceed as usize);
        let mut inv_diags: Vec<isize> = Vec::with_capacity(size + diags_exceed as usize);
        for _i in 0..size {
            verticals.push(-1);
            diagonals.push(-1);
            inv_diags.push(-1);
            collisions.push(false);
        }
        for _i in 0..diags_exceed as usize {
            diagonals.push(-1);
            inv_diags.push(-1);
        }

        for (row, queen) in self.0.iter().enumerate() {
            let mut collision = verticals[*queen as usize];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            verticals[*queen as usize] = row as isize;

            // A collision exists in the diagonal if col-row have the same value
            // for more than one queen
            let diag = ((*queen as isize - row as isize) + diags_exceed) as usize;
            collision = diagonals[diag];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            diagonals[diag] = row as isize;

            // A collision exists in the inverse diagonal if n-1-col-row have the
            // same value for more than one queen
            let inv_diag =
                ((diags_exceed - *queen as isize - row as isize) + diags_exceed) as usize;
            collision = inv_diags[inv_diag];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            inv_diags[inv_diag] = row as isize;
        }

        (size - collisions.into_iter().filter(|r| *r).count()) as f64
    }

    fn mutate(&mut self, rgen: &mut SmallRng, index: usize) {
        self.0[index] = rgen.sample(Uniform::from(0..self.0.len())) as u8;
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness as usize == self.0.len()
    }

    fn hash(&self) -> Self {
        self.clone()
    }
}

fn main() {
    let n_queens: u8 = std::env::args()
        .nth(1)
        .expect("Enter a number between 4 and 255 as argument")
        .parse()
        .expect("Enter a number between 4 and 255 as argument");
    let progress_log = File::create("progress.csv").expect("Error creating progress log file");
    let population_log =
        File::create("population.txt").expect("Error creating population log file");
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();
    let mut population_size = 2_i32.pow(log2 as u32) as usize;
    if n_queens <= 8 {
        population_size *= 2;
    }

    let (solutions, generation, mut stats, _population) =
        GeneticExecution::<u8, QueensBoard>::new()
            .population_size(population_size)
            .environment(n_queens as u8)
            .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
                start: f64::from(n_queens) / (2_f64 + 4_f64 * log2) / 100_f64,
                bound: 0.005,
                coefficient: -0.00002,
            })))
            .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
                start: 3_f64,
                bound: 6_f64,
                coefficient: 0.05,
            })))
            .select_function(Box::new(SelectionFunctions::Tournaments(NTournaments(
                population_size / 2,
            ))))
            .crossover_function(Box::new(CrossoverFunctions::UniformCross))
            .population_refitness_function(Box::new(PopulationRefitnessFunctions::Niches(
                NichesAlpha(0.8),
                Box::new(NichesBetaRates::Linear(SlopeParams {
                    start: 0.0025,
                    bound: 10.0_f64.min(log2 * log2 / 6.0),
                    coefficient: 0.000001 * log2 * log2,
                })),
                NichesSigma(0.6),
            )))
            // Fighting to parents works but the evolution is slower with many queens
            /*
            .survival_pressure_function(Box::new(
                SurvivalPressureFunctions::ChildrenFightParentsAndTheRestWorst,
            ))*/
            .survival_pressure_function(Box::new(SurvivalPressureFunctions::Worst))
            .age_function(Box::new(AgeFunctions::Linear(
                AgeThreshold(5),
                AgeSlope(0.5),
            )))
            .stop_criterion(Box::new(StopCriteria::SolutionsFound(
                4.min(n_queens as usize / 2),
            )))
            .stats_log(2_000, progress_log)
            .population_log(2_000, population_log)
            .run();

    println!(
        "Finished in the generation {} with an average progress of {}",
        generation,
        OxigenStatsFields::AvgProgressAvg.function()(&mut stats)
    );
    for sol in &solutions {
        println!("{}", sol);
    }
}
