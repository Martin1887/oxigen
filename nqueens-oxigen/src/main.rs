extern crate oxigen;
extern crate rand;

use oxigen::prelude::*;
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

fn main() {
    let n_queens: u8 = std::env::args()
        .nth(1)
        .expect("Enter a number between 4 and 255 as argument")
        .parse()
        .expect("Enter a number between 4 and 255 as argument");
    let log = (f64::from(n_queens) * 4_f64).log2().ceil();
    let population_size = 2_i32.pow(log as u32) as usize;
    let (solutions, generation, progress) = GeneticExecution::<u8, QueensBoard>::new()
        .population_size(population_size)
        .genotype_size(n_queens as u8)
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: f64::from(n_queens) / (8_f64 + 2_f64 * log) / 100_f64,
            bound: 0.005,
            coefficient: -0.0002,
        })))
        .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: log - 2_f64,
            bound: log / 1.5,
            coefficient: -0.0005,
        })))
        .select_function(Box::new(SelectionFunctions::Cup))
        .age_function(Box::new(AgeFunctions::Cuadratic(
            AgeThreshold(50),
            AgeSlope(1_f64),
        )))
        .run();

    println!(
        "Finished in the generation {} with a progress of {}",
        generation, progress
    );
    for sol in &solutions {
        println!("{}", sol);
    }
}
