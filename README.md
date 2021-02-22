# oxigen

[![Build Status](https://travis-ci.com/Martin1887/oxigen.svg?branch=master)](https://travis-ci.com/Martin1887/oxigen)
[![Current Crates.io Version](https://img.shields.io/crates/v/oxigen.svg)](https://crates.io/crates/oxigen)

Oxigen is a parallel genetic algorithm library implemented in Rust. The name comes from the merge of `OXI`dación (Rust translated to Spanish) and `GEN`etic.

Oxigen provides the following features:

* Fast and parallel genetic algorithm implementation (it solves the N Queens problem for N=255 in few seconds). For benchmarks view benchmarks section of this file.
* Customizable mutation and selection rates with constant, linear and cuadratic functions according to generations built-in (you can implement your own functions via the `MutationRate` and `SelectionRate` traits).
* Customizable age unfitness of individuals, with no unfitness, linear and cuadratic unfitness with threshold according to generations of the individual built-in (you can implement your own age functions via the `Age` trait).
* Accumulated `Roulette`, `Tournaments` and `Cup` built-in selection functions (you can implement your own selection functions via the `Selection` trait).
* `SingleCrossPoint` built-in crossover function (you can implement your own crossover function via the `Crossover` trait).
* `Worst` built-in survival pressure function (the worst individuals are killed until reaching the original population size). You can implement your own survival pressure functions via the `SurvivalPressure` trait.
* `SolutionFound`, `Generation` and `Progress` built-in stop criteria (you can implement your own stop criteria via the `StopCriterion` trait).
* `Genotype` trait to define the genotype of your genetic algorithm. Whatever struct can implement the `Genotype` trait under the following restrictions:
    - It has a `iter` function that returns a `use std::slice::Iter` over its genes.
    - It has a `into_iter` function that consumes the individual and returns a `use std::vec::IntoIter` over its genes.
    - It implements `FromIterator` over its genes type, `Display`, `Clone`, `Send` and `Sync`.
    - It has functions to `generate` a random individual, to `mutate` an individual, to get the `fitness` of an individual and to know if and individual `is_solution` of the problem.
* Individual's fitness is cached to not do unnecessary recomputations (this can be disabled with `.cache_fitness(false)` if your fitness function is stochastic and so you need to recompute fitness in each generation).
* Progress statistics can be configured to be printed every certain number of generations to a file.
* Population individuals with their fitnesses can be configured to be printed every certain number of generations to a file.
* Specific initial individuals can be inserted in the genetic algorithm execution.
* Genetic executions can be resumed using the population of the last generation as initial population.
* Coevolution is possible executing little genetic algorithm re-executions inside the fitness function.


## Usage

In your `Cargo.toml` file add the `oxigen` dependency:

```
[dependencies]
oxigen = "^1.5"
```

To use `oxigen` `use oxigen::prelude::*` and call the `run` method over a `GeneticExecution` instance overwriting the default hyperparameters and functions folllowing your needs:

```rust
let n_queens: u8 = std::env::args()
    .nth(1)
    .expect("Enter a number between 4 and 255 as argument")
    .parse()
    .expect("Enter a number between 4 and 255 as argument");

let progress_log = File::create("progress.csv").expect("Error creating progress log file");
let population_log = File::create("population.txt").expect("Error creating population log file");
let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();

let population_size = 2_i32.pow(log2 as u32) as usize;

let (solutions, generation, progress) = GeneticExecution::<u8, QueensBoard>::new()
    .population_size(population_size)
    .genotype_size(n_queens as u8)
    .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
        start: f64::from(n_queens) / (8_f64 + 2_f64 * log2) / 100_f64,
        bound: 0.005,
        coefficient: -0.0002,
    })))
    .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
        start: log2 - 2_f64,
        bound: log2 / 1.5,
        coefficient: -0.0005,
    })))
    .select_function(Box::new(SelectionFunctions::Cup))
    .age_function(Box::new(AgeFunctions::Cuadratic(
        AgeThreshold(50),
        AgeSlope(1_f64),
    )))
    .progress_log(20, progress_log)
    .population_log(2000, population_log)
    .run();
```

For a full example visit the [nqueens-oxigen](nqueens-oxigen/src/main.rs) example.

For more information visit the [documentation](https://docs.rs/oxigen).


### Resuming a previous execution

Since version 1.1.0, genetic algorithm executions return the population of the last generation and new genetic executions accept a initial population. This permits to resuming previous executions and it also enables coevolution, since little genetic algorithm re-executions can be launched in the fitness function.

In the following example a execution with 10000 generations is launched and after it is resumed until finding a solution with different rates.

```rust
let n_queens: u8 = std::env::args()
    .nth(1)
    .expect("Enter a number between 4 and 255 as argument")
    .parse()
    .expect("Enter a number between 4 and 255 as argument");

let progress_log = File::create("progress.csv").expect("Error creating progress log file");
let population_log = File::create("population.txt").expect("Error creating population log file");
let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();

let population_size = 2_i32.pow(log2 as u32) as usize;

let (_solutions, _generation, _progress, population) = GeneticExecution::<u8, QueensBoard>::new()
    .population_size(population_size)
    .genotype_size(n_queens as u8)
    .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
        start: f64::from(n_queens) / (8_f64 + 2_f64 * log2) / 100_f64,
        bound: 0.005,
        coefficient: -0.0002,
    })))
    .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
        start: log2 - 2_f64,
        bound: log2 / 1.5,
        coefficient: -0.0005,
    })))
    .select_function(Box::new(SelectionFunctions::Cup))
    .age_function(Box::new(AgeFunctions::Cuadratic(
        AgeThreshold(50),
        AgeSlope(1_f64),
    )))
    .stop_criterion(Box::new(StopCriteria::Generation(10000)))
    .run();

let (solutions, generation, progress, _population) = GeneticExecution::<u8, QueensBoard>::new()
    .population_size(population_size)
    .genotype_size(n_queens as u8)
    .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
        start: f64::from(n_queens) / (8_f64 + 4_f64 * log2) / 100_f64,
        bound: 0.005,
        coefficient: -0.0002,
    })))
    .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
        start: log2 - 4_f64,
        bound: log2 / 1.5,
        coefficient: -0.0005,
    })))
    .select_function(Box::new(SelectionFunctions::Cup))
    .age_function(Box::new(AgeFunctions::Cuadratic(
        AgeThreshold(50),
        AgeSlope(1_f64),
    )))
    .population(population)
    .progress_log(20, progress_log)
    .population_log(2000, population_log)
    .run();
```


## Building

To build oxigen, use `cargo` like for any Rust project:

* `cargo build` to build in debug mode.
* `cargo build --release` to build with optimizations.

To run benchmarks, you will need a nightly Rust compiler. Uncomment the lines `// #![feature(test)]` and `// mod benchmarks;` from `lib.rs` and then bechmarks can be run using `cargo bench`.


## Benchmarks

The following benchmarks have been created to measure the genetic algorithm functions performance:

```
running 16 tests
test benchmarks::bench_cross_multi_point_255inds       ... bench:     379,946 ns/iter (+/- 15,055)
test benchmarks::bench_cross_single_point_255inds      ... bench:     171,017 ns/iter (+/- 18,285)
test benchmarks::bench_cross_uniform_255inds           ... bench:     118,836 ns/iter (+/- 33,141)
test benchmarks::bench_fitness_1024inds                ... bench:     511,328 ns/iter (+/- 71,904)
test benchmarks::bench_fitness_age_1024inds            ... bench:      54,180 ns/iter (+/- 6,610)
test benchmarks::bench_get_fitnesses_1024inds          ... bench:      19,679 ns/iter (+/- 859)
test benchmarks::bench_get_solutions_1024inds          ... bench:      31,845 ns/iter (+/- 3,719)
test benchmarks::bench_mutation_1024inds               ... bench:           7 ns/iter (+/- 0)
test benchmarks::bench_not_cached_fitness_1024inds     ... bench:     488,846 ns/iter (+/- 57,395)
test benchmarks::bench_not_cached_fitness_age_1024inds ... bench:     492,824 ns/iter (+/- 42,625)
test benchmarks::bench_selection_cup_255inds           ... bench:     354,222 ns/iter (+/- 53,300)
test benchmarks::bench_selection_roulette_256inds      ... bench:     140,935 ns/iter (+/- 2,051)
test benchmarks::bench_selection_tournaments_256inds   ... bench:     482,830 ns/iter (+/- 47,955)
test benchmarks::bench_sort_population_1024inds        ... bench:       1,441 ns/iter (+/- 22)
test benchmarks::bench_survival_pressure_255inds       ... bench:          62 ns/iter (+/- 2)
test benchmarks::bench_update_progress_1024inds        ... bench:       7,431 ns/iter (+/- 352)
```

These benchmarks have been executed in a Intel Core i7 6700K with 16 GB of DDR4
and a 512 GB Samsung 950 Pro NVMe SSD in ext4 format in Fedora 30
with Linux kernel 5.2.9.

The difference of performance among the different fitness benchmarks have the following explanations:

 * `bench_fitness` measures the performance of a cached execution cleaning the fitnesses after each bench iteration. This cleaning is the reason of being a bit slower than not cached benchmarks.
 * `bench_fitness_age` measures the performance with fitness cached in all bench iterations, so it is very much faster.
 * Not cached benchmarks measure the performance of not cached executions, with 1 generation individuals in the last case, so the performance is similar but a bit slower for the benchmark that must apply age unfitness.


## Contributing

Contributions are absolutely, positively welcome and encouraged! Contributions
come in many forms. You could:

  1. Submit a feature request or bug report as an [issue](https://github.com/Martin1887/oxigen/issues).
  2. Ask for improved documentation as an [issue](https://github.com/Martin1887/oxigen/issues).
  3. Comment on issues that require
     feedback.
  4. Contribute code via [pull requests](https://github.com/Martin1887/oxigen/pulls), don't forget to run `cargo fmt` before submitting your PR!

We aim to keep Rocket's code quality at the highest level. This means that any
code you contribute must be:

  * **Commented:** Public items _must_ be commented.
  * **Documented:** Exposed items _must_ have rustdoc comments with
    examples, if applicable.
  * **Styled:** Your code should be `rustfmt`'d when possible.
  * **Simple:** Your code should accomplish its task as simply and
     idiomatically as possible.
  * **Tested:** You should add (and pass) convincing tests for any functionality you add when it is possible.
  * **Focused:** Your code should do what it's supposed to do and nothing more.

Note that unless you
explicitly state otherwise, any contribution intentionally submitted for
inclusion in oxigen by you shall be dual licensed under the MIT License and
Apache License, Version 2.0, without any additional terms or conditions.

## Reference
Pozo, Martín. "Oxigen: Fast, parallel, extensible and adaptable genetic algorithm library written in Rust".


### Bibtex
```tex
@misc{
  title={Oxigen: Fast, parallel, extensible and adaptable genetic algorithm library written in Rust},
  author={Pozo, Martín},
  howpublised = "\url{https://github.com/Martin1887/oxigen}"
}
```

## License

oxigen is licensed under Mozilla Public License 2.0.
