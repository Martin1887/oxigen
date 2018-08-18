# oxigen

oxigen is a parallel genetic algorithm library implemented in Rust. The name comes from the merge of `OXI`dación (Rust translated to Spanish) and `GEN`etic.

oxigen provides the following features:

* Fast and parallel genetic algorithm implementation (it solves the N Queens problem for N=255 in few seconds). For benchmarks view benchmarks section of this file.
* Customizable mutation and selection rates with constant, linear and cuadratic functions according to generations built-in (you can implement your own functions via the `MutationRate` and `SelectionRate` traits).
* Customizable age unfitness of individuals, with no unfitness, linear and cuadratic unfitness with threshold according to generations of the individual built-in (you can implement your own age functions via the `Age` trait).
* Accumulated `Roulette`, `Tournaments` and `Cup` built-in selection functions (you can implement your own selection functions via the `Selection` trait).
* `SingleCrossPoint` built-in crossover function (you can implement your own crossover function via the `Crossover` trait).
* `Worst` built-in survival pressure function (the worst individuals are killed until reaching the original population size). You can implement your own survival pressure functions via the `SurvivalPressure` trait.
* `SolutionFound`, `Generation` and `Progress` built-in stop criteria (you can implement your own stop criteria via the `StopCriterion` trait).
* `Genotype` trait to define the genotype of your genetic algorithm. Whatever struct can implement the `Genotype` trait under the following restrictions:
    - It has a `iter` function that returns a `use std::slice::Iter` over its genes.
    - It has a 'into_iter' function that consumes the individual and returns a ' use std::vec::IntoIter' over its genes.
    - It implements `FromIterator` over its genes type, `Display`, `Clone`, `Send` and `Sync`.
    - It has functions to `generate` a random individual, to `mutate` an individual, to get the `fitness` of an individual and to know if and individual `is_solution` of the problem.
* Individual's fitness is cached to not do unnecessary recomputations (this can be disabled with `.cache_fitness(false)` if your fitness function is stochastic and so you need to recompute fitness in each generation).
* Progress statistics can be configured to be printed every certain number of generations to a file.
* Population individuals with their fitnesses can be configured to be printed every certain number of generations to a file.


## Usage

To use `oxigen` `use oxigen::prelude::*` and call the `run` method over a `GeneticExecution` instance overwriting the default hyperparameters and functions folllowing your needs:

```
let n_queens: u8 = std::env::args()
    .nth(1)
    .expect("Enter a number between 4 and 255 as argument")
    .parse()
    .expect("Enter a number between 4 and 255 as argument");
let progress_log = File::create("progress.csv").expect("Error creating progress log file");
let population_log =
    File::create("population.txt").expect("Error creating population log file");
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

For more information visit the documentation.


## Building

To build oxigen, use `cargo` like for any Rust project:

* `cargo build` to build in debug mode.
* `cargo build --release` to build with optimizations.

To run benchmarks, you will need a nightly Rust compiler. Uncomment the lines `// #![feature(test)]` and `// mod tests;` from `lib.rs` and then bechmarks can be run using `cargo bench`.


## Benchmarks

The following benchmarks have been created to measure the genetic algorithm functions performance:

```
running 14 tests
test tests::bench_cross                  ... bench:      33,192 ns/iter (+/- 10,847)
test tests::bench_cup                    ... bench:     319,823 ns/iter (+/- 91,331)
test tests::bench_fitness                ... bench:     547,453 ns/iter (+/- 111,564)
test tests::bench_fitness_age            ... bench:      44,959 ns/iter (+/- 5,206)
test tests::bench_get_fitness            ... bench:      17,015 ns/iter (+/- 2,185)
test tests::bench_get_solutions          ... bench:      32,714 ns/iter (+/- 3,604)
test tests::bench_mutation               ... bench:           7 ns/iter (+/- 0)
test tests::bench_not_cached_fitness     ... bench:     532,210 ns/iter (+/- 97,308)
test tests::bench_not_cached_fitness_age ... bench:     532,407 ns/iter (+/- 81,423)
test tests::bench_roulette               ... bench:     281,544 ns/iter (+/- 10,570)
test tests::bench_sort_population        ... bench:       1,485 ns/iter (+/- 58)
test tests::bench_survival_pressure      ... bench:          20 ns/iter (+/- 0)
test tests::bench_tournaments            ... bench:     964,301 ns/iter (+/- 127,458)
test tests::bench_update_progress        ... bench:       5,779 ns/iter (+/- 286)

test result: ok. 0 passed; 0 failed; 0 ignored; 14 measured; 0 filtered out
```

The difference of performance among the different fitness benchmarks have the following explanations:

 * `bench_fitness` measures the performance of a cached execution cleaning the fitnesses after each bench iteration. This cleaning is the reason of being a bit slower than not cached benchmarks.
 * `bench_fitness_age` measures the performance with fitness cached in all bench iterations, so it is very much faster.
 * Not cached benchmarks measure the performance of not cached executions, with 1 generation individuals in the last case, so the performance is similar but a bit slower for the benchmark that must apply age unfitness.


 The `bench_tournaments` is slower than cup and roulette because it has been configured with `population_size / 2` tournaments of the same size of individuals. In practice, a configuration like this is only used in the last generations of the algorithm where the selection rate is high, being very much faster in the previous generations.


## Contributing

Contributions are absolutely, positively welcome and encouraged! Contributions
come in many forms. You could:

  1. Submit a feature request or bug report as an [issue](https://github.com/Martin1887/oxigen/issues).
  2. Ask for improved documentation as an [issue](https://github.com/Martin1887/oxigen/issues).
  3. Comment on issues that require
     feedback.
  4. Contribute code via [pull requests](https://github.com/Martin1887/oxigen/pulls).

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

## License

oxigen is licensed under either of the following, at your option:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
