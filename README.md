# oxigen

oxigen is a parallel genetic algorithm library implemented in Rust. The name comes from the merge of `OXI`dación (Rust translated to Spanish) and `GEN`etic.

oxigen provides the following features:

* Fast and parallel genetic algorithm implementation (it solves the N Queens problem in a few seconds).
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
  * **Tested:** You must add (and pass) convincing tests for any functionality you add.
  * **Focused:** Your code should do what it's supposed to do and nothing more.

Note that unless you
explicitly state otherwise, any contribution intentionally submitted for
inclusion in oxigen by you shall be dual licensed under the MIT License and
Apache License, Version 2.0, without any additional terms or conditions.

## License

oxigen is licensed under either of the following, at your option:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
