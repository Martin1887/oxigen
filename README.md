# oxigen

[![Build Status](https://travis-ci.com/Martin1887/oxigen.svg?branch=master)](https://travis-ci.com/Martin1887/oxigen)
[![Current Crates.io Version](https://img.shields.io/crates/v/oxigen.svg)](https://crates.io/crates/oxigen)

Oxigen is a parallel genetic algorithm library implemented in Rust. The name comes from the merge of `OXI`dación (Rust translated to Spanish) and `GEN`etic.

Oxigen provides the following features:

* Fast and parallel genetic algorithm implementation (it solves the N Queens problem for N=255 in few seconds). For benchmarks view benchmarks section of this file.
* Customizable mutation and selection rates with constant, linear and Quadratic functions according to generations built-in (you can implement your own functions via the `MutationRate` and `SelectionRate` traits).
* Customizable age unfitness of individuals, with no unfitness, linear and Quadratic unfitness with threshold according to generations of the individual built-in (you can implement your own age functions via the `Age` trait).
* Accumulated `Roulette`, `Tournaments` and `Cup` built-in selection functions (you can implement your own selection functions via the `Selection` trait).
* `SingleCrossPoint`, `MultiCrossPoint`, `UniformCross`, and `UniformPartiallyMatched` built-in crossover functions (you can implement your own crossover function via the `Crossover` trait).
* Many built-in survival pressure functions. You can implement your own survival pressure functions via the `SurvivalPressure` trait.
* `Niches` built-in `PopulationRefitness` function. You can implement your own population refitness functions via the `PopulationRefitness` trait.
* `SolutionFound`, `Generation` and `Progress` and more built-in stop criteria (you can implement your own stop criteria via the `StopCriterion` trait).
* `Genotype` trait to define the genotype of your genetic algorithm. Whatever struct can implement the `Genotype` trait under the following restrictions:
    - It has a `iter` function that returns a `use std::slice::Iter` over its genes.
    - It has a `into_iter` function that consumes the individual and returns a `use std::vec::IntoIter` over its genes.
    - It has a `from_iter` function that set the genes from an iterator.
    - It implements `Display`, `Clone`, `Send` and `Sync`.
    - It has functions to `generate` a random individual, to `mutate` an individual, to get the `fitness` of an individual and to know if and individual `is_solution` of the problem.
* Individual's fitness is cached to not do unnecessary recomputations (this can be disabled with `.cache_fitness(false)` if your fitness function is stochastic and so you need to recompute fitness in each generation).
* Progress statistics can be configured to be printed every certain number of generations to a file.
* Population individuals with their fitnesses can be configured to be printed every certain number of generations to a file.
* Specific initial individuals can be inserted in the genetic algorithm execution.
* Genetic executions can be resumed using the population of the last generation as initial population.
* Coevolution is possible executing little genetic algorithm re-executions inside the fitness function.

## Differences between 2 and 1 versions:
* Oxigen 2 is more flexible because any `struct` with a `Vec` inside can implement `Genotype`. In 1 versions this was not possible because `Genotype` had to implement `FromIterator`. In 2 versions a `from_iter` function has been added instead.
* Oxigen 2 fix the issue #3 ('Cuadratic' has been replaced by 'Quadratic' in built-in enums). This has not been fixed in 1 versions to not break the interface.
* The `fix` function in `Genotype` returns a boolean to specify if the individual has been changed to recompute its fitness.
* The number of solutions gotten in each generation is now the number of different solutions using the new `distance` function of `Genotype`.
* The `u16` type has been changed by `usize` in `StopCriterion`, `MutationRate` and `SelectionRate` traits.
* `PopulationRefitness` trait has been added to optionally refit the individuals of the population comparing them to the other individuals. `Niches` built-in `PopulationRefitness` function has been added.
* The `SurvivalPressure` trait has been redefined and now it kills the individuals instead of returning the indexes to remove. It also receives a list with the pairs of parents and children of the generation.
* Many `SurvivalPressure` built-in functions have been added, like `Overpouplation`, `CompetitiveOverpopulation`, `DeterministicOverpopulation`, `ChildrenFightParents`, `ChildrenFightMostsimilar`, etc.
* The two previous additions allow to search different solutions in different search space areas in order to avoid local suboptimal solutions and find different solutions.
* Other minor improvements.

## New in version 2.1:
The optional feature `global_cache` adds a `HashMap` saving the evaluation of each individual in the full execution.

This cache is useful when the evaluation of each individual is expensive, and it complements the individual-based cache already existing in previous versions (if an individual has been evaluated it is not reevaluated unless `cache_fitness` is `false`). In other words, this global cache saves the evaluation of new individuals that are equal to another individual that was evaluated before.

Note that the global cache is not always better, since if the fitness function is cheap the cost of getting and inserting into the cache can be more expensive than it. Take also into account the increasing of RAM usage of the global cache.

To enable the global cache add the feature `global_cache` in the Cargo.toml of your project and set to `true` the `cache_fitness` (always `true` by default) and `global_cache` (`true` by default when the `global_cache` is enabled) properties of your `GeneticExeution`. Example of Cargo.toml:
```
[dependencies]
oxigen = { version="^2.1", features=["global_cache"] }
```


## Usage

In your `Cargo.toml` file add the `oxigen` dependency:

```
[dependencies]
oxigen = "^2"
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
    .environment(n_queens as u8)
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
    .age_function(Box::new(AgeFunctions::Quadratic(
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
    .environment(n_queens as u8)
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
    .age_function(Box::new(AgeFunctions::Quadratic(
        AgeThreshold(50),
        AgeSlope(1_f64),
    )))
    .stop_criterion(Box::new(StopCriteria::Generation(10000)))
    .run();

let (solutions, generation, progress, _population) = GeneticExecution::<u8, QueensBoard>::new()
    .population_size(population_size)
    .environment(n_queens as u8)
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
    .age_function(Box::new(AgeFunctions::Quadratic(
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
running 25 tests
test benchmarks::bench_cross_multi_point_255inds                                                           ... bench:     348,371 ns/iter (+/- 10,506)
test benchmarks::bench_cross_single_point_255inds                                                          ... bench:     113,986 ns/iter (+/- 10,657)
test benchmarks::bench_cross_uniform_255inds                                                               ... bench:      88,426 ns/iter (+/- 2,302)
test benchmarks::bench_distance_255                                                                        ... bench:      20,715 ns/iter (+/- 1,648)
test benchmarks::bench_fitness_1024inds                                                                    ... bench:     377,344 ns/iter (+/- 8,617)
test benchmarks::bench_fitness_age_1024inds                                                                ... bench:      31,360 ns/iter (+/- 1,204)
test benchmarks::bench_fitness_age_not_cached_1024inds                                                     ... bench:     395,056 ns/iter (+/- 24,407)
test benchmarks::bench_fitness_global_cache_1024inds                                                       ... bench:     340,087 ns/iter (+/- 28,889)
test benchmarks::bench_fitness_not_cached_1024inds                                                         ... bench:     373,966 ns/iter (+/- 60,244)
test benchmarks::bench_get_fitnesses_1024inds                                                              ... bench:      18,951 ns/iter (+/- 868)
test benchmarks::bench_get_solutions_1024inds                                                              ... bench:      30,133 ns/iter (+/- 1,612)
test benchmarks::bench_mutation_1024inds                                                                   ... bench:          13 ns/iter (+/- 0)
test benchmarks::bench_selection_cup_255inds                                                               ... bench:     344,873 ns/iter (+/- 40,519)
test benchmarks::bench_selection_roulette_256inds                                                          ... bench:     140,994 ns/iter (+/- 1,294)
test benchmarks::bench_selection_tournaments_256inds                                                       ... bench:     420,272 ns/iter (+/- 49,178)
test benchmarks::bench_survival_pressure_children_fight_most_similar_255inds                               ... bench:  14,948,961 ns/iter (+/- 989,864)
test benchmarks::bench_survival_pressure_children_fight_parents_255inds                                    ... bench:     108,691 ns/iter (+/- 157)
test benchmarks::bench_survival_pressure_children_replace_most_similar_255inds                             ... bench:  14,935,080 ns/iter (+/- 1,221,611)
test benchmarks::bench_survival_pressure_children_replace_parents_255inds                                  ... bench:     167,392 ns/iter (+/- 15,839)
test benchmarks::bench_survival_pressure_children_replace_parents_and_the_rest_most_similar_255inds        ... bench: 737,347,573 ns/iter (+/- 16,773,890)
test benchmarks::bench_survival_pressure_children_replace_parents_and_the_rest_random_most_similar_255inds ... bench:   7,757,258 ns/iter (+/- 612,047)
test benchmarks::bench_survival_pressure_competitive_overpopulation_255inds                                ... bench:  10,727,219 ns/iter (+/- 770,681)
test benchmarks::bench_survival_pressure_deterministic_overpopulation_255inds                              ... bench:     173,745 ns/iter (+/- 609)
test benchmarks::bench_survival_pressure_overpopulation_255inds                                            ... bench:  10,710,336 ns/iter (+/- 657,369)
test benchmarks::bench_survival_pressure_worst_255inds                                                     ... bench:      20,318 ns/iter (+/- 298)
test benchmarks::bench_update_progress_1024inds                                                            ... bench:       7,424 ns/iter (+/- 273)
```

These benchmarks have been executed in a Intel Core i7 6700K with 16 GB of DDR4
and a 512 GB Samsung 950 Pro NVMe SSD in ext4 format in Fedora 30
with Linux kernel 5.2.9.

The difference of performance among the different fitness benchmarks have the following explanations:

 * `bench_fitness` measures the performance of a cached execution cleaning the fitnesses after each bench iteration. This cleaning is the reason of being a bit slower than not cached benchmarks.
 * `bench_fitness_age` measures the performance with fitness cached in all bench iterations, so it is very much faster.
 * Not cached benchmarks measure the performance of not cached executions, with 1 generation individuals in the last case, so the performance is similar but a bit slower for the benchmark that must apply age unfitness.
 * The `children_fight_most_similar` and `children_replace_most_similar` functions have to call the distance function `c * p` times, where `c` is the number of children and `p` is the size of the population (255 and 1024 respectively in the benchmarks).
 * The `overpopulation` and `competitive_overpopulation` functions are similaer to `children_replace_most_similar` and `children_fight_most_similar` except to they are compared only with `m` individuals of the population (`m` is bigger than the number of children and smaller than the population size, 768 in the benchmarks). Therefore, 3/4 of the comparisons are done in these benchmarks compared to `children_replace_most_similar` and `children_fight_most_similar`.
 * `children_replace_parents_and_the_rest_random_most_similar` is similar to `children_replace_parents` but, after it, random individuals are chosen to fight against the most similar individual in the population until the population size is the original population size. This means between 0 and 254 times random chosing and distance cmoputation over the entire population in function of the repeated parents in each generation.
 * `children_replace_parents_and_the_rest_most_similar` is like the previous function but it searchs the pairs of most similar individuals in the population, which means p<sup>2</sup> distance function calls (2<sup>20</sup> in the benchmark).


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
Pozo, M.M. "Oxigen: Fast, parallel, extensible and adaptable genetic algorithm library written in Rust".


### Bibtex
```tex
@misc{
  title={Oxigen: Fast, parallel, extensible and adaptable genetic algorithm library written in Rust},
  author={Pozo M.M.},
  howpublised = "\url{https://github.com/Martin1887/oxigen}"
}
```

## License

oxigen is licensed under Mozilla Public License 2.0.