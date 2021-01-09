# oxigen

[![Build Status](https://travis-ci.com/Martin1887/oxigen.svg?branch=master)](https://travis-ci.com/Martin1887/oxigen)
[![Current Crates.io Version](https://img.shields.io/crates/v/oxigen.svg)](https://crates.io/crates/oxigen)

Oxigen is a parallel genetic algorithm framework implemented in Rust. The name comes from the merge of `OXI`dación (Rust translated to Spanish) and `GEN`etic.

The changes introduced in each version can be found in [CHANGELOG.md](CHANGELOG.md).

To migrate between major version check the migration guide ([MIGRATE.md](MIGRATE.md)).

Oxigen provides the following features:

* Fast and parallel genetic algorithm implementation (it solves the N Queens problem for N=255 in few seconds). For benchmarks view benchmarks section of this file.
* Customizable mutation and selection rates with constant, linear and quadratic functions according to generations built-in (you can implement your own functions via the `MutationRate` and `SelectionRate` traits).
* Customizable age unfitness of individuals, with no unfitness, linear and quadratic unfitness with threshold according to generations of the individual built-in (you can implement your own age functions via the `Age` trait).
* Accumulated `Roulette`, `Tournaments` and `Cup` built-in selection functions (you can implement your own selection functions via the `Selection` trait).
* `SingleCrossPoint`, `MultiCrossPoint`, `UniformCross`, `UniformPartiallyMatched`, and `PartiallyMatched` built-in crossover functions (you can implement your own crossover function via the `Crossover` trait).
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

## Optional feature `global_cache`
The optional feature `global_cache` adds a `HashMap` saving the evaluation of each individual in the full execution.

This cache is useful when the evaluation of each individual is expensive, and it complements the individual-based cache already existing in previous versions (if an individual has been evaluated it is not reevaluated unless `cache_fitness` is `false`). In other words, this global cache saves the evaluation of new individuals that are equal to another individual that was evaluated before.

Note that the global cache is not always better, since if the fitness function is cheap the cost of getting and inserting into the cache can be more expensive than it. Take also into account the increase of RAM usage of the global cache.

To enable the global cache add the feature `global_cache` in the Cargo.toml of your project and set to `true` the `cache_fitness` (always `true` by default) and `global_cache` (`true` by default when the `global_cache` is enabled) properties of your `GeneticExecution`. Example of Cargo.toml:
```
[dependencies]
oxigen = { version="2.1", features=["global_cache"] }
```


## Usage

In your `Cargo.toml` file add the `oxigen` dependency. Oxigen follows the [semver](https://semver.org/) specification for the names of the versions, so major version changes will never break the existent API and the last version should always be used. If a minimum version is required specify that minor version to include that version and all minor versions bigger than it.

```
[dependencies]
oxigen = "2"
```

To use `oxigen` `use oxigen::prelude::*` and call the `run` method over a `GeneticExecution` instance overwriting the default hyperparameters and functions following your needs:

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

Since version 1.1.0, genetic algorithm executions return the population of the last generation and new genetic executions accept a initial population. This permits to resume previous executions and it also enables coevolution, since little genetic algorithm re-executions can be launched in the fitness function.

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

To run benchmarks, you will need a nightly Rust compiler. Uncomment the lines `// #![feature(test)]` and `// mod benchmarks;` from `lib.rs` and then benchmarks can be run using `cargo bench --jobs 1 --all-features`.


## Benchmarks

The following benchmarks have been created to measure the genetic algorithm functions performance:

```
running 29 tests
test benchmarks::bench_cross_multi_point_255inds                                                           ... bench:     895,332 ns/iter (+/- 34,409)
test benchmarks::bench_cross_single_point_255inds                                                          ... bench:     227,517 ns/iter (+/- 4,802)
test benchmarks::bench_cross_uniform_255inds                                                               ... bench:      73,370 ns/iter (+/- 9,106)
test benchmarks::bench_distance_255                                                                        ... bench:      41,669 ns/iter (+/- 45)
test benchmarks::bench_fitness_1024inds                                                                    ... bench:      14,260 ns/iter (+/- 3,789)
test benchmarks::bench_fitness_age_1024inds                                                                ... bench:      32,495 ns/iter (+/- 5,705)
test benchmarks::bench_fitness_age_not_cached_1024inds                                                     ... bench:     581,263 ns/iter (+/- 3,988)
test benchmarks::bench_fitness_global_cache_1024inds                                                       ... bench:     343,314 ns/iter (+/- 25,763)
test benchmarks::bench_fitness_not_cached_1024inds                                                         ... bench:     554,870 ns/iter (+/- 32,916)
test benchmarks::bench_generation_run_tournaments_1024inds                                                 ... bench:   4,202,844 ns/iter (+/- 111,604)
test benchmarks::bench_get_fitnesses_1024inds                                                              ... bench:         777 ns/iter (+/- 17)
test benchmarks::bench_get_solutions_1024inds                                                              ... bench:       2,126 ns/iter (+/- 7)
test benchmarks::bench_mutation_1024inds                                                                   ... bench:   1,553,265 ns/iter (+/- 23,022)
test benchmarks::bench_refitness_niches_1024inds                                                           ... bench:      29,616 ns/iter (+/- 783)
test benchmarks::bench_refitness_none_1024inds                                                             ... bench:      29,756 ns/iter (+/- 3,576)
test benchmarks::bench_selection_cup_255inds                                                               ... bench:     357,611 ns/iter (+/- 37,254)
test benchmarks::bench_selection_roulette_256inds                                                          ... bench:     141,654 ns/iter (+/- 1,338)
test benchmarks::bench_selection_tournaments_256inds                                                       ... bench:     616,907 ns/iter (+/- 50,645)
test benchmarks::bench_survival_pressure_children_fight_most_similar_255inds                               ... bench:  17,748,382 ns/iter (+/- 762,602)
test benchmarks::bench_survival_pressure_children_fight_parents_255inds                                    ... bench:     139,405 ns/iter (+/- 2,267)
test benchmarks::bench_survival_pressure_children_replace_most_similar_255inds                             ... bench:  17,716,416 ns/iter (+/- 739,662)
test benchmarks::bench_survival_pressure_children_replace_parents_255inds                                  ... bench:     202,788 ns/iter (+/- 18,250)
test benchmarks::bench_survival_pressure_children_replace_parents_and_the_rest_most_similar_255inds        ... bench: 1,387,504,266 ns/iter (+/- 45,914,604)
test benchmarks::bench_survival_pressure_children_replace_parents_and_the_rest_random_most_similar_255inds ... bench:   9,389,378 ns/iter (+/- 1,224,136)
test benchmarks::bench_survival_pressure_competitive_overpopulation_255inds                                ... bench:  12,803,024 ns/iter (+/- 1,946,079)
test benchmarks::bench_survival_pressure_deterministic_overpopulation_255inds                              ... bench:     220,667 ns/iter (+/- 2,790)
test benchmarks::bench_survival_pressure_overpopulation_255inds                                            ... bench:  12,243,512 ns/iter (+/- 726,154)
test benchmarks::bench_survival_pressure_worst_255inds                                                     ... bench:      20,339 ns/iter (+/- 1,113)
test benchmarks::bench_update_progress_1024inds                                                            ... bench:       7,595 ns/iter (+/- 378)
```

These benchmarks have been executed in a Intel Core i7 6700K with 16 GB of DDR4
and a 1024 GB Samsung 970 Evo Plus NVMe SSD in ext4 format in Fedora 33
with Linux kernel 5.9.16.

The difference of performance among the different fitness benchmarks have the following explanations:

 * `bench_fitness` measures the performance of a cached execution without cleaning the fitnesses after each bench iteration. This cleaning was not done in previous versions of this README and so it was higher.
 * `bench_mutation` was very much faster in previous versions of this README because an error in the benchmark (empty population).
 * `bench_fitness_age` measures the performance with fitness cached in all bench iterations, so it is slightly slower.
 * Not cached benchmarks measure the performance of not cached executions, with 1 generation individuals in the last case, so the performance is similar but a bit slower for the benchmark that must apply age unfitness.
 * The `children_fight_most_similar` and `children_replace_most_similar` functions have to call the distance function `c * p` times, where `c` is the number of children and `p` is the size of the population (255 and 1024 respectively in the benchmarks).
 * The `overpopulation` and `competitive_overpopulation` functions are similar to `children_replace_most_similar` and `children_fight_most_similar` except to they are compared only with `m` individuals of the population (`m` is bigger than the number of children and smaller than the population size, 768 in the benchmarks). Therefore, 3/4 of the comparisons are done in these benchmarks compared to `children_replace_most_similar` and `children_fight_most_similar`.
 * `children_replace_parents_and_the_rest_random_most_similar` is similar to `children_replace_parents` but, after it, random individuals are chosen to fight against the most similar individual in the population until the population size is the original population size. This means between 0 and 254 times random choosing and distance computation over the entire population in function of the repeated parents in each generation.
 * `children_replace_parents_and_the_rest_most_similar` is like the previous function but it searches the pairs of most similar individuals in the population, which means p<sup>2</sup> distance function calls (2<sup>20</sup> in the benchmark).


## Contributing

Contributions are absolutely, positively welcome and encouraged! Contributions come in many forms. You could:

  1. Submit a feature request or bug report as an [issue](https://github.com/Martin1887/oxigen/issues).
  2. Ask for improved documentation as an [issue](https://github.com/Martin1887/oxigen/issues).
  3. Comment on issues that require feedback.
  4. Contribute code via [pull requests](https://github.com/Martin1887/oxigen/pulls).

We aim to keep Oxigen's code quality at the highest level. This means that any code you contribute must be:

  * **Commented:** Public items _must_ be commented.
  * **Documented:** Exposed items _must_ have rustdoc comments with
    examples, if applicable.
  * **Styled:** Your code should be `rustfmt`'d when possible.
  * **Simple:** Your code should accomplish its task as simply and
     idiomatically as possible.
  * **Tested:** You should add (and pass) convincing tests for any functionality you add when it is possible.
  * **Focused:** Your code should do what it's supposed to do and nothing more.

Note that unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in Oxigen by you shall be licensed under Mozilla Public License 2.0.


## Reference
Pozo, Martín "Oxigen: Fast, parallel, extensible and adaptable genetic algorithms framework written in Rust".


### Bibtex
```tex
@misc{
  title={Oxigen: Fast, parallel, extensible and adaptable genetic algorithms framework written in Rust},
  author={Pozo, Martín},
  howpublised = "\url{https://github.com/Martin1887/oxigen}"
}
```

## License

Oxigen is licensed under Mozilla Public License 2.0.
