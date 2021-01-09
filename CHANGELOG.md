# Oxigen Changelog

## 2.2.0 - 2021/01/09
- Better population refitness taking into account the age effect but not the refitness of previous generations (in previous versions the original fitness was taken instead).
- Fix niches formula: `1.0 - (d / sigma.0).powf(alfa.0)` instead of `(1.0 - (d / sigma.0)).powf(alfa.0)`.
- Historical of solutions instead of checking only the current generation.
- Improve performance of getting fitnesses and solutions using a normal iterator instead of a parallel one.

### 2.1.1 - 2021/01/09
- Fix niches formula: `1.0 - (d / sigma.0).powf(alfa.0)` instead of `(1.0 - (d / sigma.0)).powf(alfa.0)`.

## 2.1.0 - 2020/01/19
- Optional global cache as `global_cache` feature.

### 2.0.3 - 2021/01/09
- Fix niches formula: `1.0 - (d / sigma.0).powf(alfa.0)` instead of `(1.0 - (d / sigma.0)).powf(alfa.0)`.

### 2.0.2 - 2019/10/18
- Better default `distance` function taking into account individuals of different length.

### 2.0.1 - 2019/10/18
- Make public `ind` and `fitness` attributes of `IndWithFitness` struct.

# 2.0.0 - 2019/09/28
- More flexibility allowing any struct with a `Vec` inside can implement `Genotype`. In 1.x versions this was not possible because Genotype had to implement FromIterator. In 2 versions a from_iter function has been added instead.
- Oxigen 2 fix the issue #3 ('Cuadratic' has been replaced by 'Quadratic' in built-in enums). This has not been fixed in 1 versions to not break the interface.
- The `fix` function in `Genotype` returns a boolean to specify if the individual has been changed to recompute its fitness.
- The number of solutions gotten in each generation is now the number of different solutions using the new `distance` function of `Genotype`.
- The `u16` type has been changed to `usize` in `StopCriterion`, `MutationRate` and `SelectionRate` traits.
- `PopulationRefitness` trait has been added to optionally refit the individuals of the population comparing them to the other individuals. `Niches` built-in `PopulationRefitness` function has been added.
- The `SurvivalPressure` trait has been redefined and now it kills the individuals instead of returning the indexes to remove. It also receives a list with the pairs of parents and children of the generation.
- Many `SurvivalPressure` built-in functions have been added, like `Overpopulation`, `CompetitiveOverpopulation`, `DeterministicOverpopulation`, `ChildrenFightParents`, `ChildrenFightMostSimilar`, etc.
- The two previous additions allow to search different solutions in different search space areas in order to avoid local suboptimal solutions and find different solutions.
- Other minor improvements.

## 1.5.0 - 2019/08/24
- Multipoint and uniform cross functions.

### 1.4.3 - 2019/07/22
- Fix error in tournaments that did select individuals from 0 to the selection rate instead of the entire population.

### 1.4.2 - 2019/01/13
- Fix that the age unfitness was not applied to the fitness.

### 1.4.1 - 2019/01/06
- Fix that cross points cannot be 0, since it generates children identical to parents.

## 1.4.0 - 2019/01/01
- Fix that the age was not incremented in each generation.
- The progress is NaN at the first generation also in `MutationRate` and `SelectionRate`.

## 1.3.0 - 2018/12/27
- Fix empty fitnesses at the first generation sent to `StopCriterion`, `MutationRate` and `SelectionRate`.
- Add `GenerationAndProgress`, `MaxFitness`, `MinFitness` and `AvgFitness` built-in stop criteria.

### 1.2.1 - 2018/12/27
- Fix survival pressure `Worst`, that was reducing population size to a half.

## 1.2.0 - 2018/12/26
- Add `fix` function to `Genotype` to allow to repair individuals to satisfy restrictions.
- `SolutionsFound` stop criterion added to specify the desired number of solutions.
- `Progress` stop criterion fixed (it was stopping always at iteration 0).
- Improved populations output.

## 1.1.0 - 2018/08/20
- Permit initial individuals enabling to resume previous executions and coevolution.

# 1.0 - 2018/08/20
- First version.