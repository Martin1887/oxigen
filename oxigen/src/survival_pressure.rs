//! This module contains the definition of the SurvivalPressure trait and the provided
//! survival_pressure functions.

use genotype::Genotype;
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::PartialEq;
use std::sync::mpsc::channel;
use IndWithFitness;
use SurvivalPressureFunctions::*;

/// This struct represents a `m` value, that shall be lower than the
/// population size and greater than the number of generated individuals
pub struct M(usize);
impl M {
    /// Creates a M object.
    ///
    /// # Panics
    /// If m is not bigger than the number of children and lower than the
    /// population size.
    pub fn new(m: usize, n_children: usize, population_size: usize) -> M {
        if m > n_children && m < population_size {
            M { 0: m }
        } else {
            panic!(
                "m shall be greater than the number of children and lower than the population size"
            )
        }
    }
}

/// This struct represents a reproduction step allocating the indexes of the
/// parents and children in the population.
pub struct Reproduction {
    pub parents: (usize, usize),
    pub children: (usize, usize),
}

/// This trait defines the kill function used to remove individuals at the end of a generation.
pub trait SurvivalPressure<T: PartialEq + Send + Sync, G: Genotype<T>>: Send + Sync {
    /// Removes individuals according to the population size; the population and
    /// the fitness of the population; and the parents and children relationships.
    /// Population is not sorted in any particular order, but children are after `population_size`.
    /// Note that in oxigen 1.x versions this function returned the indexes to be
    /// deleted without modifying the population, not it directly removes the individuals.
    fn kill(
        &self,
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    );
}

/// Provided survival pressure functions.
pub enum SurvivalPressureFunctions {
    /// Kill worst individuals until reach the population size.
    Worst,
    /// Kill the most similar individual to each child.
    ChildrenReplaceMostSimilar,
    /// Kill individuals that have been crossed. Note that since the same parents can produce
    /// many different children (they are selected to cross several times) the population size
    /// can be increased using this function.
    ChildrenReplaceParents,
    /// Kill individuals that have been crossed and after that the worst individuals until reach
    /// the initial population size.
    ChildrenReplaceParentsAndTheRestWorst,
    /// Kill individuals that have been crossed and after that random individuals until reach
    /// the initial population size.
    ChildrenReplaceParentsAndTheRestRandomly,
    /// Kill individuals that have been crossed and after that randomly select
    /// an individual to fight to the the most similar to it
    /// until reach the initial population size. Note that this function is slow using the
    /// default Genotype similarity function because genes are compared one by one.
    ChildrenReplaceParentsAndTheRestRandomMostSimilar,
    /// Kill individuals that have been crossed and after that the most similar individuals
    /// until reach the initial population size. Note that this function is slow using the
    /// default Genotype similarity function because genes are compared one by one.
    ChildrenReplaceParentsAndTheRestMostSimilar,
    /// Kill individuals that have been crossed and after that the oldest individuals until reach
    /// the initial population size.
    /// /// Note that this function is slow because each individual similarity is computed across the
    /// rest of individuals of the population.
    ChildrenReplaceParentsAndTheRestOldest,
    /// Each child fight with the most similar individual in the population.
    ChildrenFightMostSimilar,
    /// Each child fight with a random parent in a parricide battle for survival.
    /// Note that because the same parents can produce many different children (they are selected
    /// to cross several times) the population size can be increased using this function.
    ChildrenFightParents,
    /// Each child fight with a random parent in a parricide battle for survival and after that
    /// the worst individuals until reach the initial population size.
    ChildrenFightParentsAndTheRestWorst,
    /// Each child fight with a random parent in a parricide battle for survival and after that
    /// random individuals until reach the initial population size.
    ChildrenFightParentsAndTheRestRandomly,
    /// Each child fight with a random parent in a parricide battle for survival and after that
    /// randomly select an individual to fight to the the most similar to it
    /// until reach the initial population size.
    /// Note that this function is slow because each individual similarity is computed across the
    /// rest of individuals of the population.
    ChildrenFightParentsAndTheRestRandomMostSimilar,
    /// Each child fight with a random parent in a parricide battle for survival and after that
    /// the most similar individuals until reach the initial population size.
    /// Note that this function is slow because each individual similarity is computed across the
    /// rest of individuals of the population.
    ChildrenFightParentsAndTheRestMostSimilar,
    /// Each child fight with a random parent in a parricide battle for survival and after that
    /// the oldest individuals until reach the initial population size.
    ChildrenFightParentsAndTheRestOldest,
    /// `m` individuals are randomly selected from the population (note that n < m < population_size
    /// where n is the number of generated individuals shall be true) and the most similar ones to the
    /// children are killed one by one.
    Overpopulation(M),
    /// Like `OverPopulation` but the existing individuals in the population that are most
    /// similar to each children fight with the corresponding children and the best survive.
    CompetitiveOverpopulation(M),
    /// Like `ChildrenFightParents` but instead of random fight between one child with one parent,
    /// the tuple of most similar parent-child is selected to fight.
    DeterministicOverpopulation,
}

impl<T: PartialEq + Send + Sync, G: Genotype<T>> SurvivalPressure<T, G>
    for SurvivalPressureFunctions
{
    fn kill(
        &self,
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        match self {
            Worst => {
                Self::remove_worst(population_size, population);
            }
            ChildrenReplaceMostSimilar => {
                Self::children_replace_most_similar(population, parents_children, population_size);
            }
            ChildrenReplaceParents => {
                Self::children_replace_parents(population, parents_children);
            }
            ChildrenReplaceParentsAndTheRestWorst => {
                Self::children_replace_parents(population, parents_children);
                Self::remove_worst(population_size, population);
            }
            ChildrenReplaceParentsAndTheRestRandomly => {
                Self::children_replace_parents(population, parents_children);
                Self::remove_randomly(population_size, population);
            }
            ChildrenReplaceParentsAndTheRestRandomMostSimilar => {
                Self::children_replace_parents(population, parents_children);
                Self::random_most_similar(population_size, population);
            }
            ChildrenReplaceParentsAndTheRestMostSimilar => {
                Self::children_replace_parents(population, parents_children);
                Self::most_similar(population_size, population);
            }
            ChildrenReplaceParentsAndTheRestOldest => {
                Self::children_replace_parents(population, parents_children);
                Self::remove_oldest(population_size, population);
            }
            ChildrenFightMostSimilar => {
                Self::children_fight_most_similar(population, parents_children, population_size);
            }
            ChildrenFightParents => {
                Self::children_fight_parents(population, parents_children);
            }
            ChildrenFightParentsAndTheRestWorst => {
                Self::children_fight_parents(population, parents_children);
                Self::remove_worst(population_size, population);
            }
            ChildrenFightParentsAndTheRestRandomly => {
                Self::children_fight_parents(population, parents_children);
                Self::remove_randomly(population_size, population);
            }
            ChildrenFightParentsAndTheRestRandomMostSimilar => {
                Self::children_fight_parents(population, parents_children);
                Self::random_most_similar(population_size, population);
            }
            ChildrenFightParentsAndTheRestMostSimilar => {
                Self::children_fight_parents(population, parents_children);
                Self::most_similar(population_size, population);
            }
            ChildrenFightParentsAndTheRestOldest => {
                Self::children_fight_parents(population, parents_children);
                Self::remove_oldest(population_size, population);
            }
            Overpopulation(m) => {
                Self::overpopulation(m.0, population_size, population, parents_children);
            }
            CompetitiveOverpopulation(m) => {
                Self::competitive_overpopulation(
                    m.0,
                    population_size,
                    population,
                    parents_children,
                );
            }
            DeterministicOverpopulation => {
                Self::deterministic_overpopulation(population, parents_children);
            }
        }
    }
}

impl SurvivalPressureFunctions {
    fn remove_worst<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
    ) {
        sort_population(population);
        let mut i = population.len();
        while i > population_size {
            population.pop();
            i -= 1;
        }
    }

    fn remove_randomly<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
    ) {
        let mut rgen = SmallRng::from_entropy();
        let mut i = population.len();
        while i > population_size {
            let chosen = rgen.sample(Uniform::from(0..i));
            population.remove(chosen);
            i -= 1;
        }
    }
    fn remove_oldest<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
    ) {
        sort_population_by_age(population);
        let mut i = population.len();
        while i > population_size {
            population.pop();
            i -= 1;
        }
    }

    fn children_replace_most_similar<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
        population_size: usize,
    ) {
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        for repr in parents_children.iter() {
            let most_similar_0 = population
                .par_iter()
                .enumerate()
                // Children start at population_size (added at the end)
                .filter(|(i, _el)| *i < population_size && !killed.contains(i))
                .map(|(i, el)| (i, population[repr.children.0].ind.distance(&el.ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            killed.push(most_similar_0);

            let most_similar_1 = population
                .par_iter()
                .enumerate()
                // Children start at population_size (added at the end)
                .filter(|(i, _el)| *i < population_size && !killed.contains(i))
                .map(|(i, el)| (i, population[repr.children.1].ind.distance(&el.ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            killed.push(most_similar_1);
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn children_replace_parents<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        let (sender, receiver) = channel();
        parents_children
            .par_iter()
            .for_each_with(sender, |s, repr| {
                s.send(repr.parents).unwrap();
            });
        for parents in receiver {
            killed.push(parents.0);
            killed.push(parents.1);
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        killed.dedup();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn children_fight_most_similar<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
        population_size: usize,
    ) {
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        for repr in parents_children.iter() {
            let most_similar_0 = population
                .par_iter()
                .enumerate()
                // Children start at population_size (added at the end)
                .filter(|(i, _el)| *i < population_size && !killed.contains(i))
                .map(|(i, el)| (i, population[repr.children.0].ind.distance(&el.ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            if population[most_similar_0].fitness.unwrap().fitness
                > population[repr.children.0].fitness.unwrap().fitness
            {
                killed.push(repr.children.0);
            } else {
                killed.push(most_similar_0);
            }

            let most_similar_1 = population
                .par_iter()
                .enumerate()
                // Children start at population_size (added at the end)
                .filter(|(i, _el)| *i < population_size && !killed.contains(i))
                .map(|(i, el)| (i, population[repr.children.1].ind.distance(&el.ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            if population[most_similar_1].fitness.unwrap().fitness
                > population[repr.children.1].fitness.unwrap().fitness
            {
                killed.push(repr.children.1);
            } else {
                killed.push(most_similar_1);
            }
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn children_fight_parents<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        let mut rgen = SmallRng::from_entropy();
        for repr in parents_children {
            let par0 = &population[repr.parents.0];
            let par1 = &population[repr.parents.1];
            let child0 = &population[repr.children.0];
            let child1 = &population[repr.children.1];
            if rgen.gen_bool(0.5) {
                // First parent fights with first child
                if par0.fitness.unwrap().fitness > child0.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.0);
                } else {
                    // The child survives
                    killed.push(repr.parents.0);
                }
                // Second parent fights with the second child
                if par1.fitness.unwrap().fitness > child1.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.1);
                } else {
                    // The child survives
                    killed.push(repr.parents.1);
                }
            } else {
                // First parent fights with second child
                if par0.fitness.unwrap().fitness > child1.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.1);
                } else {
                    // The child survives
                    killed.push(repr.parents.0);
                }
                // Second parent fights with the first child
                if par1.fitness.unwrap().fitness > child0.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.0);
                } else {
                    // The child survives
                    killed.push(repr.parents.1);
                }
            }
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        killed.dedup();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn deterministic_overpopulation<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        for repr in parents_children {
            let par0 = &population[repr.parents.0];
            let par1 = &population[repr.parents.1];
            let child0 = &population[repr.children.0];
            let child1 = &population[repr.children.1];
            let dist0_0 = par0.ind.distance(&child0.ind);
            let dist0_1 = par0.ind.distance(&child1.ind);
            let dist1_0 = par1.ind.distance(&child0.ind);
            let dist1_1 = par1.ind.distance(&child1.ind);
            // The sum of distances comparing parent 0 to child 0 and
            // parent 1 to child 1 is lower than the another case
            if dist0_0 + dist1_1 <= dist0_1 + dist1_0 {
                // First parent fights with first child
                if par0.fitness.unwrap().fitness > child0.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.0);
                } else {
                    // The child survives
                    killed.push(repr.parents.0);
                }
                // Second parent fights with the second child
                if par1.fitness.unwrap().fitness > child1.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.1);
                } else {
                    // The child survives
                    killed.push(repr.parents.1);
                }
            } else {
                // First parent fights with second child
                if par0.fitness.unwrap().fitness > child1.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.1);
                } else {
                    // The child survives
                    killed.push(repr.parents.0);
                }
                // Second parent fights with the first child
                if par1.fitness.unwrap().fitness > child0.fitness.unwrap().fitness {
                    // The parent survives
                    killed.push(repr.children.0);
                } else {
                    // The child survives
                    killed.push(repr.parents.1);
                }
            }
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        killed.dedup();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn overpopulation<T: PartialEq + Send + Sync, G: Genotype<T>>(
        m: usize,
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        if m < parents_children.len() * 2 {
            panic!(
                "m < children length. m: {}, children_len: {}",
                m,
                parents_children.len() * 2
            )
        }
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        let mut choosable: Vec<usize> = Vec::with_capacity(m);
        let mut rgen = SmallRng::from_entropy();
        let mut children: Vec<usize> = Vec::with_capacity(parents_children.len() * 2);
        for (child0, child1) in parents_children
            .iter()
            .map(|pc| (pc.children.0, pc.children.1))
        {
            children.push(child0);
            children.push(child1);
        }
        for _i in 0..m {
            let mut chosen = rgen.sample(Uniform::from(0..population_size));
            // children have not to be checked because they are after population_size
            while choosable.contains(&chosen) {
                chosen = rgen.sample(Uniform::from(0..population_size));
            }
            choosable.push(chosen);
        }
        for child in children {
            let most_similar = choosable
                .par_iter()
                .enumerate()
                .map(|(i, el)| (i, population[child].ind.distance(&population[*el].ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            killed.push(choosable[most_similar]);
            choosable.remove(most_similar);
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn competitive_overpopulation<T: PartialEq + Send + Sync, G: Genotype<T>>(
        m: usize,
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
        parents_children: &[Reproduction],
    ) {
        if m < parents_children.len() * 2 {
            panic!(
                "m < children length. m: {}, children_len: {}",
                m,
                parents_children.len() * 2
            )
        }
        let mut killed = Vec::with_capacity(parents_children.len() * 2);
        let mut choosable: Vec<usize> = Vec::with_capacity(m);
        let mut rgen = SmallRng::from_entropy();
        let mut children: Vec<usize> = Vec::with_capacity(parents_children.len() * 2);
        for (child0, child1) in parents_children
            .iter()
            .map(|pc| (pc.children.0, pc.children.1))
        {
            children.push(child0);
            children.push(child1);
        }
        for _i in 0..m {
            let mut chosen = rgen.sample(Uniform::from(0..population_size));
            // children have not to be checked because they are after population_size
            while choosable.contains(&chosen) {
                chosen = rgen.sample(Uniform::from(0..population_size));
            }
            choosable.push(chosen);
        }
        for child in children {
            let most_similar = choosable
                .par_iter()
                .enumerate()
                .map(|(i, el)| (i, population[child].ind.distance(&population[*el].ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            if population[choosable[most_similar]].fitness.unwrap().fitness
                > population[child].fitness.unwrap().fitness
            {
                killed.push(child);
            } else {
                killed.push(choosable[most_similar]);
            }
            choosable.remove(most_similar);
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn random_most_similar<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
    ) {
        let tobekilled = population.len() - population_size;
        let mut killed = Vec::with_capacity(tobekilled);
        let mut choosable: Vec<usize> = population.iter().enumerate().map(|(i, _p)| i).collect();
        let mut rgen = SmallRng::from_entropy();
        let mut n_killed = 0;
        while n_killed < tobekilled {
            let chosen = rgen.sample(Uniform::from(0..choosable.len()));

            let most_similar = choosable
                .par_iter()
                .enumerate()
                .filter(|(i, _el)| *i != chosen)
                .map(|(i, el)| (i, population[chosen].ind.distance(&population[*el].ind)))
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .unwrap()
                .0;
            if population[choosable[chosen]].fitness.unwrap().fitness
                > population[choosable[most_similar]].fitness.unwrap().fitness
            {
                killed.push(choosable[most_similar]);
            } else {
                killed.push(choosable[chosen]);
            }
            if chosen > most_similar {
                choosable.remove(chosen);
                choosable.remove(most_similar);
            } else {
                choosable.remove(most_similar);
                choosable.remove(chosen);
            }
            n_killed += 1;
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for el in killed.iter().rev() {
            population.remove(*el);
        }
    }

    fn most_similar<T: PartialEq + Send + Sync, G: Genotype<T>>(
        population_size: usize,
        population: &mut Vec<IndWithFitness<T, G>>,
    ) {
        let tobekilled = population.len() - population_size;
        // Vec with elements to kill in population
        let mut killed = Vec::with_capacity(tobekilled);
        // Vec with the similarity of each individual in the population with
        // any other individual in the population
        let mut dists: Vec<Vec<(usize, f64)>> = population
            .iter()
            .enumerate()
            .map(|_el| Vec::with_capacity(population.len()))
            .collect();
        dists.par_iter_mut().enumerate().for_each(|(i, dists)| {
            for (j, indwf) in population.iter().enumerate() {
                dists.push((j, population[i].ind.distance(&indwf.ind)));
            }
        });
        let mut n_killed = 0;
        while n_killed < tobekilled {
            // Get the element in the population that has the biggest similarity
            // with any other and that individual
            let (chosen, (most_similar, _distance)) = dists
                .par_iter()
                .enumerate()
                .filter(|(i, _el)| !killed.contains(i))
                .map(|(i, distances)| {
                    (
                        i,
                        distances
                            .iter()
                            .filter(|(j, _dist)| i != *j && !killed.contains(j))
                            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                            .unwrap(),
                    )
                })
                .min_by(|x, y| (x.1).1.partial_cmp(&(y.1).1).unwrap())
                .unwrap();

            if population[chosen].fitness.unwrap().fitness
                > population[*most_similar].fitness.unwrap().fitness
            {
                killed.push(*most_similar);
            } else {
                killed.push(chosen);
            }
            n_killed += 1;
        }
        // killed must be sorted because the indexes of elements at the right
        // change when removing elements from vector
        killed.par_sort_unstable();
        for i in killed.iter().rev() {
            population.remove(*i);
        }
    }
}

fn sort_population<T: PartialEq + Send + Sync, G: Genotype<T>>(
    population: &mut [IndWithFitness<T, G>],
) {
    population.par_sort_unstable_by(|el1, el2| {
        el2.fitness
            .unwrap()
            .fitness
            .partial_cmp(&el1.fitness.unwrap().fitness)
            .unwrap()
    });
}

fn sort_population_by_age<T: PartialEq + Send + Sync, G: Genotype<T>>(
    population: &mut [IndWithFitness<T, G>],
) {
    population.par_sort_unstable_by(|el1, el2| {
        el2.fitness
            .unwrap()
            .age
            .partial_cmp(&el1.fitness.unwrap().age)
            .unwrap()
    });
}
