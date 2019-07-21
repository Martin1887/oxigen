//! This module contains the Selection trait and the provided selection functions.

use rand::distributions::{Standard, Uniform};
use rand::prelude::*;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::mpsc::channel;
use SelectionFunctions::*;

/// This trait defines the select function used to select individuals for crossover.
pub trait Selection: Send + Sync {
    /// Returns a vector with the indices of the selected individuals according to
    /// the fitness of the population and the selection rate.
    fn select(&self, fitnesses: &[f64], selection_rate: usize) -> Vec<usize>;
}

/// Defines the number of tournaments parameter.
pub struct NTournaments(pub usize);

/// Provided selection functions
pub enum SelectionFunctions {
    /// Roulette function.
    Roulette,
    /// Tournaments function.
    Tournaments(NTournaments),
    /// Cup function, a cup tournament where the `selection_rate` first phases
    /// of the tournament are selected (winner, final, semifinal, etc.).
    Cup,
}

impl Selection for SelectionFunctions {
    fn select(&self, fitnesses: &[f64], selection_rate: usize) -> Vec<usize> {
        match self {
            Roulette => {
                let mut rgen = SmallRng::from_entropy();
                let mut winners = Vec::with_capacity(selection_rate);
                let mut probs = Vec::with_capacity(fitnesses.len());
                let fitness_sum: f64 = fitnesses.iter().sum();
                for fit in fitnesses {
                    probs.push(fit / fitness_sum);
                }
                for _i in 0..selection_rate {
                    let random: f64 = rgen.sample(Standard);
                    let mut accum = 0_f64;
                    for (ind, p) in probs.iter().enumerate() {
                        if random <= accum + p {
                            winners.push(ind);
                            break;
                        }
                        accum += p;
                    }
                }

                winners
            }
            Tournaments(n_tournaments) => {
                let (sender, receiver) = channel();
                let mut winners = Vec::with_capacity(n_tournaments.0);
                Range {
                    start: 0,
                    end: n_tournaments.0,
                }
                .into_par_iter()
                .for_each_with(sender, |s, _t| {
                    let mut rgen = SmallRng::from_entropy();
                    let mut fighters = Vec::with_capacity(selection_rate);
                    for _f in 0..selection_rate {
                        let sel = rgen.sample(Uniform::from(0..fitnesses.len()));
                        fighters.push((sel, fitnesses[sel]));
                    }
                    s.send(
                        fighters
                            .par_iter()
                            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                            .unwrap()
                            .0,
                    )
                    .unwrap();
                });

                for win in receiver {
                    winners.push(win);
                }
                winners
            }
            Cup => {
                let mut rgen = SmallRng::from_entropy();
                let mut winners = Vec::with_capacity(2_i32.pow(selection_rate as u32) as usize - 1);
                let n_phases = (fitnesses.len() as f64).log2() as usize + 1;
                let mut phases: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_phases);
                // Initialize the phases members
                for i in 0..n_phases {
                    let size = 2_i32.pow(i as u32) as usize;
                    phases.push(Vec::with_capacity(size));
                    for _j in 0..size {
                        phases[i].push((0, 0_f64));
                    }
                }
                // Randomly initialize the last phase
                let mut candidates = Vec::with_capacity(fitnesses.len());
                for f in fitnesses {
                    candidates.push(*f);
                }
                {
                    let last_phase = &mut phases[n_phases - 1];
                    let mut i = 0;
                    while i < last_phase.len() {
                        let sel = rgen.sample(Uniform::from(0..candidates.len()));
                        last_phase[i] = (sel, candidates[sel]);
                        candidates.remove(sel);
                        i += 1;
                    }
                }

                // Do the fights
                Self::cup_phase_fight(&mut phases, n_phases - 1);

                // Push winners
                for phase in phases
                    .iter()
                    .enumerate()
                    .filter(|(i, _p)| *i < selection_rate as usize)
                    .map(|(_i, p)| p)
                {
                    for winner in phase.iter() {
                        if winners.is_empty() {
                            winners.push(winner.0);
                        } else {
                            let index = rgen.sample(Uniform::from(0..winners.len()));
                            winners.insert(index, winner.0);
                        }
                    }
                }

                winners
            }
        }
    }
}

impl SelectionFunctions {
    fn cup_phase_fight(phases: &mut Vec<Vec<(usize, f64)>>, phase: usize) {
        let (sender, receiver) = channel();

        Range {
            start: 0,
            end: phases[phase].len() / 2,
        }
        .into_par_iter()
        .for_each_with(sender, |s, i| {
            let ind1 = i * 2;
            let ind2 = ind1 + 1;

            if phases[phase][ind1].1 >= phases[phase][ind2].1 {
                s.send((i, phases[phase][ind1])).unwrap();
            } else {
                s.send((i, phases[phase][ind2])).unwrap();
            }
        });
        for (i, child) in receiver {
            phases[phase - 1][i] = child;
        }
        if phase > 1 {
            Self::cup_phase_fight(phases, phase - 1);
        }
    }
}
