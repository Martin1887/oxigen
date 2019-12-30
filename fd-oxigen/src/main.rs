//! Genetic algorithm to find the optimal admissible heuristics to use in
//! Fast-Downward to find optimal solutions using the max value of all of them.

extern crate clap;
extern crate oxigen;
extern crate rand;
extern crate rayon;

use crate::rayon::prelude::*;
use clap::{App, Arg};
use oxigen::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::time::{Duration, Instant};

/// Each gen means a heuristic to be used in A* algorithm selecting the max value
/// of all of them. A 0 means that that heuristic is not used (20% or probability).
/// Each heuristic have probability of 80% to be chosen, i.e., each option for each
/// heuristic have probability 0.8 * (1 / (options)) to be chosen.
/// The genes are:
/// 0: blind heuristic (only transform changes, 4 options)
/// 1: cegar (transform, use_general_costs, pick={MAX_REFINED, RANDOM, MIN_UNWANTED,
///     MAX_UNWANTED, MIN_REFINED, MIN_HADD, MAX_HADD}, 56 options)
/// 2: hm (m=2) (only transform changes, 4 options)
/// 3: hmax (only transform changes, 4 options)
/// 4: lmcount (optimal, pref, alm and transform changes, 32 options)
/// 5: lmcut (only transform changes, 4 options)
/// 6: Merge and Shrink (only transform changes, 4 options)
/// 7: CPDBs (canonical PDBs) (transform changes, 4 options)
/// 8: IPDBs (transform changes, 4 options)
/// 9: PDBs (transform changes, 4 options)
/// 10: ZOPDBs (transform changes, 4 options)
#[derive(Clone, PartialEq, Eq, std::hash::Hash)]
struct FdParams<'a> {
    genes: Vec<u8>,
    fd_path: &'a str,
    domains_and_problems_dir_path: &'a str,
}
impl<'a> Display for FdParams<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mut s = String::from("astar(max([");
        // blind
        match self.genes[0] {
            1 => s.push_str("blind(),"),
            2 => s.push_str("blind(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("blind(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("blind(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }
        // cegar
        if self.genes[1] > 0 {
            s.push_str("cegar(");
            if self.genes[1] - 1 % 4 == 1 {
                s.push_str("transform=adapt_costs(cost_type=NORMAL),");
            } else if self.genes[1] - 1 % 4 == 2 {
                s.push_str("transform=adapt_costs(cost_type=ONE),");
            } else if self.genes[1] - 1 % 4 == 3 {
                s.push_str("transform=adapt_costs(cost_type=PLUSONE),");
            }
            if self.genes[1] - 1 % 8 >= 4 {
                s.push_str("use_general_costs=false,");
            }
            if self.genes[1] - 1 >= 48 {
                s.push_str("pick=MAX_HADD,");
            } else if self.genes[1] - 1 >= 40 {
                s.push_str("pick=MIN_HADD,");
            } else if self.genes[1] - 1 >= 32 {
                s.push_str("pick=MIN_REFINED,");
            } else if self.genes[1] - 1 >= 24 {
                s.push_str("pick=MAX_UNWANTED,");
            } else if self.genes[1] - 1 >= 16 {
                s.push_str("pick=MIN_UNWANTED,");
            } else if self.genes[1] - 1 >= 8 {
                s.push_str("pick=RANDOM,");
            }
            if s.ends_with(",") {
                s.pop();
            }
            s.push_str("),")
        }

        // hm
        match self.genes[2] {
            1 => s.push_str("hm(),"),
            2 => s.push_str("hm(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("hm(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("hm(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        // hmax
        match self.genes[3] {
            1 => s.push_str("hmax(),"),
            2 => s.push_str("hmax(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("hmax(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("hmax(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        // lmcount
        if self.genes[4] > 0 {
            if self.genes[4] - 1 < 32 {
                s.push_str("lmcount(lm_rhw,admissible=true,");
            } else if self.genes[4] - 1 < 64 {
                s.push_str("lmcount(lm_exhaust,admissible=true,");
            } else if self.genes[4] - 1 < 96 {
                s.push_str("lmcount(lm_hm,admissible=true,");
            } else if self.genes[4] - 1 < 128 {
                s.push_str("lmcount(lm_zg,admissible=true,");
            }
            if self.genes[4] - 1 % 4 == 1 {
                s.push_str("transform=adapt_costs(cost_type=NORMAL),");
            } else if self.genes[4] - 1 % 4 == 2 {
                s.push_str("transform=adapt_costs(cost_type=ONE),");
            } else if self.genes[4] - 1 % 4 == 3 {
                s.push_str("transform=adapt_costs(cost_type=PLUSONE),");
            }
            if self.genes[4] - 1 % 8 >= 4 {
                s.push_str("optimal=true,");
            }
            if self.genes[4] - 1 % 16 >= 8 {
                s.push_str("pref=true,");
            }
            if self.genes[4] - 1 >= 16 {
                s.push_str("alm=false,");
            }
            if s.ends_with(",") {
                s.pop();
            }
            s.push_str("),")
        }

        // lmcut
        match self.genes[5] {
            1 => s.push_str("lmcut(),"),
            2 => s.push_str("lmcut(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("lmcut(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("lmcut(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        // merge and shrink
        if self.genes[6] > 0 {
            s.push_str("merge_and_shrink(merge_strategy=merge_sccs(order_of_sccs=topological,");
            s.push_str("merge_selector=score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order])),");
            s.push_str("shrink_strategy=shrink_bisimulation(greedy=false),");
            match self.genes[6] {
                2 => s.push_str("transform=adapt_costs(cost_type=NORMAL)"),
                3 => s.push_str("transform=adapt_costs(cost_type=ONE)"),
                4 => s.push_str("transform=adapt_costs(cost_type=PLUSONE)"),
                _ => {}
            }
            if s.ends_with(",") {
                s.pop();
            }
            s.push_str("),")
        }

        // cPDBs
        match self.genes[7] {
            1 => s.push_str("cpdbs(),"),
            2 => s.push_str("cpdbs(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("cpdbs(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("cpdbs(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        match self.genes[8] {
            1 => s.push_str("ipdb(),"),
            2 => s.push_str("ipdb(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("ipdb(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("ipdb(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        match self.genes[9] {
            1 => s.push_str("pdb(),"),
            2 => s.push_str("pdb(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("pdb(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("pdb(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        match self.genes[10] {
            1 => s.push_str("zopdbs(),"),
            2 => s.push_str("zopdbs(transform=adapt_costs(cost_type=NORMAL)),"),
            3 => s.push_str("zopdbs(transform=adapt_costs(cost_type=ONE)),"),
            4 => s.push_str("zopdbs(transform=adapt_costs(cost_type=PLUSONE)),"),
            _ => {}
        }

        if s.ends_with(",") {
            s.pop();
        }
        s.push_str("]))");
        write!(f, "{}", s)
    }
}

impl<'a> FdParams<'a> {
    fn n_options_without_zero(index: usize) -> u8 {
        match index {
            0 => 4,
            1 => 56,
            2 => 4,
            3 => 4,
            4 => 128,
            5 => 4,
            6 => 4,
            7 => 4,
            8 => 4,
            9 => 4,
            10 => 4,
            _ => panic!(format!("Incorrect index passed to n_values: {}", index)),
        }
    }

    fn random_value(rgen: &mut SmallRng, index: usize) -> u8 {
        if rgen.sample(Uniform::from(0..5)) < 1 {
            0
        } else {
            let n = Self::n_options_without_zero(index);
            rgen.sample(Uniform::from(0..n)) + 1 as u8
        }
    }

    fn run_fd(
        hash: &u64,
        fd_path: &str,
        domain_path: &str,
        problem_path: &str,
        search_arguments: &str,
    ) -> u128 {
        // the processes sometimes fail, so it must be rerun until valid execution
        let mut reruns = 0;
        let mut status_code = 247;

        // An output.sas SAS file must be generated only for this execution to avoid
        // conflicts with concurrent executions
        let sas_file = format!(
            "output_{}_{}.sas",
            problem_path.split("/").last().unwrap(),
            hash
        );

        let mut before_time = Instant::now();
        // 0 is SUCCESS, 23 is SEARCH_OUT_OF_TIME, 24 is SEARCH_OUT_OF_MEMORY_AND_TIME.
        // Otherwise rerun at most 3 times.
        while ((status_code != 0 && status_code <= 22) || status_code > 24) && reruns <= 3 {
            before_time = Instant::now();
            // The command may fail because exceeding time limit, ignore it
            match Command::new("python3")
                .arg(fd_path)
                .arg("--sas-file")
                .arg(&sas_file)
                .arg("--search-time-limit")
                .arg("180")
                .arg(domain_path)
                .arg(problem_path)
                .arg("--search")
                .arg(search_arguments)
                .output()
            {
                Ok(out) => {
                    status_code = out.status.code().unwrap_or(247);
                }
                Err(_) => {}
            };
            fs::remove_file(&sas_file).ok();
            if (status_code != 0 && status_code <= 22) || status_code > 24 {
                reruns += 1;
                std::thread::sleep(Duration::from_millis(2000));
            }
        }
        // 30-40 codes are errors, maximum time
        if (status_code != 0 && status_code <= 22) || status_code > 24 {
            return 1_000_000;
        }
        before_time.elapsed().as_millis()
    }
}

impl<'a> Genotype<u8> for FdParams<'a> {
    // The path to the fast-downard.py and the folder with domains and problems
    // A instance of a default hasher is used too to hash the genes as SAS filenames.
    type ProblemSize = &'a [&'a str];
    type GenotypeHash = Self;

    fn iter(&self) -> std::slice::Iter<u8> {
        self.genes.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<u8> {
        self.genes.into_iter()
    }
    fn from_iter<I: Iterator<Item = u8>>(&mut self, genes: I) {
        self.genes = genes.collect();
    }

    fn generate(size: &Self::ProblemSize) -> Self {
        let mut individual = Vec::with_capacity(8 as usize);
        let mut rgen = SmallRng::from_entropy();
        for i in 0..11 {
            individual.push(Self::random_value(&mut rgen, i));
        }
        FdParams {
            genes: individual,
            fd_path: size[0],
            domains_and_problems_dir_path: size[1],
        }
    }

    // The negative value of the sum of execution time of the used benchmarks.
    fn fitness(&self) -> f64 {
        // Zero heuristic is not allowed, minimum heuristic value
        if self.genes.iter().filter(|x| **x == 0).count() == self.genes.len() {
            return -10_000_000.0;
        }

        let search = self.to_string();
        let executions = [
            (
                format!("{}/transport.pddl", self.domains_and_problems_dir_path),
                format!("{}/transport_p04.pddl", self.domains_and_problems_dir_path),
            ),
            (
                format!("{}/elevators.pddl", self.domains_and_problems_dir_path),
                format!("{}/elevators_p05.pddl", self.domains_and_problems_dir_path),
            ),
            (
                format!("{}/parcprinter.pddl", self.domains_and_problems_dir_path),
                format!(
                    "{}/parcprinter_p05.pddl",
                    self.domains_and_problems_dir_path
                ),
            ),
            (
                format!("{}/visitall.pddl", self.domains_and_problems_dir_path),
                format!("{}/visitall_p12.pddl", self.domains_and_problems_dir_path),
            ),
        ];

        let mut hasher = DefaultHasher::new();
        Hash::hash(&self, &mut hasher);
        let hash = hasher.finish();

        let total_time: u128 = executions
            .par_iter()
            .map(|(dom, prob)| Self::run_fd(&hash, self.fd_path, &dom, &prob, &search))
            .sum();

        -(total_time as f64)
    }

    fn mutate(&mut self, mut rgen: &mut SmallRng, index: usize) {
        self.genes[index] = Self::random_value(&mut rgen, index);
    }

    fn is_solution(&self, _fitness: f64) -> bool {
        false
    }

    fn hash(&self) -> Self {
        self.clone()
    }
}

fn main() {
    let matches =
        App::new("Fast Downward planner heuristic optimization for optimal solutions using oxigen")
            .version("1.0")
            .author("Martin1887")
            .arg(
                Arg::with_name("fast_downward_py_path")
                    .help("fast-downward.py executable path")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name("domains_and_problems_dir_path")
                    .help("Path to the directory where domains and problems files are. The required files are transport.pddl, transport_p04.pddl, elevators.pddl, elevators_p05.pddl, parcprinter.pddl, parcprinter_p05.pddl, visitall.pddl, visitall_p12.pddl")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name("progress_and_population_log_dir_path")
                    .help("Path to the directory where progress and population log will be written")
                    .index(3)
                    .required(true),
            )
            .get_matches();

    let fd_params = [
        matches
            .value_of("fast_downward_py_path")
            .expect("Missing first argument"),
        matches
            .value_of("domains_and_problems_dir_path")
            .expect("Missnig second argument"),
    ];
    let logs_dir = matches
        .value_of("progress_and_population_log_dir_path")
        .expect("Missing third argument");
    let progress_log = File::create(format!("{}/progress.csv", logs_dir))
        .expect("Error creating progress log file");
    let population_log = File::create(format!("{}/population.txt", logs_dir))
        .expect("Error creating population log file");
    let population_size = 128 as usize;

    let (_solutions, generation, progress, _population) = GeneticExecution::<u8, FdParams>::new()
        .population_size(population_size)
        .genotype_size(&fd_params)
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: 0.05,
            bound: 0.005,
            coefficient: -0.0001,
        })))
        .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: 6_f64,
            bound: 3_f64,
            coefficient: -0.0025,
        })))
        .select_function(Box::new(SelectionFunctions::Cup))
        .crossover_function(Box::new(CrossoverFunctions::SingleCrossPoint))
        .survival_pressure_function(Box::new(
            SurvivalPressureFunctions::CompetitiveOverpopulation(M::new(
                population_size * 3 / 4,
                population_size / 2,
                population_size,
            )),
        ))
        .stop_criterion(Box::new(StopCriteria::Generation(1_000_000)))
        .progress_log(1, progress_log)
        .population_log(10, population_log)
        .run();

    println!(
        "Finished in the generation {} with a progress of {}",
        generation, progress
    );
}
