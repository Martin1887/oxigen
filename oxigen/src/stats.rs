//! This module provides the structs and functions used to get fitness
//! and progress statistics for stop criteria and outputs.
pub mod stats_aux;
pub mod stats_fields;
pub mod stats_impl;
pub mod stats_schema;

use std::collections::vec_deque::VecDeque;
use OxigenStatsFieldFunction;
use OxigenStatsFields;
use OxigenStatsSchema;

/// Cached values used to compute statistics.
pub struct OxigenStatsCache {
    /// The best fitness in the last generations.
    best_fitnesses: Option<Vec<f64>>,
    /// The worst fitness in the last generations.
    worst_fitnesses: Option<Vec<f64>>,
    /// The progress in fitness between the best individual in each generation
    /// and the best individual in the previous generation for the last generations.
    best_progresses: Option<Vec<f64>>,
    /// The average fitness in the last generations.
    avg_fitnesses: Option<Vec<f64>>,
    /// The progress between the average fitness in each generation and the
    /// average fitness in the previous generation for the last generations.
    avg_progresses: Option<Vec<f64>>,
    /// The sorted best progress in the last generations.
    best_progress_histogram: Option<Vec<f64>>,
    /// The sorted avg progress in the last generations.
    avg_progress_histogram: Option<Vec<f64>>,
    /// The sorted fitness in the last generation.
    fitness_histogram: Option<Vec<f64>>,
}

impl Default for OxigenStatsCache {
    fn default() -> Self {
        Self::new()
    }
}

impl OxigenStatsCache {
    pub fn new() -> Self {
        OxigenStatsCache {
            best_fitnesses: None,
            worst_fitnesses: None,
            best_progresses: None,
            avg_fitnesses: None,
            avg_progresses: None,
            best_progress_histogram: None,
            avg_progress_histogram: None,
            fitness_histogram: None,
        }
    }
}

/// Fitness values of each generation that can be used to extract statistics.
#[derive(Debug, PartialEq)]
pub struct OxigenStatsGenerationValues {
    /// The fitness of each individual in the generation.
    pub(crate) fitnesses: Vec<f64>,
}

/// A `VecDeque` where each element is a generation and the number of generations
/// is the defined in `GeneticExecution.stats_generations` field. Each generation
/// consists of a `OxigenStatsGenerationValues` object.
/// Generations are in order: the first element in `last_generations` is the
/// oldest recorded generation and the last element is the newest one.
pub struct OxigenStatsValues {
    /// The fitness and progress values in the last generations.
    pub(crate) last_generations: VecDeque<OxigenStatsGenerationValues>,
    /// The number of last generations to keep for statistics. Needed because
    /// VecDeque can put a bigger capacity in the creation.
    /// TODO: check when the exact capacity is stabilized or think in using another struct.
    pub(crate) capacity: usize,
    /// Cache statistics that are cleared in each update.
    pub(crate) cache: OxigenStatsCache,
}

impl OxigenStatsValues {
    /// Initialize the `VecDeque` with the `generations_stats` capacity.
    fn new(generations_stats: usize) -> Self {
        OxigenStatsValues {
            last_generations: VecDeque::with_capacity(generations_stats),
            capacity: generations_stats,
            cache: OxigenStatsCache::new(),
        }
    }
    /// Update the fitness and progress values in each generation.
    pub(crate) fn update(&mut self, gen_fitnesses: &[f64]) {
        // stats are computed after create the object to avoid fitnesses cloning
        let values = OxigenStatsGenerationValues {
            fitnesses: gen_fitnesses.to_vec(),
        };

        // if the queue is full move all elements one place to left and
        // replace the last one, insert at the end otherwise
        if self.last_generations.len() == self.capacity {
            self.last_generations.rotate_left(1);
            if let Some(x) = self.last_generations.back_mut() {
                *x = values
            }
        } else {
            self.last_generations.push_back(values);
        }

        // clear cache
        self.cache = OxigenStatsCache::new();
    }
}

/// Contain the actual values of the last generations, the schema to use to
/// print the statistics and the delimiter used in the CSV output stats file.
/// The stats hierarchy is:
/// OxigenStats
///     OxigenStatsValues
///         OxigenStatsGenerationValues
///         OxigenStatsCache
///     OxigenStatsSchema
///         OxigenStatsInstantiatedField
///             OxigenStatsAllFields
///                 OxigenStatsFields
pub(crate) struct OxigenStats {
    pub(crate) values: OxigenStatsValues,
    schema: OxigenStatsSchema,
    fields_delimiter: String,
}

impl OxigenStats {
    pub(crate) fn new(generations_stats: usize, delimiter: &str) -> Self {
        OxigenStats {
            values: OxigenStatsValues::new(generations_stats),
            schema: OxigenStatsSchema::new(),
            fields_delimiter: delimiter.to_string(),
        }
    }

    pub(crate) fn add_field(
        &mut self,
        name: &str,
        field: Box<dyn OxigenStatsFieldFunction>,
    ) -> &mut Self {
        self.schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from(name),
            enabled: true,
            field,
        });
        self
    }
    pub(crate) fn enable_field(&mut self, field: OxigenStatsFields) -> &mut Self {
        self.schema.fields[field as usize].enabled = true;
        self
    }
    pub(crate) fn disable_field(&mut self, field: OxigenStatsFields) -> &mut Self {
        self.schema.fields[field as usize].enabled = false;
        self
    }
    pub(crate) fn set_delimiter(&mut self, delimiter: &str) -> &mut Self {
        self.fields_delimiter = delimiter.to_string();
        self
    }

    /// Header String without ending line break.
    pub(crate) fn header(&self) -> String {
        let mut header = String::from("");
        let mut empty = true;
        for field in &self.schema.fields {
            if field.enabled {
                if empty {
                    header.push_str(&field.name);
                } else {
                    header.push_str(&format!("{}{}", self.fields_delimiter, field.name));
                }
                empty = false;
            }
        }

        header
    }

    /// The value of the current generation stats without ending line break.
    pub(crate) fn stats_line(&mut self, generation: u64, solutions_found: usize) -> String {
        let mut line = String::from("");
        // generation and solutions found are special and provided by parameters
        line.push_str(&format!(
            "{}{}{}",
            generation, self.fields_delimiter, solutions_found
        ));
        // the rest of enabled fields (statistics)
        for field in &self.schema.fields[2..] {
            line.push_str(&format!(
                "{}{}",
                self.fields_delimiter,
                field.field.function()(&mut self.values),
            ));
        }

        line
    }
}

/// Definition of each field of the statistics file specifying name, enabling
/// status and the field.
pub(crate) struct OxigenStatsInstantiatedField {
    name: String,
    enabled: bool,
    field: Box<dyn OxigenStatsFieldFunction>,
}

/// Private struct containing the stats fields plus generation and solutions.
pub(crate) enum OxigenStatsAllFields {
    /// Current generation.
    Generation,
    /// Number of found solutions.
    Solutions,
    /// Statistics fields.
    StatsField(OxigenStatsFields),
}
impl OxigenStatsAllFields {
    /// The number of all fields. Note that this function must be changed when
    /// new fields are added. This can be done with a macro but it is too much
    /// cumbersome. Probably the `strum` crate could be used, but it seems a
    /// unnecessary dependency.
    fn count() -> usize {
        31
    }
}
impl OxigenStatsFieldFunction for OxigenStatsAllFields {
    fn function(&self) -> &dyn Fn(&mut OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsAllFields::StatsField(field) => field.function(),
            _ => panic!("The function cannot be applied over generation or solutions!"),
        }
    }
    fn uncached_function(&self) -> &dyn Fn(&OxigenStatsValues) -> f64 {
        match self {
            OxigenStatsAllFields::StatsField(field) => field.uncached_function(),
            _ => panic!("The function cannot be applied over generation or solutions!"),
        }
    }
}
