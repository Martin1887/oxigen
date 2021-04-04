use OxigenStatsAllFields;
use OxigenStatsFields;
use OxigenStatsInstantiatedField;

/// Schema for the statistics CSV file.
pub(crate) struct OxigenStatsSchema {
    pub(crate) fields: Vec<OxigenStatsInstantiatedField>,
}

impl OxigenStatsSchema {
    pub(crate) fn new() -> Self {
        let mut schema = OxigenStatsSchema {
            fields: Vec::with_capacity(OxigenStatsAllFields::count()),
        };
        // Probably strum or a custom macro could be used for this
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Generation"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::Generation),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Solutions"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::Solutions),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best last progress"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestLastProgress,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress average"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Best progress p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::BestProgressP90,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness avg"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Fitness p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::FitnessP90,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg last progress"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgLastProgress,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress average"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressAvg,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress std"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressStd,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress max"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMax,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress min"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMin,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p10"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP10,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p25"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP25,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress median"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressMedian,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p75"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP75,
            )),
        });
        schema.fields.push(OxigenStatsInstantiatedField {
            name: String::from("Avg progress p90"),
            enabled: true,
            field: Box::new(OxigenStatsAllFields::StatsField(
                OxigenStatsFields::AvgProgressP90,
            )),
        });

        schema
    }
}
