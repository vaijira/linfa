use std::collections::BTreeSet;

use petgraph::stable_graph::StableGraph;

pub struct CausalModel<'a> {
    causal_graph: StableGraph<&'a str, (), u32>,
    treatment: BTreeSet<&'a str>,
    outcome: BTreeSet<&'a str>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa::Dataset;

    #[test]
    fn test_causal_model_params() {
        let data = linfa_datasets::diabetes();
        let feature_names = vec![
            "age",
            "sex",
            "body mass index",
            "blood pressure",
            "t-cells",
            "low-density lipoproteins",
            "high-density lipoproteins",
            "thyroid stimulating hormone",
            "lamotrigine",
            "blood sugar level",
        ];
        data.feature_names();
    }
}
