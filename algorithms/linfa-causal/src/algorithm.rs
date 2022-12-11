use crate::admg::ADMG;
use crate::identifier::{Estimand, IDIdentifier, Identifier};
use crate::setutils::OrderedSet;
use anyhow::Result;
use linfa::prelude::*;
use linfa_linear::TweedieRegressor;
use ndarray::prelude::*;

pub struct CausalModel<'a> {
    causal_graph: ADMG<&'a str>,
    treatment: OrderedSet<&'a str>,
    outcome: OrderedSet<&'a str>,
}

impl<'a> CausalModel<'a> {
    pub fn new(
        causal_graph: ADMG<&'a str>,
        treatment: OrderedSet<&'a str>,
        outcome: OrderedSet<&'a str>,
    ) -> CausalModel<'a> {
        CausalModel {
            causal_graph,
            treatment,
            outcome,
        }
    }

    pub fn id(&self) -> Result<Estimand<&str>> {
        let identifier = IDIdentifier::new();
        identifier.identify(&self.causal_graph, &self.treatment, &self.outcome)
    }

    pub fn estimate(&self, dataset: &Dataset<f64, f64>, estimand: &Estimand<&str>) -> f64 {
        let conditional_vars = estimand.conditional_vars();
        let _cols = self.treatment.len() + conditional_vars.len();
        let mut regression_records = Array2::<f64>::zeros((dataset.nsamples(), 0));
        for treatment in self.treatment.iter() {
            let index = dataset
                .feature_names()
                .iter()
                .position(|f| f == *treatment)
                .unwrap();
            println!("Found treatment {} index {}", treatment, index);
            regression_records.push_column(dataset.records().column(index));
        }

        for conditional_var in conditional_vars.iter() {
            if self.treatment.contains(*conditional_var) {
                continue;
            }
            let index = dataset
                .feature_names()
                .iter()
                .position(|f| f == *conditional_var)
                .unwrap();
            println!("Found conditional {} index {}", conditional_var, index);
            regression_records.push_column(dataset.records().column(index));
        }

        let outcome = self.outcome.iter().next().cloned().unwrap();
        let index = dataset
            .feature_names()
            .iter()
            .position(|f| f == outcome)
            .unwrap();
        let targets = dataset.records().column(index).to_owned();
        let regression_dataset = Dataset::new(regression_records, targets);

        let lin_reg = TweedieRegressor::params().power(0.).alpha(0.);
        let model = lin_reg.fit(&regression_dataset).unwrap();
        *model.coef.get(0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa::Dataset;
    use ndarray::Array1;
    use ndarray_rand::rand::prelude::SmallRng;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::{Normal, StandardNormal};
    use ndarray_rand::RandomExt;

    /*
    Estimating the causal effect of sodium on blood pressure in a simulated example
    adapted from Luque-Fernandez et al. (2018):
        https://academic.oup.com/ije/article/48/2/640/5248195
    */
    fn generate_data(
        samples: usize,
        seed: u64,
        beta1: f64,
        alpha1: f64,
        alpha2: f64,
    ) -> Dataset<f64, f64> {
        // Our random number generator, seeded for reproducibility
        let mut rng = SmallRng::seed_from_u64(seed);

        let age = Array1::<f64>::random_using(samples, Normal::new(65.0, 5.0).unwrap(), &mut rng);

        let sodium = &age / 18.0 + Array1::<f64>::random_using(samples, StandardNormal, &mut rng);

        let blood_pressure = beta1 * &sodium
            + 2.0 * &age
            + Array1::<f64>::random_using(samples, StandardNormal, &mut rng);

        let protein_in_uria = alpha1 * &sodium
            + alpha2 * &blood_pressure
            + Array1::<f64>::random_using(samples, StandardNormal, &mut rng);

        let mut records = Array2::<f64>::zeros((samples, 0));

        records.push_column(age.view());
        records.push_column(sodium.view());
        records.push_column(blood_pressure.view());
        records.push_column(protein_in_uria.view());

        let targets = Array1::<f64>::zeros(samples);

        let feature_names = vec!["age", "sodium", "blood_pressure", "protein_in_uria"];

        Dataset::new(records, targets).with_feature_names(feature_names)
    }
    #[test]
    fn test_causal_model_params() {
        let data = generate_data(1_000_000, 42, 1.05, 0.4, 0.3);

        println!(
            "records cols: {}, rows: {}",
            data.records().ncols(),
            data.records().nrows()
        );
        println!(
            "record[1, 1] = {:?}, record [999, 3] = {:?}",
            data.records().get((0, 0)),
            data.records().get((999, 3))
        );
        println!("{:?}", data.feature_names());
        let mut g = ADMG::<&str>::new();
        let sodium = g.add_node("sodium");
        let age = g.add_node("age");
        let blood_pressure = g.add_node("blood_pressure");
        let protein_in_uria = g.add_node("protein_in_uria");
        g.add_edge(sodium, blood_pressure, ());
        g.add_edge(age, sodium, ());
        g.add_edge(age, blood_pressure, ());
        g.add_edge(sodium, protein_in_uria, ());
        g.add_edge(blood_pressure, protein_in_uria, ());

        let causal_model = CausalModel::new(
            g,
            ["sodium"].iter().copied().collect(),
            ["blood_pressure"].iter().copied().collect(),
        );

        let estimand = causal_model.id().unwrap();
        println!("Estimand: {}", estimand);
        let estimate = causal_model.estimate(&data, &estimand);
        println!("Estimate: {}", estimate);

        // True estimate 1.05
        assert_abs_diff_eq!(1.05, estimate, epsilon = 1e-2);
    }
}
