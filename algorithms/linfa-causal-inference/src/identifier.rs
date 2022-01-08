use anyhow::{bail, Result};
use petgraph::algo::toposort;
use petgraph::stable_graph::NodeIndex;
use std::{fmt::Debug, fmt::Display, hash::Hash};

use crate::admg::ADMG;
use crate::setutils::{setdiff, setintersect, setunion, OrderedSet};

#[derive(Clone)]
enum EstimandProbabilities<T> {
    // outcome variables, conditional variables
    Basic(Vec<T>, Vec<T>),
    // product, sum
    Compose(Vec<Estimand<T>>, Vec<T>),
}
#[derive(Clone)]
pub struct Estimand<T> {
    value: EstimandProbabilities<T>,
}

impl<T> Display for Estimand<T>
where
    T: Display + Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.value {
            EstimandProbabilities::Basic(outcome_vars, conditional_vars) => {
                write!(f, "P(")?;
                let mut first = true;
                for outcome_var in outcome_vars.iter() {
                    if !first {
                        write!(f, ",")?;
                    } else {
                        first = false;
                    }
                    write!(f, "{}", outcome_var)?;
                }

                if !conditional_vars.is_empty() {
                    write!(f, "|")?;
                    first = true;
                    for conditional_var in conditional_vars.iter() {
                        if !first {
                            write!(f, ",")?;
                        } else {
                            first = false;
                        }
                        write!(f, "{}", conditional_var)?;
                    }
                }

                write!(f, ")")?;
            }
            EstimandProbabilities::Compose(product, sumset) => {
                if !sumset.is_empty() {
                    write!(f, "∑{{")?;
                    let mut first = true;
                    for var in sumset.iter() {
                        if !first {
                            write!(f, ",")?;
                        } else {
                            first = false;
                        }
                        write!(f, "{}", var)?;
                    }
                    write!(f, "}}(")?;
                }
                for prod in product.iter() {
                    write!(f, "{}", prod)?;
                }
                if !sumset.is_empty() {
                    write!(f, ")")?;
                }
            }
        }
        Ok(())
    }
}

pub struct IDIdentifier {}

impl IDIdentifier {
    /*
      Implementation of the ID algorithm.
      Link - https://ftp.cs.ucla.edu/pub/stat_ser/shpitser-thesis.pdf
      helpful link for implementation in R: https://arxiv.org/pdf/1806.07161.pdf
      The pseudo code has been provided on Pg 40.
      Reference implementations:
      - In R: https://github.com/santikka/causaleffect/blob/master/R/id.R
      - In python: https://github.com/microsoft/dowhy/blob/master/dowhy/causal_identifiers/id_identifier.py
    */
    fn id_internal<T>(
        &self,
        outcome: &OrderedSet<NodeIndex>,
        treatment: &OrderedSet<NodeIndex>,
        probability: &mut Estimand<T>,
        graph: &ADMG<T>,
    ) -> Result<Estimand<T>>
    where
        T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord + Debug,
    {
        let topo = toposort(graph.graph(), None)
            .or_else(|_| bail!("Detected cycle in topological ordering"))?;

        // Step 1
        // If no action has been taken, the effect on Y is just the marginal of the observational distribution P(v) on Y.
        if treatment.is_empty() {
            println!("Executing step 1");
            let basic = Estimand {
                value: EstimandProbabilities::Basic(
                    topo.iter()
                        .map(|x| graph.node_weight(*x).unwrap().clone())
                        .collect(),
                    vec![],
                ),
            };

            let sumset = topo
                .iter()
                .filter(|node| !outcome.contains(*node))
                .map(|node| graph.node_weight(*node).unwrap().clone())
                .collect();

            return Ok(Estimand {
                value: EstimandProbabilities::Compose(vec![basic], sumset),
            });
        }

        // Step 2
        // If we are interested in the effect on Y (outcome), it is sufficient to restrict our
        // attention on the parts of the model ancestral to Y.
        let ancestors = graph.ancestors(outcome);
        dbg!(&ancestors);
        let toposet: OrderedSet<NodeIndex> = topo.iter().copied().collect();
        let v_minus_ancestors: OrderedSet<NodeIndex> = setdiff(&toposet, &ancestors);

        if !v_minus_ancestors.is_empty() {
            println!("Executing step 2");
            let new_treatment: OrderedSet<NodeIndex> = setintersect(treatment, &ancestors);
            let subgraph = graph.induced_subgraph(&ancestors);

            let estimand = self.id_internal(outcome, &new_treatment, probability, &subgraph)?;

            return Ok(Estimand {
                value: EstimandProbabilities::Compose(vec![estimand], vec![]),
            });
        }

        // Step 3
        // Forces an action on any node where such an action would have no effect on Y (outcome)
        // assuming we already acted on X (treatment).
        // Modify graph to obtain that corresponding to do(X)
        let v_minus_treament: OrderedSet<NodeIndex> = setdiff(&toposet, treatment);
        let ancestors_g_upper_treatment = graph.upper_bar_variables(treatment).ancestors(outcome);
        let w: OrderedSet<NodeIndex> = setdiff(&v_minus_treament, &ancestors_g_upper_treatment);
        if !w.is_empty() {
            println!("Executing step 3");
            return self.id_internal(outcome, &setunion(treatment, &w), probability, graph);
        }

        // Step 4
        // Decomposes the problem into a set of smaller problems using the key property of C-component factorization of causal models.
        // If the entire graph is a single C-component already, further problem decomposition is impossible, and we must provide base cases.
        // Modify graph to remove treatment variables.
        let g_minus_treatment = graph.induced_subgraph(&v_minus_treament);
        let s = g_minus_treatment.c_components(&topo);
        if s.len() > 1 {
            println!("Executing step 4");
            let outcome_treatment: OrderedSet<NodeIndex> = setunion(outcome, treatment);

            let sumset: Vec<T> = toposet
                .difference(&outcome_treatment)
                .into_iter()
                .copied()
                .map(|x| graph.graph().node_weight(x).unwrap().clone())
                .collect();
            dbg!(&sumset);
            let mut product = vec![];

            for components in s.iter() {
                let v_minus_s: OrderedSet<NodeIndex> = setdiff(&toposet, components);
                let estimand = self.id_internal(components, &v_minus_s, probability, graph)?;
                product.push(estimand);
            }

            return Ok(Estimand {
                value: EstimandProbabilities::Compose(product, sumset),
            });
        }

        // Step 5
        // The algorithms fails due to the presence of a hedge - the graph G, and a subgraph S that does not contain any X nodes.
        let c_components_g = graph.c_components(&topo);
        if c_components_g.len() == 1 && c_components_g[0] == toposet {
            println!("Executing step 5");
            bail!("Hedge found");
        }

        // Step 6
        // If there are no bidirected arcs from X to the other nodes in the current subproblem under consideration,
        // then we can replace acting on X by conditioning, and thus solve the subproblem.
        let s_exists_in_g = {
            let mut found = false;
            for c_component in c_components_g.iter() {
                if s[0] == *c_component {
                    found = true;
                    break;
                }
            }
            found
        };
        if s_exists_in_g {
            println!("Executing step 6");
            dbg!(&s);

            let sumset: Vec<T> = s[0]
                .difference(outcome)
                .into_iter()
                .copied()
                .map(|x| graph.graph().node_weight(x).unwrap().clone())
                .collect();
            dbg!(&sumset);
            let mut product = vec![];
            let mut prev_nodes = vec![];

            for node in topo.iter() {
                if s[0].contains(node) {
                    let outcome_var = vec![graph.node_weight(*node).unwrap().clone()];
                    let basic = Estimand {
                        value: EstimandProbabilities::Basic(outcome_var, prev_nodes.clone()),
                    };
                    product.push(basic);
                }
                prev_nodes.push(graph.node_weight(*node).unwrap().clone());
            }

            return Ok(Estimand {
                value: EstimandProbabilities::Compose(product, sumset),
            });
        }

        // Step 7
        // This is the most complicated case in the algorithm.
        // Explain in the second last paragraph on Pg 41 of the link provided in the docstring above.
        for c_component in c_components_g.iter() {
            println!("Executing step 7");
            if s[0].difference(c_component).into_iter().next() == None {
                return self.id_internal(
                    outcome,
                    &setintersect(treatment, c_component),
                    probability,
                    &graph.induced_subgraph(c_component),
                );
            }
        }

        bail!("This code should be unreachable, this should never happen");
    }

    pub fn id<T>(
        &self,
        outcome: &OrderedSet<T>,
        treatment: &OrderedSet<T>,
        graph: &ADMG<T>,
    ) -> Result<Estimand<T>>
    where
        T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord + Debug,
    {
        let mut probability: Estimand<T> = Estimand {
            value: EstimandProbabilities::Compose(vec![], vec![]),
        };

        let outcome_internal: OrderedSet<NodeIndex> = graph
            .node_indices()
            .filter(|i| outcome.contains(graph.node_weight(*i).unwrap()))
            .collect();

        let treatment_internal: OrderedSet<NodeIndex> = graph
            .node_indices()
            .filter(|i| treatment.contains(graph.node_weight(*i).unwrap()))
            .collect();

        self.id_internal(
            &outcome_internal,
            &treatment_internal,
            &mut probability,
            graph,
        )
    }

    pub fn id_index<T>(
        &self,
        outcome: &OrderedSet<NodeIndex>,
        treatment: &OrderedSet<NodeIndex>,
        graph: &ADMG<T>,
    ) -> Result<Estimand<T>>
    where
        T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord + Debug,
    {
        let mut probability: Estimand<T> = Estimand {
            value: EstimandProbabilities::Compose(vec![], vec![]),
        };

        self.id_internal(outcome, treatment, &mut probability, graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_dag() {
        let mut causal_graph = ADMG::<&str>::new();
        let t = causal_graph.add_node("T");
        let y = causal_graph.add_node("Y");
        causal_graph.add_edge(t, y, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();

        let identifier = IDIdentifier {};
        if let Ok(estimand) = identifier.id_index(&y_set, &t_set, &causal_graph) {
            assert_eq!(format!("{}", estimand), "P(Y|T)");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_cycle_dag() {
        let mut g = ADMG::<&str>::new();
        let t = g.add_node("T");
        let y = g.add_node("Y");
        g.add_edge(t, y, ());
        g.add_edge(y, t, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();

        let identifier = IDIdentifier {};
        if let Err(msg) = identifier.id_index(&y_set, &t_set, &g) {
            assert_eq!(format!("{}", msg), "Detected cycle in topological ordering");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_mediator_dag() {
        let mut g = ADMG::<&str>::new();
        let t = g.add_node("T");
        let m = g.add_node("M");
        let y = g.add_node("Y");
        g.add_edge(t, m, ());
        g.add_edge(m, y, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();

        let identifier = IDIdentifier {};
        if let Ok(estimand) = identifier.id_index(&y_set, &t_set, &g) {
            assert_eq!(format!("{}", estimand), "∑{M}(P(M|T)P(Y|T,M))");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_fork_dag() {
        let mut g = ADMG::<&str>::new();
        let t = g.add_node("T");
        let x = g.add_node("X");
        let y = g.add_node("Y");
        g.add_edge(t, x, ());
        g.add_edge(t, y, ());
        g.add_edge(x, y, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();
        let identifier = IDIdentifier {};
        if let Ok(estimand) = identifier.id_index(&y_set, &t_set, &g) {
            assert_eq!(format!("{}", estimand), "∑{X}(P(X|T)P(Y|T,X))");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_common_cause_dag() {
        let mut g = ADMG::<&str>::new();
        let t = g.add_node("T");
        let x1 = g.add_node("X1");
        let x2 = g.add_node("X2");
        let y = g.add_node("Y");
        g.add_edge(t, y, ());
        g.add_edge(x1, t, ());
        g.add_edge(x1, y, ());
        g.add_edge(x2, t, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();
        let identifier = IDIdentifier {};
        if let Ok(estimand) = identifier.id_index(&y_set, &t_set, &g) {
            assert_eq!(format!("{}", estimand), "∑{X1}(P(X1)P(Y|X2,X1,T))");
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_no_direct_cause_dag() {
        let mut g = ADMG::<&str>::new();
        let t = g.add_node("T");
        let x = g.add_node("X");
        let y = g.add_node("Y");
        g.add_edge(x, y, ());
        let t_set: OrderedSet<NodeIndex> = [t].iter().copied().collect();
        let y_set: OrderedSet<NodeIndex> = [y].iter().copied().collect();
        let identifier = IDIdentifier {};
        if let Ok(estimand) = identifier.id_index(&y_set, &t_set, &g) {
            assert_eq!(format!("{}", estimand), "∑{X}(P(X,Y))");
        } else {
            unreachable!();
        }
    }
}
