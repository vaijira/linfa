use crate::setutils::OrderedSet;
use anyhow::{bail, Result};
use petgraph::graph::UnGraph;
use petgraph::stable_graph::{NodeIndex, StableGraph};
use petgraph::visit::Dfs;
use std::cmp::Ord;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::ops::{Deref, DerefMut};

pub(crate) type Graph<T> = StableGraph<T, ()>;

#[derive(Clone)]
struct BidirectedEdge<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    node1: T,
    node2: T,
}

/// Acyclic directed mixed graph.
/// This graph is use to specify a causal model it can contains directed edges
/// specifying causal relations in a directed acyclic graph or bidirected edges
/// for unobserved or unknown relations.
pub struct ADMG<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    graph: Graph<T>,
    bidirected_edges: Vec<BidirectedEdge<T>>,
}

impl<T> Deref for ADMG<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    type Target = Graph<T>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl<T> DerefMut for ADMG<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl<T> Default for ADMG<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ADMG<T>
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    pub fn new() -> Self {
        ADMG {
            graph: Graph::new(),
            bidirected_edges: vec![],
        }
    }

    pub fn graph(&self) -> &Graph<T> {
        &self.graph
    }

    pub fn add_bidirected_edge(&mut self, node1: T, node2: T) -> Result<()> {
        if node1 == node2 {
            bail!("Self reference bidirected edges within node are not allowed")
        }

        let edge = BidirectedEdge { node1, node2 };

        self.bidirected_edges.push(edge);

        Ok(())
    }

    pub fn c_components(&self, topo_order: &[NodeIndex]) -> Vec<OrderedSet<NodeIndex>> {
        let mut ug: UnGraph<T, ()> =
            UnGraph::with_capacity(self.graph().node_count(), self.bidirected_edges.len());
        let mut node2index: BTreeMap<&T, NodeIndex> = BTreeMap::new();
        let mut new_index2index: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        for node_index in topo_order {
            if let Some(w) = self.graph().node_weight(*node_index) {
                let i = ug.add_node(w.clone());
                node2index.insert(w, i);
                new_index2index.insert(i, *node_index);
            }
        }

        for edge in self.bidirected_edges.iter() {
            ug.add_edge(
                *node2index.get(&edge.node1).unwrap(),
                *node2index.get(&edge.node2).unwrap(),
                (),
            );
        }

        let mut c_components = vec![];
        let mut visited: OrderedSet<NodeIndex> = OrderedSet::new();

        for (_, new_index) in node2index.iter() {
            if !visited.contains(new_index) {
                let mut component: OrderedSet<NodeIndex> = OrderedSet::new();
                component.insert(*new_index2index.get(new_index).unwrap());
                let mut dfs = Dfs::new(&ug, *new_index);
                while let Some(nx) = dfs.next(&ug) {
                    visited.insert(nx);
                    component.insert(*new_index2index.get(new_index).unwrap());
                }
                c_components.push(component);
            }
        }

        c_components
    }

    pub fn upper_bar_variables(&self, nodes: &OrderedSet<NodeIndex>) -> ADMG<T> {
        let g = self.graph.filter_map(
            |_, node| Some(node.clone()),
            |index, _| {
                if let Some((_, n2)) = self.graph.edge_endpoints(index) {
                    if nodes.contains(&n2) {
                        None
                    } else {
                        Some(())
                    }
                } else {
                    None
                }
            },
        );

        ADMG {
            graph: g,
            bidirected_edges: self.bidirected_edges.clone(),
        }
    }

    pub fn induced_subgraph(&self, nodes: &OrderedSet<NodeIndex>) -> ADMG<T> {
        let g = self.graph.filter_map(
            |index, node| {
                if nodes.contains(&index) {
                    Some(node.clone())
                } else {
                    None
                }
            },
            |index, _| {
                if let Some((n1, n2)) = self.graph.edge_endpoints(index) {
                    if nodes.contains(&n1) || nodes.contains(&n2) {
                        Some(())
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
        );

        ADMG {
            graph: g,
            bidirected_edges: self.bidirected_edges.clone(),
        }
    }

    /// Returns the ancestors of a set of nodes.
    /// The nodes you are calculating the nodes from are included in the output,
    /// to follow causal inference literature meaning.
    pub fn ancestors(&self, nodes: &OrderedSet<NodeIndex>) -> OrderedSet<NodeIndex> {
        let mut ancestors = nodes.clone();
        let mut pending_nodes: OrderedSet<NodeIndex> = nodes.clone();

        while !pending_nodes.is_empty() {
            let node_index = crate::setutils::pop(&mut pending_nodes);
            for node in self
                .graph
                .neighbors_directed(node_index, petgraph::EdgeDirection::Incoming)
            {
                if !ancestors.contains(&node) {
                    ancestors.insert(node);
                    pending_nodes.insert(node);
                }
            }
        }

        ancestors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_induced_subgraph() {
        let mut causal_graph = ADMG::<&str>::new();
        let t = causal_graph.add_node("T");
        let y = causal_graph.add_node("Y");
        causal_graph.add_edge(t, y, ());

        let mut filtered = OrderedSet::new();
        filtered.insert(y);

        let mut result = OrderedSet::new();
        result.insert(y);

        assert_eq!(
            causal_graph
                .induced_subgraph(&filtered)
                .c_components(&[t, y]),
            vec![result]
        );
    }

    #[test]
    fn test_c_components() {
        let mut causal_graph = ADMG::<&str>::new();
        let a = causal_graph.add_node("A");
        let b = causal_graph.add_node("B");
        let c = causal_graph.add_node("C");

        let c_components = causal_graph.c_components(&[a, b, c]);

        let mut a_set: OrderedSet<NodeIndex> = OrderedSet::new();
        a_set.insert(a);
        let mut b_set: OrderedSet<NodeIndex> = OrderedSet::new();
        b_set.insert(b);
        let mut c_set: OrderedSet<NodeIndex> = OrderedSet::new();
        c_set.insert(c);

        assert_eq!(c_components, vec![a_set, b_set, c_set]);
    }

    #[test]
    fn test_ancestors() {
        let mut causal_graph = ADMG::<&str>::new();
        let t = causal_graph.add_node("T");
        let y = causal_graph.add_node("Y");
        causal_graph.add_edge(t, y, ());

        let mut outcomes = OrderedSet::new();
        outcomes.insert(y);

        let mut result = OrderedSet::new();
        result.insert(t);
        result.insert(y);

        assert_eq!(causal_graph.ancestors(&outcomes), result);

        let v0 = causal_graph.add_node("V0");
        let v1 = causal_graph.add_node("V1");
        let x = causal_graph.add_node("X");
        causal_graph.add_edge(v0, t, ());
        causal_graph.add_edge(v1, t, ());
        causal_graph.add_edge(x, v0, ());
        causal_graph.add_edge(x, v1, ());

        result.insert(v0);
        result.insert(v1);
        result.insert(x);
        result.insert(y);

        assert_eq!(causal_graph.ancestors(&outcomes), result);
    }
}
