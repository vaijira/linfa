use std::collections::BTreeSet;
use std::hash::Hash;

pub(crate) type OrderedSet<T> = BTreeSet<T>;

// split off an arbitrary element from a (non-empty) set
pub(crate) fn pop<T>(set: &mut OrderedSet<T>) -> T
where
    T: Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    let elt = set.iter().next().cloned().unwrap();
    set.remove(&elt);
    elt
}

pub(crate) fn setdiff<T>(s1: &OrderedSet<T>, s2: &OrderedSet<T>) -> OrderedSet<T>
where
    T: Copy + Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    s1.difference(s2).into_iter().copied().collect()
}

pub(crate) fn setunion<T>(s1: &OrderedSet<T>, s2: &OrderedSet<T>) -> OrderedSet<T>
where
    T: Copy + Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    s1.union(s2).into_iter().copied().collect()
}

pub(crate) fn setintersect<T>(s1: &OrderedSet<T>, s2: &OrderedSet<T>) -> OrderedSet<T>
where
    T: Copy + Clone + PartialEq + Eq + Clone + Hash + PartialOrd + Ord,
{
    s1.intersection(s2).into_iter().copied().collect()
}
