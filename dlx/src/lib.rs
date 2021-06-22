//! Dancing Links solver for “algorithm X” by Knuth
//!
//! This solver solves the exact cover problem using “algorithm X”, implemented using Dancing Links
//! (“Dlx”).

use std::fmt;
use std::iter::repeat;

// Direction of list link
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Prev,
    Next,
    Up,
    Down,
}
use Direction::*;

impl Direction {
    /// Get opposite direction.
    ///
    /// Opposites:
    /// - Prev and Next
    /// - Up and Down
    #[inline(always)]
    fn opp(self) -> Direction {
        match self {
            Prev => Next,
            Next => Prev,
            Up => Down,
            Down => Up,
        }
    }
}

/// Link node in the Dancing Links structure,
/// which is linked along two axes - prev/next and up/down.
#[derive(Copy, Clone, Default, PartialEq)]
pub(crate) struct Node<T> {
    /// Prev, Next, Up, Down
    link: [Index; 4],
    pub(crate) value: T,
}

/// Internal index type used by the Node
type Index = usize;

impl<T> fmt::Debug for Node<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // avoid "pretty" debug
        // link as isize so that !0 shows as -1.
        write!(f, "{} {{ {}: [{:3}, {:3}; {:3}, {:3}], {}: {} }}",
            stringify!(Node), stringify!(link),
            self.link[0] as isize, self.link[1] as isize,
            self.link[2] as isize, self.link[3] as isize,
            stringify!(value), &format_args!("{:?}", self.value))
    }
}

impl<T> Node<T> {
    /// Create a new node from the value.
    fn new(value: T) -> Self {
        Node {
            value,
            link: [!0; 4], // invalid link values to start with
        }
    }

    /// Get link in the given direction
    fn get(&self, dir: Direction) -> Index {
        self.link[dir as usize]
    }

    /// Set link in the given direction
    fn set(&mut self, dir: Direction, index: Index) -> &mut Self {
        self.link[dir as usize] = index;
        self
    }

    /// Assign link in the given direction
    fn assign(&mut self, dir: Direction) -> &mut Index {
        &mut self.link[dir as usize]
    }
}

// These macros are used trace debug logging

#[cfg(feature="trace")]
macro_rules! if_trace {
    ($($t:tt)*) => { $($t)* }
}

#[cfg(not(feature="trace"))]
macro_rules! if_trace {
    ($($t:tt)*) => { }
}

macro_rules! trace {
    ($($t:tt)*) => { if_trace!(eprintln!($($t)*)) }
}


/// Universe integer type
pub type UInt = u32;

/// Dancing Links structure
///
/// This is a “Dancing Links” data structure. The structure corresponds to a sparse binary matrix
/// and it uses two-dimensional doubly linked lists.
///
/// See Knuth for papers about this structure and about “algorithm X”.
#[derive(Debug, Clone, PartialEq)]
pub struct Dlx {
    /// Node layout in DLX:
    ///
    /// ```text
    ///
    ///            ..    ..    ..
    ///            ||    ||    ||
    /// :> Head <> C1 <> C2 <> C3 <> ... <:    (Head and column heads)
    ///            ||    ||    ||
    ///         :> R1  <    >  R2  <       ..  (Row items)
    ///            ||    ||          ||
    ///               :> R3  <    >  R4  < ..
    ///                  ||          ||
    ///                  ..          ..
    ///
    ///   ... etc.
    ///   where || are a Up/Down links and <> Prev/Next links.
    /// ```
    ///
    /// Head is only linked to the column row.
    /// Note that R1 links directly to R2 and so on, the matrix is sparse.
    ///
    /// The lists are doubly linked and circular, in two dimensions: Prev, Next
    /// and Up, Down.
    ///
    /// All the column heads and row items can “dance”: you can remove them and
    /// then restore them again.
    /// The head node is never removed, it always stays, and when the matrix is
    /// empty, it links to itself in all directions.
    ///
    /// All nodes carry a value and how it is used is described under Point.
    nodes: Vec<Node<Point>>,
    /// Number of columns
    columns: UInt,
    /// Index with the start of each row (sorted, ascending order);
    /// used for lookup from node index to row index.
    row_table: Vec<Index>,
}

/// Value stored inside the node.
///
/// The variant indentifies the kind of node,
/// and the number is used as indicated.
#[derive(Debug, Copy, Clone, PartialEq)]
enum Point {
    /// Singleton head node before all columns; value ignored.
    Head(UInt),
    /// Column head with counter for items alive in the column
    Column(UInt),
    /// Row body item, with column number for reference to column header
    Body(UInt),
}

impl Point {
    #[inline]
    pub(crate) fn value(&self) -> UInt {
        use Point::*;

        match self {
            Head(_) => debug_assert!(false, "Possible error: no need to access Head's value"),
            Column(_) | Body(_) => {}
        }

        match *self {
            Head(x) | Column(x) | Body(x) => x
        }
    }

    #[inline]
    pub(crate) fn value_mut(&mut self) -> &mut UInt {
        use Point::*;

        match self {
            Head(_) | Body(_) => debug_assert!(false, "Possible error: no need to modfiy Head, Body"),
            Column(_) => {}
        }

        match self {
            Head(x) | Column(x) | Body(x) => x
        }
    }
}

fn enumerate<T>(it: impl IntoIterator<Item=T>) -> impl Iterator<Item=(usize, T)> {
    it.into_iter().enumerate()
}

/// Error that can occur in the setup of Dlx.
#[derive(Debug, Clone, PartialEq)]
pub enum DlxError {
    InputMustNotBeEmpty,
    InputOutsideUniverse,
    InputNotInSortedOrder,
    InvalidColumnZero,
}

impl Dlx {
    /// Create a new Dlx with the universe of points in 1..=universe
    pub fn new(universe: UInt) -> Self {
        let mut dlx = Dlx {
            nodes: Vec::with_capacity(4 * universe as usize),
            columns: universe,
            row_table: Vec::new(),
        };
        dlx.initialize(universe);
        dlx
    }

    /// Clear the Dlx
    ///
    /// It is reinitialized for use with the universe of points in 1..=universe.
    pub fn reset(&mut self, universe: UInt) {
        self.nodes.clear();
        self.row_table.clear();
        self.columns = universe;
        self.initialize(universe);
    }

    fn initialize(&mut self, universe: UInt) {
        // Insert head node and the column row.
        let nodes = &mut self.nodes;
        nodes.push(Node::new(Point::Head(0)));
        nodes.extend(repeat(Node::new(Point::Column(0))).take(universe as usize));

        // link the whole header row in both dimensions
        for (index, node) in enumerate(&mut *nodes) {
            // self-link in up-down axis
            *node.assign(Up) = index;
            *node.assign(Down) = index;
            *node.assign(Prev) = index.wrapping_sub(1);
            *node.assign(Next) = index + 1;
        }
        // fixup begin/end
        let len = nodes.len();
        *nodes[0].assign(Prev) = len - 1;
        *nodes[len - 1].assign(Next) = 0;
    }

    #[inline]
    fn head(&self) -> Index { 0 }

    #[inline]
    fn head_node(&self) -> &Node<Point> {
        &self.nodes[0]
    }

    #[inline]
    fn head_node_mut(&mut self) -> &mut Node<Point> {
        &mut self.nodes[0]
    }

    /// Get the user-defined tag value for this Dlx
    pub fn tag(&self) -> UInt {
        match self.head_node().value {
            Point::Head(v) => v,
            Point::Column(_) | Point::Body(_) => unreachable!(),
        }
    }

    /// Set a user-defined tag value for this Dlx
    ///
    /// The tag has no effect for the algorithm, but it can be used to store an integer
    /// of information about the Dlx.
    pub fn set_tag(&mut self, tag: UInt) {
        match self.head_node_mut().value {
            Point::Head(ref mut v) => *v = tag,
            Point::Column(_) | Point::Body(_) => unreachable!(),
        }
    }

    /// Create a borrowless traversal state that can walk the linked lists.
    ///
    /// The walk finishes when the starting point is reached (the starting point is not emitted
    /// anywhere in the walk).
    pub(crate) fn walk_from(&self, index: Index) -> Walker {
        Walker { index, start: index }
    }

    pub(crate) fn get_value(&self, index: Index) -> UInt {
        self.nodes[index].value.value()
    }

    /// Get the column head for row item `index`
    ///
    /// `index` must be a row item.
    pub(crate) fn get_column_head_of(&self, index: Index) -> Index {
        debug_assert!(index > self.columns as Index, "Expected row item index, got {}", index);
        self.get_value(index) as Index
    }

    /// Get the mutable value of the row item's column head
    pub(crate) fn column_head_value_mut(&mut self, index: Index) -> &mut UInt {
        match self.get_column_head_of(index) {
            chead => self.nodes[chead].value.value_mut()
        }
    }

    fn append_to_column(&mut self, col: UInt, new_index: Index) {
        debug_assert!(col <= self.columns && col != 0, "invalid column {}", col);
        debug_assert!(new_index < self.nodes.len(), "invalid index {}", new_index);
        debug_assert!(matches!(self.nodes[new_index].value, Point::Body(_)));
        let head_index = col as Index;
        let head = &mut self.nodes[head_index];

        let former_end = head.get(Up);
        head.set(Up, new_index);
        *head.value.value_mut() += 1;
        self.nodes[former_end].set(Down, new_index);
        self.nodes[new_index]
            .set(Up, former_end)
            .set(Down, head_index);
    }

    /// Append a row (a subset) to the Dlx
    ///
    /// The items of the row use one-based indexing and must be in ascending order;
    /// the items must be in 1..=universe.
    ///
    /// Empty rows are not allowed. The universe is allowed to be empty, but don't add any
    /// rows or “subsets” in that case.
    pub fn append_row(&mut self, row: impl IntoIterator<Item=UInt>) -> Result<(), DlxError> {
        // try creating nodes for all items
        let start_index = self.nodes.len();
        let try_append = (|| {
            let mut max_seen = None;
            for r in row {
                if let Some(ms) = max_seen {
                    if ms >= r {
                        return Err(DlxError::InputNotInSortedOrder);
                    }
                }
                if r == 0 {
                    return Err(DlxError::InvalidColumnZero);
                }
                if r > self.columns {
                    return Err(DlxError::InputOutsideUniverse);
                }
                max_seen = Some(r);
                let body_node = Node::new(Point::Body(r));
                self.nodes.push(body_node);
            }

            if let None = max_seen {
                return Err(DlxError::InputMustNotBeEmpty);
            }
            Ok(())
        })();

        if let Err(_) = try_append {
            // roll back changes on error (only changes are .push() calls so far)
            self.nodes.truncate(start_index);
            return try_append;
        }

        // after error checks,
        // append new items to each column
        for index in start_index..self.nodes.len() {
            self.append_to_column(self.nodes[index].value.value(), index);
        }

        // now link prev-next axis
        let end_index = self.nodes.len();
        for (index, node) in enumerate(&mut self.nodes[start_index..]) {
            let prev_index = if index == 0 { end_index - 1 } else { start_index + index - 1 };
            let next_index = if start_index + index + 1 == end_index { start_index } else { start_index + index + 1 };
            node.set(Prev, prev_index);
            node.set(Next, next_index);
        }
        self.row_table.push(start_index);

        Ok(())
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.row_table.len()
    }

    #[cfg(test)]
    pub(crate) fn assert_links(&self) {
        for (node_i, node) in enumerate(&self.nodes) {
            for &i in &node.link {
                assert_ne!(i, !0, "Uninitialized link {} for node {}", i, node_i);
                assert!(i < self.nodes.len(), "Out of bounds link {} for node {}", i, node_i);
            }
        }
    }

    /// Get row index for node index
    pub(crate) fn row_index_of(&self, index: Index) -> usize {
        let pos = self.row_table.partition_point(move |&x| x <= index);
        debug_assert_ne!(pos, 0, "solution contains index before first row");
        pos - 1
    }

    /// Return solution as the row indexes (zero-indexed)
    pub(crate) fn solution_to_rows(&self, sol: &[Index]) -> Vec<UInt> {
        // Given a table like
        // [8, 11, 13, 17]
        // we map indexes to:
        // 8, 9, 10 => 0
        // 11, 12 => 1
        // 13 => 2
        // 17, 18 => 3
        sol.iter().map(|&s| self.row_index_of(s) as UInt).collect()
    }

    /// Remove `x` from the list in direction `dir`, where the list is doubly linked.
    /// (It does not matter if the first or second item of direction axis is passed.)
    ///
    /// x.left.right ← x.right;
    /// x.right.left ← x.left;
    pub(crate) fn remove(&mut self, index: Index, dir: Direction) {
        let right = dir;
        let left = dir.opp();
        let x = &self.nodes[index];
        let xr = x.get(right);
        let xl = x.get(left);

        self.nodes[xl].set(right, xr);
        self.nodes[xr].set(left, xl);
    }

    /// Restore `x` to the list, reversing a previous removal.
    /// (It does not matter if the first or second item of direction axis is passed.)
    ///
    /// x.left.right ← x;
    /// x.right.left ← x;
    pub(crate) fn restore(&mut self, index: Index, dir: Direction) {
        let right = dir;
        let left = dir.opp();
        let x = &self.nodes[index];
        let xr = x.get(right);
        let xl = x.get(left);

        self.nodes[xl].set(right, index);
        self.nodes[xr].set(left, index);
    }

    /// Cover column c
    pub(crate) fn cover(&mut self, c: Index) {
        // cover column
        //
        // start from column head c
        // column head c unlinked in Prev/Next
        // step Down to i (loop til c):
        //   step Next from i to j (loop til i):
        //      unlink in Up/Down (Not unlinking i itself)
        //      Decrement j's column's count
        trace!(">> cover column {}", c);
        debug_assert!(c > 0 && c <= self.columns as _,
                      "Not a column head: {}", c);

        self.remove(c, Next);
        let mut rows = self.walk_from(c);
        while let Some(row_i) = rows.next(self, Down) {
            let mut row_i_walk = self.walk_from(row_i);
            while let Some(row_i_j) = row_i_walk.next(self, Next) {
                self.remove(row_i_j, Down);
                *self.column_head_value_mut(row_i_j) -= 1;
            }
        }
    }

    /// Uncover column c
    pub(crate) fn uncover(&mut self, c: Index) {
        // uncover column
        //
        // steps taken in the reverse order of cover.
        //
        // start from column head c
        // step Up as i (loop til c):
        //   step Prev from i to j (loop til i):
        //      restore link in Up/Down (Not i itself)
        //      Increment j's column's count
        // column head c restored in Prev/Next
        trace!("<< uncover column {}", c);
        debug_assert!(c > 0 && c <= self.columns as _,
                      "Not a column head: {}", c);

        let mut rows = self.walk_from(c);
        while let Some(row_i) = rows.next(self, Down.opp()) {
            let mut row_i_walk = self.walk_from(row_i);
            while let Some(row_i_j) = row_i_walk.next(self, Next.opp()) {
                self.restore(row_i_j, Down);
                *self.column_head_value_mut(row_i_j) += 1;
            }
        }
        self.restore(c, Next);
    }

    /// Print a debug representation of the Dlx
    pub fn debug_print(&self) {
        let n_blocks = self.nodes.len().saturating_sub(1 + self.columns as usize);
        eprintln!("Dlx columns={}, rows={}, nodes={} (blocks={})",
            self.columns, self.nrows(), self.nodes.len(), n_blocks);

        let mut visible_rows = vec![None; self.nrows() as usize];

        let mut headings = self.walk_from(self.head());
        let mut ncols = 0;
        eprint!("{:?} ", self.head_node().value);
        while let Some(col_head) = headings.next(self, Next) {
            eprint!("{:4} ", self.nodes[col_head].value.value());

            let mut col_items = self.walk_from(col_head);
            while let Some(r) = col_items.next(self, Down) {
                let ri = self.row_index_of(r);
                visible_rows[ri].get_or_insert(r);
            }
            ncols += 1;
        }
        eprint!("({} columns)", ncols);
        eprintln!();

        for row_head in visible_rows.iter().filter_map(|x| x.as_ref().copied()) {
            let index = self.row_index_of(row_head);
            eprint!("Row({}) {:3}, ", index, self.nodes[row_head].value.value());
            let mut row_items = self.walk_from(row_head);
            while let Some(block) = row_items.next(self, Next) {
                let col_head = self.nodes[block].value.value();
                eprint!("{:3}, ", col_head);
            }
            eprintln!();
        }
    }
}

/// Walker: for borrowless traversal along the linked lists
struct Walker {
    index: Index,
    start: Index,
}

impl Walker {
    /// Take the next step in the walk. The walk finishes when the starting point is reached
    /// (the starting point is not emitted anywhere in the walk).
    pub(crate) fn next(&mut self, dlx: &Dlx, dir: Direction) -> Option<Index> {
        let next = dlx.nodes[self.index].get(dir);
        self.index = next;
        debug_assert_ne!(next, !0, "Invalid index found in traversal");
        if next == self.start {
            None
        } else {
            Some(next)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum XError {
    /// Used internall to stop after the first solution (if enabled)
    RequestedStop
}

/// Statistics from the execution of algorithm X.
#[derive(Clone, Debug, Default)]
pub struct AlgoXStats {
    /// Number of (recursive) calls
    #[cfg(feature = "stats")]
    pub calls: u32,
    /// Number of cover-uncover operations
    #[cfg(feature = "stats")]
    pub cover: u32,
    /// Number of column list walks
    #[cfg(feature = "stats")]
    pub col_seek: u32,
    /// Number of backtracks from dead ends
    #[cfg(feature = "stats")]
    pub backtracks: u32,
    /// Number of solutions emitted
    #[cfg(feature = "stats")]
    pub solutions: u32,
}

/// Configuration for algorithm X
#[derive(Clone, Debug, Default)]
pub struct AlgoXConfig {
    /// If true, stop at first solution
    pub stop_at_first: bool,
    /// If stats are enabled, they are written here if the struct is initialized to Some(_) on entry
    pub stats: Option<AlgoXStats>,
}

/// A solution from algorithm X
///
/// This solution can be converted (on demand) to the row indices it corresponds to.
#[derive(Clone, Debug)]
pub struct AlgoXSolution<'a> {
    raw: &'a [Index],
    dlx: &'a Dlx,
}

impl AlgoXSolution<'_> {
    /// Return length of solution
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Return solution as row identifiers (zero-based row indices)
    pub fn get(&self) -> Vec<UInt> {
        self.dlx.solution_to_rows(&self.raw)
    }
}

/// Knuth's “Algorithm X”, a constraint satisfaction problem solver for the exact cover problem.
///
/// Implemented using Dancing Links.
///
/// - dlx: Problem formulation in terms of a dancing links graph
/// - out: Solution callback, called once for each solution.
///
/// The dlx is mutable, but at the exit of this function, it is always in the same state that
/// it was on entry.
///
/// This version uses the default configuration and emits all solutions.
pub fn algox(dlx: &mut Dlx, out: impl FnMut(AlgoXSolution<'_>)) {
    let mut config = AlgoXConfig::default();
    algox_config(dlx, &mut config, out);
}

/// Knuth's “Algorithm X”, a constraint satisfaction problem solver for the exact cover problem.
///
/// Implemented using Dancing Links.
///
/// - dlx: Problem formulation in terms of a dancing links graph
/// - config: Configuration and statistics
/// - out: Solution callback, called once for each solution.
///
/// The dlx is mutable, but at the exit of this function, it is always in the same state that
/// it was on entry.
pub fn algox_config(dlx: &mut Dlx, config: &mut AlgoXConfig, mut out: impl FnMut(AlgoXSolution<'_>)) {
    trace!("Algorithm X start");
    if_trace!(dlx.debug_print(true));
    if cfg!(feature = "stats_trace") && config.stats.is_none() {
        config.stats = Some(AlgoXStats::default());
    }
    let _ = algox_inner(dlx, &mut Vec::new(), config, &mut out);
    if cfg!(feature = "stats_trace") {
        if let Some(stats) = &config.stats {
            eprintln!("{:#?}", stats);
        }
    }
}

macro_rules! stat {
    ($c:ident . $field:ident $($t:tt)*) => {
        if cfg!(feature = "stats") {
            if let Some(ref mut st) = $c.stats {
                st . $field $($t)*;
            }
        }
    }
}

/// The implementation of algorithm X
///
/// The partial_solution is stored as node indexes; only when a solution is found,
/// will this be converted to row indexes (the actually useful form of solution).
///
/// The solver runs through the whole search tree if not interrupted; the XError status is used to
/// short-circuit and exit as soon as possible if requested.
///
/// The Dlx state is always restored in all ways of exiting the solver, meaning that the Dlx will
/// be unmodified from its starting state once this function returns.
fn algox_inner<F>(dlx: &mut Dlx, partial_solution: &mut Vec<usize>, config: &mut AlgoXConfig, out: &mut F)
    -> Result<(), XError>
where
    F: FnMut(AlgoXSolution<'_>)
{
    /*
    1. If the matrix A has no columns, the current partial solution is a valid solution; terminate successfully.
    2. Otherwise choose a column c (deterministically).
    3. Choose a row r such that Ar, c = 1 (nondeterministically). [This means: all possibilities are explored]
    4. Include row r in the partial solution.
    5. For each column j such that Ar, j = 1,

        for each row i such that Ai, j = 1,

            delete row i from matrix A.

        delete column j from matrix A.

    6. Repeat this algorithm recursively on the reduced matrix A.
    */
    let mut status = Ok(());
    stat!(config.calls += 1);

    // 1. is the matrix empty
    let empty = dlx.head_node().get(Next) == dlx.head();
    if empty {
        // We have a solution
        let xsolution = AlgoXSolution { raw: &partial_solution, dlx: &dlx };
        trace!("==> Valid solution: {:?} ({:?})", dlx.solution_to_rows(&xsolution.raw), xsolution.raw);
        stat!(config.solutions += 1);
        out(xsolution);
        return if config.stop_at_first { Err(XError::RequestedStop) } else { Ok(()) };
    }

    // 2. Pick the least populated column
    let mut col_index = 0;
    {
        let mut min = !0;
        let mut col_heads = dlx.walk_from(dlx.head());
        while let Some(index) = col_heads.next(dlx, Next) {
            stat!(config.col_seek += 1);
            let count = dlx.get_value(index);
            if count < min {
                min = count;
                col_index = index;
                if min == 0 { break; } // found a minimum
            }
        }

        if min == 0 {
            trace!("Column {} unsatsified, backtracking", col_index);
            stat!(config.backtracks += 1);
            return Ok(());
        }
        trace!("Selected col_index = {} with population = {}", col_index, min);
    }

    // 3. Explore the rows in the chosen column

    // cover column
    dlx.cover(col_index);
    stat!(config.cover += 1);

    // now cover other columns sharing a one with this one
    let mut col_items = dlx.walk_from(col_index);
    while let Some(col_i) = col_items.next(dlx, Down) {

        // 4. Include row r in the partial solution
        partial_solution.push(col_i);
        trace!("partial_solution {:?}", partial_solution);

        // 5. Cover each column
        let mut row_iter = dlx.walk_from(col_i);
        while let Some(row_j) = row_iter.next(dlx, Next) {
            dlx.cover(dlx.get_column_head_of(row_j));
            stat!(config.cover += 1);
        }

        // 6. Repeat this algorithm recursively on the reduced matrix A.
        trace!("Recurse!");
        status = algox_inner(dlx, partial_solution, config, out);

        let _ = partial_solution.pop();

        let mut row_iter = dlx.walk_from(col_i);
        while let Some(row_j) = row_iter.next(dlx, Next.opp()) {
            dlx.uncover(dlx.get_column_head_of(row_j));
        }

        if status.is_err() {
            break;
        }
    }
    dlx.uncover(col_index);
    status
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut dlx = Dlx::new(3);
        println!("{:#?}", dlx);
        dlx.append_row([1, 3]).unwrap();
        dlx.append_row([2]).unwrap();
        dlx.append_row([2, 3]).unwrap();
        dlx.set_tag(1729);
        println!("{:#?}", dlx);
        assert_eq!(dlx.get_value(1), 1);
        assert_eq!(dlx.get_value(2), 2);
        assert_eq!(dlx.get_value(3), 2);
        dlx.assert_links();
        dlx.debug_print();
        assert_eq!(dlx.tag(), 1729);
    }

    #[test]
    fn wiki_example() {
        /*
        A = {1, 4, 7};
        B = {1, 4};
        C = {4, 5, 7};
        D = {3, 5, 6};
        E = {2, 3, 6, 7}; and
        F = {2, 7}.

        with solution B, D, F (indices 1, 3, 5)
        */
        let mut dlx = Dlx::new(7);
        dlx.append_row([1, 4, 7]).unwrap();
        dlx.append_row([1, 4]).unwrap();
        dlx.append_row([4, 5, 7]).unwrap();
        dlx.append_row([3, 5, 6]).unwrap();
        dlx.append_row([2, 3, 6, 7]).unwrap();
        dlx.append_row([2, 7]).unwrap();
        println!("{:#?}", dlx);
        dlx.debug_print();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, Some(vec![1, 3, 5]), "solution mismatch");
    }

    #[test]
    fn dlx_paper_example() {
        /*
        a = {3, 5, 6};
        b = {1, 4, 7};
        c = {2, 3, 6};
        d = {1, 4};
        e = {2, 7}; and
        f = {4, 5, 7}.

        Solution is:
            {1, 4}
            {3, 5, 6}
            {2, 7}
         with indices 3, 0, 4
        */
        let mut dlx = Dlx::new(7);
        dlx.append_row([3, 5, 6]).unwrap();
        dlx.append_row([1, 4, 7]).unwrap();
        dlx.append_row([2, 3, 6]).unwrap();
        dlx.append_row([1, 4]).unwrap();
        dlx.append_row([2, 7]).unwrap();
        dlx.append_row([4, 5, 7]).unwrap();
        println!("{:#?}", dlx);
        dlx.debug_print();
        dlx.assert_links();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, Some(vec![3, 0, 4]), "solution mismatch");
    }

    #[test]
    fn dlx_size7_no_sol() {
        /*
        a = {3, 5, 7};
        b = {1, 4, 7};
        c = {2, 3, 6};
        d = {1, 4};
        e = {2, 7}; and
        f = {4, 5, 7}.
        No solution
        */
        let mut dlx = Dlx::new(7);
        dlx.append_row([3, 5, 7]).unwrap();
        dlx.append_row([1, 4, 7]).unwrap();
        dlx.append_row([2, 3, 6]).unwrap();
        dlx.append_row([1, 4]).unwrap();
        dlx.append_row([2, 7]).unwrap();
        dlx.append_row([4, 5, 7]).unwrap();
        println!("{:#?}", dlx);
        dlx.debug_print();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, None, "solution mismatch");
    }

    #[test]
    fn dlx_size0_triv() {
        let mut dlx = Dlx::new(0);
        println!("{:#?}", dlx);
        dlx.debug_print();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, Some(vec![]), "solution mismatch");
    }

    #[test]
    fn dlx_size2_triv() {
        /*
        a = {1};
        b = {2};
        Solution is 0, 1
        */
        let mut dlx = Dlx::new(2);
        dlx.append_row([1]).unwrap();
        dlx.append_row([2]).unwrap();
        println!("{:#?}", dlx);
        dlx.debug_print();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, Some(vec![0, 1]), "solution mismatch");
    }

    #[test]
    fn dlx_size2_no_sol() {
        /*
        a = {1};
        b = {1};
        No solution
        */
        let mut dlx = Dlx::new(2);
        dlx.append_row([1]).unwrap();
        dlx.append_row([1]).unwrap();
        println!("{:#?}", dlx);
        dlx.debug_print();
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s.get()));
        dlx.debug_print();
        assert_eq!(solution, None, "solution mismatch");
    }

    #[test]
    fn dlx_err_zero() {
        let mut dlx = Dlx::new(2);
        dlx.append_row([1, 2]).unwrap();
        let err = dlx.append_row([0, 1]);
        assert!(err.is_err());
        let err = dlx.append_row([1, 3]);
        assert!(err.is_err());
        assert_eq!(dlx.nrows(), 1);
        dlx.assert_links();
        println!("{:#?}", dlx);
        dlx.debug_print();
    }

    #[test]
    fn dlx_solution_convert() {
        // [8, 11, 13, 17]
        //
        // 8, 9, 10 => 0
        // 11, 12, 13 => 1,
        let dlx = Dlx {
            nodes: Vec::new(),
            columns: 0,
            row_table: vec![8, 11, 13, 17],
        };

        assert_eq!(dlx.solution_to_rows(&[8, 9, 10, 11, 12, 13, 14, 17, 23]),
                   vec![0, 0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn dlx_size3_multi_1() {
        let mut dlx = Dlx::new(4);
        dlx.append_row([3]).unwrap();
        dlx.append_row([1, 2]).unwrap();
        dlx.append_row([1, 2, 3]).unwrap();
        dlx.append_row([1, 4]).unwrap();
        dlx.append_row([1, 2, 4]).unwrap();
        dlx.append_row([4]).unwrap();
        let dlx_old = dlx.clone();
        println!("{:#?}", dlx);
        dlx.debug_print();

        let mut solutions = Vec::new();
        {
            algox(&mut dlx, |s| solutions.push(s.get()));
            println!("{:#?}", dlx);
            println!("{:?}", solutions);
            assert_eq!(solutions.len(), 3);
            assert_eq!(dlx, dlx_old, "Dlx should be restored after run");
        }

        // now just the first found solution
        {
            let first_solution = solutions.remove(0);
            solutions.clear();

            let mut config = AlgoXConfig {
                stop_at_first: true,
                stats: Some(AlgoXStats::default()),
            };
            algox_config(&mut dlx, &mut config, |s| solutions.push(s.get()));

            println!("{:#?}", config.stats);
            println!("{:?}", solutions);
            assert_eq!(solutions.len(), 1);
            assert_eq!(dlx, dlx_old, "Dlx should be restored after run");
            assert_eq!(solutions[0], first_solution);
        }
    }
}
