// from bluss dlx solver

#[cfg(feature="trace")]
macro_rules! trace {
    ($($t:tt)*) => { eprintln!($($t)*) }
}

#[cfg(feature="trace")]
macro_rules! if_trace {
    ($($t:tt)*) => { $($t)* }
}

#[cfg(not(feature="trace"))]
macro_rules! trace {
    ($($t:tt)*) => { }
}

#[cfg(not(feature="trace"))]
macro_rules! if_trace {
    ($($t:tt)*) => { }
}


use std::cmp::Ordering;
use std::fmt;
use std::convert::TryFrom;
use std::iter::repeat;

type Index = usize;

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

impl TryFrom<usize> for Direction {
    type Error = ();
    fn try_from(x: usize) -> Result<Self, Self::Error> {
        match x {
            0 => Ok(Prev),
            1 => Ok(Next),
            2 => Ok(Up),
            3 => Ok(Down),
            _ => Err(()),
        }
    }
}

/// Link node in the Dancing Links structure,
/// which is linked along two axes - prev/next and up/down.
#[derive(Copy, Clone, Default, PartialEq)]
pub(crate) struct Node<T> {
    /// Prev, Next, Up, Down
    link: [usize; 4],
    pub(crate) value: T,
}

macro_rules! lfmt {
    ($x:expr) => {
        if $x == !0 { -1 } else { $x as isize }
    }
}

impl<T> fmt::Debug for Node<T> where T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(stringify!(Node))
            .field(stringify!(link), &format_args!("[{}, {}; {}, {}]",
                lfmt!(self.link[0]), lfmt!(self.link[1]), lfmt!(self.link[2]), lfmt!(self.link[3])))
            .field(stringify!(value), &format_args!("{:?}", self.value))
            .finish()
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
    fn assign(&mut self, dir: Direction) -> &mut usize {
        &mut self.link[dir as usize]
    }
}

pub type UInt = u32;
pub type Int = i32;

#[derive(Debug, Clone, PartialEq)]
pub struct Dlx {
    /// Node layout in DLX:
    /// [ Head ]    [ Columns ... ]
    /// [ Row items ... ]
    /// [ Row items ... ]
    /// ... etc.
    ///
    /// Doubly linked list in two dimensions: Prev, Next and Up, Down.
    pub(crate) nodes: Vec<Node<Point>>,
    columns: UInt,
    rows: UInt,
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

#[derive(Debug, Clone, PartialEq)]
pub enum DlxError {
    InvalidRow(&'static str),
    InvalidInput(&'static str),
    HeadHasNoColumn,
}

impl Dlx {
    /// Create a new Dlx with the universe of points in 1..=universe
    pub fn new(universe: UInt) -> Self {
        // Insert head node and the column row.
        let mut nodes = vec![Node::new(Point::Head(0))];
        nodes.extend(repeat(Node::new(Point::Column(0))).take(universe as usize));

        // link the whole header row in both dimensions
        for (index, node) in enumerate(&mut nodes) {
            // self-link in up-down axis
            *node.assign(Up) = index;
            *node.assign(Down) = index;
            let prev_index = if index == 0 { universe as Index } else { index - 1 };
            let next_index = if index == universe as Index { 0 } else { index + 1 };
            *node.assign(Prev) = prev_index;
            *node.assign(Next) = next_index;
        }

        Dlx {
            nodes,
            columns: universe,
            rows: 0,
            row_table: Vec::new(),
        }
    }

    fn head(&self) -> Index { 0 }

    fn head_node(&self) -> &Node<Point> {
        &self.nodes[0]
    }

    fn column_head(&self, col: UInt) -> Index {
        debug_assert!(col <= self.columns);
        col as Index
    }

    #[cfg(test)]
    fn column_count(&self, col: UInt) -> UInt {
        assert!(col <= self.columns);
        self.nodes[self.column_head(col)].value.value()
    }

    pub(crate) fn walk_from(&self, index: Index) -> Walker {
        Walker {
            index,
            start: index,
        }
    }

    pub(crate) fn get_value(&self, index: Index) -> UInt {
        self.nodes[index].value.value()
    }

    /// Get the column head for row item `index`
    pub(crate) fn col_head_of(&self, index: Index) -> Result<Index, DlxError> {
        let col_head = match self.nodes[index].value {
            Point::Body(c) => self.column_head(c),
            _otherwise => return Err(DlxError::InvalidRow("Expected body point")),
        };
        Ok(col_head)
    }

    pub(crate) fn modify_col_head_of(&mut self, index: Index, incr: Int) {
        let c = self.col_head_of(index).unwrap();
        let v = self.nodes[c].value.value_mut();
        *v = (*v as Int + incr) as UInt;
    }

    fn append_to_column(&mut self, col: UInt, new_index: Index) {
        debug_assert!(col <= self.columns && col != 0, "invalid column {}", col);
        debug_assert!(new_index < self.nodes.len(), "invalid index {}", new_index);
        debug_assert!(matches!(self.nodes[new_index].value, Point::Body(_)));
        let head_index = col as Index;
        let head = &mut self.nodes[head_index];
        let old_end = head.get(Up);
        head.set(Up, new_index);
        *head.value.value_mut() += 1;
        self.nodes[old_end].set(Down, new_index);
        self.nodes[new_index].set(Up, old_end);
        self.nodes[new_index].set(Down, head_index);
    }

    /// Append a row (a subset) to the Dlx
    ///
    /// The items of the row use one-based indexing and must be in ascending order;
    /// the items must be in 1..=universe.
    pub fn append_row(&mut self, row: impl IntoIterator<Item=UInt>) -> Result<(), DlxError> {
        // try creating nodes for all items
        let start_index = self.nodes.len();
        let try_append = (|| {
            let mut max_seen = None;
            for r in row {
                if let Some(ms) = max_seen {
                    if ms >= r {
                        return Err(DlxError::InvalidRow("invalid order"));
                    }
                }
                if r == 0 {
                    return Err(DlxError::InvalidRow("invalid column zero"));
                }
                if r > self.columns {
                    return Err(DlxError::InvalidRow("row larger than column count"));
                }
                max_seen = Some(r);
                let body_node = Node::new(Point::Body(r));
                self.nodes.push(body_node);
            }

            if let None = max_seen {
                return Err(DlxError::InvalidRow("must not be empty"));
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
        self.rows += 1;
        self.row_table.push(start_index);

        Ok(())
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
        let pos = self.row_table.binary_search_by(move |&x| {
            if x <= index {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }).unwrap_err(); /* never equal */
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
        let mut res = Vec::with_capacity(sol.len());
        for &s in sol {
            let pos = self.row_index_of(s);
            res.push(pos as UInt);
        }
        res
    }

    /// Remove `x` from the list in direction `dir`, where the list is doubly linked.
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
    ///
    /// x.left.right ← x;
    /// x.right.left ← x;
    pub(crate) fn restore(&mut self, index: Index, dir: Direction) {
        let right = dir;
        let left = dir.opp();
        let x = index;
        let xr = self.nodes[x].get(right);
        let xl = self.nodes[x].get(left);

        self.nodes[xl].set(right, x);
        self.nodes[xr].set(left, x);
    }

    /// Cover column c
    pub(crate) fn cover(&mut self, c: Index) {
        // cover column
        //
        // start from column head c
        // column head c unlinked in Prev/Next
        // step Down in rows to i
        //   Go Next in row and unlink in Up/Down
        //   (Not unlinking i itself)
        //   Decrement column's count
        trace!("cover column {}", c);
        debug_assert!(c > 0 && c <= self.columns as _,
                      "Not a column head: {}", c);

        self.remove(c, Next);
        let mut rows = self.walk_from(c);
        while let Some(row_i) = rows.next(self, Down) {
            let mut row_i_walk = self.walk_from(row_i);
            while let Some(row_i_j) = row_i_walk.next(self, Next) {
                self.remove(row_i_j, Down);
                self.modify_col_head_of(row_i_j, -1);
            }
        }
        //if_trace!(self.format(true));
    }

    /// Uncover column c
    pub(crate) fn uncover(&mut self, c: Index) {
        // uncover column
        //
        // steps taken in the reverse order of cover.
        //
        // start from column head c
        // step Up in rows to i
        //   Go Prev in row and unlink in Up/Down
        //   (Not unlinking i itself)
        //   Increment column's count
        // column head c restored in Prev/Next
        trace!("uncover column {}", c);
        debug_assert!(c > 0 && c <= self.columns as _,
                      "Not a column head: {}", c);

        let mut rows = self.walk_from(c);
        while let Some(row_i) = rows.next(self, Down.opp()) {
            let mut row_i_walk = self.walk_from(row_i);
            while let Some(row_i_j) = row_i_walk.next(self, Next.opp()) {
                self.restore(row_i_j, Down);
                self.modify_col_head_of(row_i_j, 1);
            }
        }
        self.restore(c, Next);
    }

    /// Print a debug representation of the Dlx
    pub fn format(&self, include_rows: bool) {
        let n_blocks = self.nodes.len().saturating_sub(1 + self.columns as usize);
        eprintln!("Dlx columns={}, rows={}, nodes={} (blocks={})",
            self.columns, self.rows, self.nodes.len(), n_blocks);

        let mut visible_rows = vec![None; self.rows as usize];

        let mut headings = self.walk_from(0);
        eprint!("Head  ");
        while let Some(col_head) = headings.next(self, Next) {
            eprint!("{:4} ", self.nodes[col_head].value.value());

            let mut col_iter = self.walk_from(col_head);
            while let Some(r) = col_iter.next(self, Down) {
                let ri = self.row_index_of(r);
                visible_rows[ri].get_or_insert(r);
            }
        }
        eprintln!();

        if !include_rows {
            return;
        }

        for row_head in visible_rows.iter().filter_map(|x| x.as_ref().copied()) {
            let index = self.row_index_of(row_head);
            eprint!("Row({}) {:3}, ", index, self.nodes[row_head].value.value());
            let mut col = self.walk_from(row_head);
            while let Some(block) = col.next(self, Next) {
                let col_head = self.nodes[block].value.value();
                eprint!("{:3}, ", col_head);
            }
            eprintln!();
        }
    }
}

pub struct Walker {
    index: Index,
    start: Index,
}

impl Walker {
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
pub(crate) enum XError { }

#[derive(Clone, Debug, Default)]
pub struct AlgoXStats {
    calls: u32,
    cover: u32,
    col_seek: u32,
}

#[derive(Clone, Debug, Default)]
pub struct AlgoXConfig {
    // TODO: implement this config switch
    emit_all: bool,
    stats: Option<AlgoXStats>,
}

/// Knuth's “Algorithm X”, a constraint satisfaction problem solver for the exact cover problem.
///
/// Implemented using Dancing Links.
///
/// - dlx: Problem formulation in terms of a dancing links graph
/// - out: Solution callback, called once for each solution.
pub fn algox(dlx: &mut Dlx, out: impl FnMut(Vec<UInt>)) {
    let mut config = AlgoXConfig::default();
    if cfg!(feature = "stats_by_default") {
        config.stats = Some(AlgoXStats::default());
    }
    algox_config(dlx, &mut config, out);
}

pub fn algox_config(dlx: &mut Dlx, config: &mut AlgoXConfig, mut out: impl FnMut(Vec<UInt>)) {
    trace!("Algorithm X start");
    algox_inner(dlx, &mut Vec::new(), config, &mut out).unwrap();
    if cfg!(feature = "stats") {
        eprintln!("{:#?}", config.stats);
    }
}

macro_rules! stat {
    ($c:expr, $field:ident, $($t:tt)*) => {
        if cfg!(feature = "stats") {
            if let Some(ref mut st) = $c.stats {
                st . $field $($t)*;
            }
        }
    }
}

fn algox_inner<F>(dlx: &mut Dlx, partial_solution: &mut Vec<usize>, config: &mut AlgoXConfig, out: &mut F)
    -> Result<(), XError>
where
    F: FnMut(Vec<UInt>)
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
    stat!(config, calls, += 1);
    trace!("Enter algo X with exploring from partial_solution {:?}", partial_solution);
    if_trace!(dlx.format(false));

    // 1. is the matrix empty
    let empty = dlx.head_node().get(Next) == dlx.head();
    if empty {
        // We have a solution
        let sol = dlx.solution_to_rows(partial_solution);
        trace!("==> Valid solution: {:?} (index {:?})", sol, partial_solution);
        out(sol);
        return Ok(());
    }

    // 2. Pick the least populated column
    let mut col_index = 0;
    {
        let mut col_heads = dlx.walk_from(dlx.head());
        let mut min = !0;
        while let Some(index) = col_heads.next(dlx, Next) {
            stat!(config, col_seek, += 1);
            let count = dlx.get_value(index);
            if count < min {
                min = count;
                col_index = index;
                if min == 0 { break; } // found a minimum
            }
        }

        if min == 0 {
            trace!("Column {} unsatsified, backtracking", col_index);
            return Ok(());
        }
    }

    trace!("Selected col_index = {}", col_index);

    // 3. Explore the rows in the chosen column

    // cover column
    dlx.cover(col_index);
    stat!(config, cover, += 1);

    // now cover other columns sharing a one with this one
    let mut col_iter = dlx.walk_from(col_index);
    while let Some(col_i) = col_iter.next(dlx, Down) {

        // 4. Include row r in the partial solution
        partial_solution.push(col_i);
        trace!("partial_solution {:?}", partial_solution);

        // 5. Cover each column
        let mut row_iter = dlx.walk_from(col_i);
        while let Some(row_j) = row_iter.next(dlx, Next) {
            //trace!("walked to col_i={}, row_j={}", col_i, row_j);
            if let Ok(chead) = dlx.col_head_of(row_j) {
                dlx.cover(chead);
                stat!(config, cover, += 1);
            }
        }

        // 6. Repeat this algorithm recursively on the reduced matrix A.
        trace!("Recurse!");
        algox_inner(dlx, partial_solution, config, out)?;

        let _ = partial_solution.pop();
        trace!("partial_solution {:?}", partial_solution);

        let mut row_iter = dlx.walk_from(col_i);
        while let Some(row_j) = row_iter.next(dlx, Next.opp()) {
            if let Ok(chead) = dlx.col_head_of(row_j) {
                dlx.uncover(chead);
            }
        }
    }
    dlx.uncover(col_index);
    Ok(())
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
        println!("{:#?}", dlx);
        assert_eq!(dlx.column_count(0), 0);
        assert_eq!(dlx.column_count(1), 1);
        assert_eq!(dlx.column_count(2), 2);
        assert_eq!(dlx.column_count(3), 2);
        dlx.format(true);
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
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
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
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
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
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
        assert_eq!(solution, None, "solution mismatch");
    }

    #[test]
    fn dlx_size0_triv() {
        let mut dlx = Dlx::new(0);
        println!("{:#?}", dlx);
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
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
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
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
        dlx.format(true);
        let mut solution = None;
        algox(&mut dlx, |s| solution = Some(s));
        dlx.format(true);
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
        assert_eq!(dlx.rows, 1);
        dlx.assert_links();
        println!("{:#?}", dlx);
        dlx.format(true);
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
            rows: 0,
            row_table: vec![8, 11, 13, 17],
        };

        assert_eq!(dlx.solution_to_rows(&[8, 9, 10, 11, 12, 13, 14, 17, 23]),
                   vec![0, 0, 0, 1, 1, 2, 2, 3, 3]);
    }
}
