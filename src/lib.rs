// from bluss dlx solver

macro_rules! trace {
    ($($t:tt)*) => { eprintln!($($t)*) }
}

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

#[derive(Copy, Clone, Default)]
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
    fn new(value: T) -> Self
    {
        Node {
            value: value,
            link: [!0; 4],
        }
    }
    fn prev(&self) -> Index { self.get(Prev) }
    fn next(&self) -> Index { self.get(Next) }
    fn set_prev(&mut self, index: Index) -> &mut Self { self.set(Prev, index) }
    fn set_next(&mut self, index: Index) -> &mut Self { self.set(Next, index) }

    fn up(&self) -> Index { self.get(Up) }
    fn down(&self) -> Index { self.get(Down) }
    fn set_up(&mut self, index: Index) -> &mut Self { self.set(Up, index) }
    fn set_down(&mut self, index: Index) -> &mut Self { self.set(Down, index) }

    fn get(&self, dir: Direction) -> Index {
        self.link[dir as usize]
    }

    fn set(&mut self, dir: Direction, index: Index) -> &mut Self {
        self.link[dir as usize] = index;
        self
    }

    fn self_link(&mut self, dir: Direction, index: Index) -> &mut Self {
        self.link[dir as usize] = index;
        self.link[dir.opp() as usize] = index;
        self
    }

    fn assign(&mut self, dir: Direction) -> &mut usize {
        &mut self.link[dir as usize]
    }
}

pub type UInt = u32;

#[derive(Debug, Clone)]
pub struct Dlx {
    /// Node layout in DLX:
    /// [ Head ]    [ Columns ... ]
    /// [ Row Head] [ Row ... ]
    /// [ Row Head] [ Row ... ]
    /// ... etc.
    ///
    /// Doubly linked list in two dimensions: Prev, Next and Up, Down.
    pub(crate) nodes: Vec<Node<Point>>,
    columns: UInt,
    rows: UInt,
}

#[derive(Debug, Copy, Clone)]
enum Point {
    Head(UInt),
    /// Column with alive count in header
    Column(UInt),
    /// Row with alive count in header
    Row(UInt),
    Body(UInt),
}

impl Point {
    #[inline]
    pub(crate) fn value(&self) -> UInt {
        use Point::*;
        match *self {
            Head(x) | Column(x) | Row(x) | Body(x) => x
        }
    }

    #[inline]
    pub(crate) fn value_mut(&mut self) -> &mut UInt {
        use Point::*;
        match self {
            Head(x) | Column(x) | Row(x) | Body(x) => x
        }
    }
}

fn enumerate<T>(it: impl IntoIterator<Item=T>) -> impl Iterator<Item=(usize, T)> {
    it.into_iter().enumerate()
}

#[derive(Debug, Clone)]
pub enum DlxError {
    InvalidRow(&'static str),
    InvalidInput(&'static str),
}

impl Dlx {
    /// Create a new Dlx with the universe of points in 0..universe.
    pub fn new(universe: UInt) -> Self {
        // Insert head node and the column row.
        let mut nodes = vec![Node::new(Point::Head(0))];
        nodes.extend(repeat(Node::new(Point::Column(0))).take(universe as usize));

        // link the whole header row in both dimensions

        for (index, node) in enumerate(&mut nodes) {
            node.self_link(Down, index); // self-link in up-down axis
            let prev_index = if index == 0 { universe as Index } else { index - 1 };
            let next_index = if index == universe as Index { 0 } else { index + 1 };
            node.set(Prev, prev_index);
            node.set(Next, next_index);
        }

        Dlx {
            nodes,
            columns: universe,
            rows: 0,
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

    fn column_count(&self, col: UInt) -> UInt {
        assert!(col <= self.columns);
        self.nodes[self.column_head(col)].value.value()
    }

    pub(crate) fn walk_column_heads(&self) -> ColumnHeadWalker<'_> {
        ColumnHeadWalker {
            nodes: &self.nodes,
            index: 0,
        }
    }

    pub(crate) fn walk_from(&self, index: Index) -> Walker {
        Walker {
            index: index,
            start: index,
        }
    }

    pub(crate) fn col_head_of(&self, index: Index) -> Result<Index, DlxError> {
        let mut i = index;
        loop {
            i = self.nodes[i].get(Up);
            if matches!(self.nodes[i].value, Point::Column(_) | Point::Head(_)) {
                return Ok(i);
            }
            if i == index {
                panic!("Loop for index {}", i);
            }
        }
    }

    pub(crate) fn row_head_of(&self, index: Index) -> Result<Index, DlxError> {
        let mut i = index;
        loop {
            i = self.nodes[i].get(Prev);
            if matches!(self.nodes[i].value, Point::Row(_) | Point::Head(_)) {
                return Ok(i);
            }
            if i == index {
                panic!("Loop for index {}", i);
            }
        }
    }

    fn append_to_column(&mut self, col: UInt, new_index: Index) {
        debug_assert!(col <= self.columns, "invalid column {}", col);
        debug_assert!(new_index < self.nodes.len(), "invalid index {}", new_index);
        if col == 0 {
            debug_assert!(matches!(self.nodes[new_index].value, Point::Row(_)));
        } else {
            debug_assert!(matches!(self.nodes[new_index].value, Point::Body(_)));
        }
        let head_index = col as Index;
        let head = &mut self.nodes[head_index];
        let old_end = head.get(Up);
        head.set(Up, new_index);
        *head.value.value_mut() += 1;
        self.nodes[old_end].set(Down, new_index);
        self.nodes[new_index].set(Up, old_end);
        self.nodes[new_index].set(Down, head_index);
    }

    pub fn append_row(&mut self, row: impl IntoIterator<Item=UInt>) -> Result<(), DlxError> {
        let start_index = self.nodes.len();
        let row_number = self.rows + 1;
        let mut row_head = Node::new(Point::Row(row_number));
        self.nodes.push(row_head);
        self.append_to_column(0, start_index);
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
            let mut body_node = Node::new(Point::Body(row_number));
            let index = self.nodes.len();
            self.nodes.push(body_node);
            self.append_to_column(r, index);
        }

        if let None = max_seen {
            return Err(DlxError::InvalidRow("must not be empty"));
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

        Ok(())
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

    pub(crate) fn remove_row(&mut self, index: Index) {
        let delete_row_head = self.row_head_of(index).unwrap();
        assert_ne!(delete_row_head, 0, "Can't delete the column head row");
        trace!("Remove row {} {:?}", delete_row_head, self.nodes[delete_row_head].value);
        // now delete this row
        let mut cur = delete_row_head;
        loop {
            let cur = self.nodes[cur].get(Next);
            self.remove(cur, Next);
            self.remove(cur, Down);
            let this_column_head = self.col_head_of(cur).unwrap();
            *self.nodes[this_column_head].value.value_mut() -= 1;
            if cur == delete_row_head { break; }
        }
    }

    pub(crate) fn format(&self) {
        let n_blocks = self.nodes.len().saturating_sub((1 + self.columns + self.rows) as usize);
        eprintln!("Dlx columns={}, rows={}, nodes={} (blocks={})",
            self.columns, self.rows, self.nodes.len(), n_blocks);

        let mut headings = self.walk_from(0);
        eprint!("Head  ");
        while let Some(col_head) = headings.next(self, Next) {
            eprint!("{:4} ", self.nodes[col_head].value.value());
        }
        eprintln!();


        let mut rows = self.walk_from(0);
        while let Some(row_head) = rows.next(self, Down) {
            eprint!("{:?} ", self.nodes[row_head].value);
            let mut col = self.walk_from(row_head);
            while let Some(block) = col.next(self, Next) {
                let col_head = self.col_head_of(block).unwrap();
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

pub struct ColumnHeadWalker<'a> {
    nodes: &'a [Node<Point>],
    index: Index,
}

impl ColumnHeadWalker<'_> {
    pub fn next(&mut self) -> Option<(Index, UInt)> {
        let next = self.nodes[self.index].get(Next);
        if next == 0 {
            None
        } else {
            self.index = next;
            Some((next, self.nodes[next].value.value()))
        }
    }
}

#[derive(Clone, Debug)]
pub enum XError { Error }

pub fn algox(dlx: &mut Dlx) -> Result<(), XError> {
    trace!("algorithm X start");
    algox_inner(dlx)
}

fn algox_inner(dlx: &mut Dlx) -> Result<(), XError> {
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

    let mut solution = Vec::new();
    loop {
        // 1. is the matrix empty
        let empty = dlx.head_node().get(Next) == dlx.head();
        trace!("empty = {}", empty);
        if empty {
            break;
        }

        // 2. Pick the least populated column
        let mut col_heads = dlx.walk_column_heads();
        let mut min = !0;
        let mut col_index = 0;
        while let Some((index, count)) = col_heads.next() {
            if count < min {
                min = count;
                col_index = index;
            }
        }
        trace!("Selected col_index = {}", col_index);

        // 3. Explore the rows in the chosen column
        
        let mut col_c_down = dlx.walk_from(col_index);
        while let Some(col_i) = col_c_down.next(dlx, Down) {
            let row_number = dlx.nodes[col_i].value.value();
            solution.push(row_number);
            trace!("Exploring in block {}, row {}, solution {:?}", col_i, row_number, solution);

            // now, all rows that share a one (in columns) with this row need to be deleted.
            // and all columns where this row has a one need to be deleted
            let row_head_i = dlx.row_head_of(col_i).unwrap();
            trace!("Row head {}", row_head_i);

            let mut row_i = dlx.walk_from(row_head_i);
            while let Some(row_i_j) = row_i.next(dlx, Next) {
                // in the current column, delete rows that share a one
                trace!("Walk from {}", row_i_j);
                let mut rows = dlx.walk_from(row_i_j);
                while let Some(index) = rows.next(dlx, Down) {
                    let delete_row_head = dlx.row_head_of(index).unwrap();
                    if delete_row_head == 0 {
                        continue;
                    }
                    dlx.remove_row(delete_row_head);
                }
            }
            dlx.remove_row(row_head_i);
            break;
        }

        break;
    }

    Ok(())
    //Err(XError::Error)
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
        assert_eq!(dlx.column_count(0), 3);
        assert_eq!(dlx.column_count(1), 1);
        assert_eq!(dlx.column_count(2), 2);
        assert_eq!(dlx.column_count(3), 2);
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
        */
        let mut dlx = Dlx::new(7);
        dlx.append_row([1, 4, 7]).unwrap();
        dlx.append_row([1, 4]).unwrap();
        dlx.append_row([4, 5, 7]).unwrap();
        dlx.append_row([3, 5, 6]).unwrap();
        dlx.append_row([2, 3, 6, 7]).unwrap();
        dlx.append_row([2, 7]).unwrap();
        println!("{:#?}", dlx);
        dlx.format();
        algox(&mut dlx).unwrap();
        dlx.format();
    }
}
