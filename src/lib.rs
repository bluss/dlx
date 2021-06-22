// from bluss dlx solver

use std::fmt;
use std::convert::TryFrom;
use std::iter::repeat;

type Index = usize;

#[derive(Copy, Clone, Default)]
struct Node<T> {
    /// Prev, Next, Up, Down
    link: [usize; 4],
    value: T,
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
struct Dlx {
    /// Node layout in DLX:
    /// [ Head ]    [ Columns ... ]
    /// [ Row Head] [ Row ... ]
    /// [ Row Head] [ Row ... ]
    /// ... etc.
    ///
    /// Doubly linked list in two dimensions: Prev, Next and Up, Down.
    nodes: Vec<Node<Point>>,
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

    fn head() -> Index { 0 }
    fn column_head(&self, col: UInt) -> Index { col as Index }

    fn column_count(&self, col: UInt) -> UInt {
        assert!(col <= self.columns);
        self.nodes[self.column_head(col)].value.value()
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

    pub fn append_row(&mut self, row: impl IntoIterator<Item=UInt>) -> Result<(), DlxError>
    {
        let start_index = self.nodes.len();
        let mut row_head = Node::new(Point::Row(0));
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
            let mut body_node = Node::new(Point::Body(0));
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
}

// Direction of list link
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
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
}
