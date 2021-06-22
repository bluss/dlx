// from bluss dlx solver

use std::fmt;
use std::convert::TryFrom;

type Index = usize;

#[derive(Copy, Clone, Default)]
struct Node<T> {
    /// Prev, Next, Up, Down
    link: [usize; 4],
    value: T,
}

impl<T> fmt::Debug for Node<T> where T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct(stringify!(Node))
            .field(stringify!(link), &format_args!("[{}, {}; {}, {}]", self.link[0], self.link[1],
                                                   self.link[2], self.link[3]))
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
    Head,
    /// Column with alive count in header
    Column(UInt),
    /// Row with alive count in header
    Row(UInt),
    Body,
}

fn enumerate<T>(it: impl IntoIterator<Item=T>) -> impl Iterator<Item=(usize, T)> {
    it.into_iter().enumerate()
}

impl Dlx {
    /// Create a new Dlx with the universe of points in 0..universe.
    pub fn new(universe: UInt) -> Self {
        let mut nodes = vec![Node::new(Point::Head)];
        // self-link the head node
        //nodes[0].self_link(Next, 0);
        //nodes[0].self_link(Down, 0);

        nodes.resize(universe as usize + 1, Node::new(Point::Column(0)));

        // link the whole header row in both dimensions

        for (index, node) in enumerate(&mut nodes) {
            node.self_link(Down, index); // self-link in up-down axis
            let prev_index = if index == 0 { universe as usize } else { index - 1 };
            let next_index = if index == universe as usize { 0 } else { index + 1 };
            node.set(Prev, prev_index);
            node.set(Next, next_index);
        }

        Dlx {
            nodes,
            columns: 0,
            rows: 0,
        }
    }

    fn head() -> Index { 0 }
    fn column_head(&self, col: UInt) -> Index { 1 + col as usize }

    pub fn append_row(&mut self, row: impl Iterator<Item=UInt>) -> &mut Self
    {
        self
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
        let mut dlx = Dlx::new(4);
        println!("{:#?}", dlx);
    }
}
