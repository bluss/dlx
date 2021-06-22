// from bluss dlx solver

type Index = usize;

#[derive(Copy, Clone, Debug)]
struct Node<T> {
    /// Prev, Next, Up, Down
    link: [usize; 4],
    value: T,
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

    fn assign(&mut self, dir: Direction) -> &mut usize {
        &mut self.link[dir as usize]
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
