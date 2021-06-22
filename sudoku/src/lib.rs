
use dlx::Dlx;
use dlx::UInt;
use dlx::algox;

use std::fmt;

#[derive(Clone, Debug)]
struct SudokuInput(Vec<Option<UInt>>);

impl SudokuInput {
    fn to_sudoku(&self) -> Sudoku {
        Sudoku {
            values: self.0.iter().map(|x| x.unwrap_or(0)).collect(),
        }
    }
}

fn sudoku_size_for_len(len: usize) -> Option<UInt> {
    match len {
        4 => Some(2),
        16 => Some(4),
        81 => Some(9),
        256 => Some(16),
        _ => None,
    }
}

impl SudokuInput {
    fn sudoku_size(&self) -> u16 {
        match self.0.len() {
            4 => 2,
            16 => 4,
            81 => 9,
            256 => 16,
            _ => unimplemented!("doesn't support this sudoku size"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Point {
    x: UInt,
    y: UInt,
    value: Option<UInt>,
}

impl Point {
    fn box_of(&self, sudoku_size: u16) -> UInt {
        box_of(sudoku_size, self.x, self.y)
    }
}

#[derive(Debug)]
struct ParseError(&'static str);

// is blank for sudoku
fn is_blank(s: &str) -> bool {
    match s {
        "." | "0" | "_" => true,
        _otherwise => false
    }
}

fn is_spacer(s: &str) -> bool {
    match s {
        "" | ";" | "|" | "+" | "-" => true,
        _otherwise => false
    }
}

fn parse(s: &str) -> Result<SudokuInput, ParseError> {
    let parts: Vec<Option<UInt>> = s.split(char::is_whitespace)
        .filter(|s| !is_spacer(*s) && (s.len() <= 1 || !s.split("").all(is_spacer)))
        .map(|s| if is_blank(s) { Ok(None) } else { Some(s.parse::<UInt>()).transpose() })
        .collect::<Result<Vec<Option<UInt>>, _>>()
        .map_err(|_| ParseError(""))?;

    if parts.len() != 16 && parts.len() != 81 {
        return Err(ParseError("Unsupported size"));
    }

    Ok(SudokuInput(parts))
}

// make constraints
//
// For sudoku of size N we have these constraints
//
// 1. There is exactly 1 number in cell RxCy
//    N * N
//    name RxCy
// 2. Row x contains number y exactly once
//    N * N
//    name Rx #y
// 3. Col x contains number y exactly once
//    N * N
//    name Cx #y
// 4. Box x contains number y exactly once
//    N * N
//    name Bx #y
//
// For N=9, this is 4 * 9 * 9 = 324 constraints
// 
// The N * N * N solution parts are:
// R1C1#1
// R1C1#2
// ..
// RxCy#z  (Row x Col y contains number z)
//
//
// A given number like R1C1#1 will remove these constraints:
//
// 1. Constraint R1C1 is removed
// 2. Constraint R1#1 is removed
// 3. Constraint C1#1 is removed
// 4. Constraint B1#1 is removed
//
// The following subsets are removed:
//
// R1C1#x for all x
// RxC1#1 for all x in the same row
// R1Cy#1 for all y in the same col
// RxCy#1 for all x, y in the same box

fn box_of(sudoku_size: u16, x: UInt, y: UInt) -> UInt {
    // R1C1 => 0
    // ..
    // R1C4 => 1
    // ...
    // R4C1 => 3
    let sz = match sudoku_size {
        2 => 0,
        4 => 2,
        9 => 3,
        16 => 4,
        _ => unimplemented!("doesn't support this sudoku size"),
    };
    if sz == 0 {
        return 0;
    }
    let xi = x / sz;
    let yi = y / sz;
    xi * sz + yi
}

fn all_subsets(sudoku_size: u16, start_index: UInt) -> Vec<Vec<UInt>> {
    let n = sudoku_size as UInt;
    let mut res = Vec::new();

    // Rx
    for x in 0..n {
        // Cy
        for y in 0..n {
            let w = box_of(sudoku_size, x, y);
            // RxCy #z
            for z in 0..n {
                // RxCy#z fulfills:
                // RxCy
                // Rx#z
                // Cy#z
                // Bw#z
                res.push(vec![
                    ExactlyOne(x, y).encoding(n),
                    RowContains(x, z).encoding(n),
                    ColContains(y, z).encoding(n),
                    BoxContains(w, z).encoding(n),
                ]);
            }
        }
    }

    res
}

#[derive(Clone, Debug)]
struct SudokuProblem {
    dlx: Dlx,
    subset_data: Vec<[UInt; 3]>,
    subset_names: Vec<String>,
}

impl SudokuProblem {
    fn to_sudoku(&self, solution: &[UInt]) -> Sudoku {
        let mut solution_data = solution.iter().map(|&i| self.subset_data[i as usize]).collect::<Vec<_>>();
        solution_data.sort_by_key(|d| (d[0], d[1]));
        Sudoku {
            values: solution_data.iter().map(|d| d[2] + 1).collect(),
        }
    }
}

fn create_problem(sudoku: &SudokuInput) -> SudokuProblem {
    let n = sudoku.sudoku_size() as usize;
    let nu = sudoku.sudoku_size() as UInt;
    dbg!(nu);
    let mut cnames = Vec::<String>::with_capacity(n * n * 4 / 2);
    let mut subsets = Vec::<Vec<UInt>>::new();
    let mut subset_data = Vec::new();
    let mut subset_names = Vec::<String>::with_capacity(n * n * n / 2);

    let sudoku: Vec<_> = sudoku.0.iter().enumerate().map(|(i, &v)| {
        Point {
            x: i as UInt / nu,
            y: i as UInt % nu,
            value: v,
        }
    }).collect();

    // bitsets
    let mut rows_taken = vec![0u64; n];
    let mut cols_taken = vec![0u64; n];
    let mut boxes_taken = vec![0u64; n];

    // mark used
    for x in 0..nu {
        for y in 0..nu {
            let cell = sudoku[(x * nu + y) as usize];
            if let Some(v) = cell.value {
                let b = cell.box_of(n as u16);
                rows_taken[x as usize] |= 1 << v;
                cols_taken[y as usize] |= 1 << v;
                boxes_taken[b as usize] |= 1 << v;
            }
        }
    }

    // create constraints and subsets
    //
    // 1. There is exactly 1 number in cell RxCy
    //    N * N
    //    name RxCy
    // 2. Row x contains number z exactly once
    //    N * N
    //    name Rx #z
    // 3. Col y contains number z exactly once
    //    N * N
    //    name Cy #z
    // 4. Box b contains number z exactly once
    //    N * N
    //    name Bb #z
    //
    let offset = 1;
    let cat_offset = nu * nu;
    for x in 0..nu {
        for y in 0..nu {
            let cell = sudoku[(x * nu + y) as usize];
            let b = cell.box_of(n as u16);
            for z in 0..nu {
                if cell.value.is_some() && Some(z + 1) != cell.value {
                    continue;
                }
                subset_names.push(format!("R{x}C{y}#{z}", x=x + 1, y=y + 1, z=z + 1));
                subset_data.push([x, y, z]);
                subsets.push(vec![
                        offset + 0 * cat_offset + x + y * nu,  // RxCy
                        offset + 1 * cat_offset + x + z * nu,  // Rx#z
                        offset + 2 * cat_offset + y + z * nu,  // Cy#z
                        offset + 3 * cat_offset + b + z * nu,  // Bb#z
                    ]);
            }
        }
    }

    let mut dlx = Dlx::new(nu * nu * 4);
    subsets = dbg!(subsets);
    for subset in subsets {
        dlx.append_row(subset).unwrap();
    }

    SudokuProblem {
        dlx,
        subset_data,
        subset_names,
    }
}

struct Sudoku {
    values: Vec<UInt>,
}

impl fmt::Display for Sudoku {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let nu = sudoku_size_for_len(self.values.len()).unwrap();
        let empty_str = ".";

        for x in 0..nu {
            for y in 0..nu {
                let value = self.values[(x * nu + y) as usize];
                if value == 0 {
                    write!(f, "{} ", empty_str)?;
                } else {
                    write!(f, "{} ", value)?;
                }
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

type N = UInt;

use Constraint::*;

#[derive(Debug, Copy, Clone)]
enum Constraint {
    ExactlyOne(N, N),
    RowContains(N, N),
    ColContains(N, N),
    BoxContains(N, N),
}

impl Constraint {
    fn encoding(&self, size: N) -> UInt {
        let offset = match *self {
            ExactlyOne(..) => 0,
            RowContains(..) => 1,
            ColContains(..) => 2,
            BoxContains(..) => 3,
        };
        let size = size as UInt;
        offset * size * size + self.index_of(size) as UInt
    }

    fn group_size(&self, n: N) -> usize {
        n as usize * n as usize
    }

    /// index (inside its own group) of this constraint
    fn index_of(&self, n: N) -> usize {
        use Constraint::*;

        match *self {
            ExactlyOne(x, y) |
            RowContains(x, y) |
            ColContains(x, y) |
            BoxContains(x, y) => (x as usize) * (n as usize) + (y as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn all() {
        for s in all_subsets(4, 1) {
            println!("{:?}", s);
        }
    }

    #[test]
    fn test_parse() {
        let v = parse("
        1 . ; 2 3
        2 . ; . .
        4 . ; 1 2
        3 . ; . 4
        ").unwrap();
        assert_eq!(v.0,
            vec![Some(1), None, Some(2), Some(3),
                 Some(2), None, None, None,
                 Some(4), None, Some(1), Some(2),
                 Some(3), None, None, Some(4)]);
    }

    #[test]
    fn test_problem1() {
        let v = parse("
            1 . ; 2 3
            2 . ; . .
            4 . ; 1 2
            3 . ; . 4
        ").unwrap();
        let mut p = create_problem(&v);
        println!("{:?}", p);
        let mut solution = None;
        algox(&mut p.dlx, |s| solution = Some(s));
        println!("{:?}", solution);
    }

    #[test]
    fn test_full_has_solution() {
        let v = parse("
            1 4 2 3
            2 3 1 4
            3 2 4 1
            4 1 3 2
        ").unwrap();
        println!("{:?}", v);
        let mut p = create_problem(&v);
        println!("{:?}", p);
        let mut solution = None;
        algox(&mut p.dlx, |s| solution = Some(s));
        println!("{:?}", solution);
        assert!(solution.is_some());
    }

    #[test]
    fn test_4x4_problem2() {
        let v = parse("
            1 4 2 .
            2 3 1 4
            3 2 . 1
            4 . 3 2
        ").unwrap();
        println!("{:?}", v);
        let mut p = create_problem(&v);
        println!("{:?}", p);
        let mut solution = None;
        p.dlx.format();
        algox(&mut p.dlx, |s| solution = Some(s));
        println!("{:?}", solution);
        assert!(solution.is_some());
        if let Some(s) = &mut solution {
            println!("{}", v.to_sudoku());
            println!("{}", p.to_sudoku(&*s));
            let sol = p.to_sudoku(&*s);
            assert_eq!(sol.values[..],
                       [1, 4, 2, 3,
                        2, 3, 1, 4,
                        3, 2, 4, 1,
                        4, 1, 3, 2]);
        }
    }

    #[test]
    fn test_4x4_multi1() {
        let v = parse("
            1 . . .
            2 . . .
            3 . . 2
            4 . . 1
        ").unwrap();
        println!("{:?}", v);
        let mut p = create_problem(&v);
        println!("{:?}", p);
        let mut solutions = Vec::new();
        p.dlx.format();
        algox(&mut p.dlx, |s| solutions.push(s));
        println!("{:?}", solutions);
        assert_eq!(solutions.len(), 2);
        println!("{}", v.to_sudoku());
        for s in &solutions {
            println!("{}", p.to_sudoku(&*s));
        }
    }

    #[test]
    fn test_9x9_easy() {
        let v = parse("
 +-------+-------+-------+  
 | . . . | . 3 . | . . . |  
 | 7 . 5 | 9 . . | . . 2 |  
 | 9 . . | . . 1 | . . . |  
 +-------+-------+-------+  
 | . 5 1 | . . . | . 8 3 |  
 | . . . | 3 . . | 5 . . |  
 | 4 8 . | . . . | 7 6 . |  
 +-------+-------+-------+  
 | . . . | . . . | . . 1 |  
 | . . 8 | . . 2 | 9 . . |  
 | . . . | . 9 . | 6 2 . |  
 +-------+-------+-------+ 
        ").unwrap();
        let mut p = create_problem(&v);
        let mut solution = None;
        p.dlx.format();
        algox(&mut p.dlx, |s| solution = Some(s));
        println!("{:?}", solution);
        assert!(solution.is_some());
        if let Some(s) = &solution {
            println!("{}", v.to_sudoku());
            println!("{}", p.to_sudoku(&*s));
        }
    }
}
