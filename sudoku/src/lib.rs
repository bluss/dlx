//! Sudoku solver using the dlx library

use dlx::Dlx;
use dlx::UInt;

use std::fmt;

#[derive(Clone, Debug)]
pub struct SudokuInput(Vec<Option<UInt>>);

impl SudokuInput {
    pub fn to_sudoku(&self) -> Sudoku {
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
pub struct ParseError(String);

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

pub fn parse(s: &str) -> Result<SudokuInput, ParseError> {
    let parts: Vec<Option<UInt>> = s.split(char::is_whitespace)
        .filter(|s| !is_spacer(*s) && (s.len() <= 1 || !s.split("").all(is_spacer)))
        .map(|s| if is_blank(s) { Ok(None) } else { Some(s.parse::<UInt>()).transpose() })
        .collect::<Result<Vec<Option<UInt>>, _>>()
        .map_err(|e| ParseError(e.to_string()))?;

    if parts.len() != 16 && parts.len() != 81 {
        return Err(ParseError(format!("Unsupported size: got {} elements: {:?}", parts.len(), parts)));
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

#[derive(Clone, Debug)]
pub struct SudokuProblem {
    sudoku_size: UInt,
    subsets: Vec<Vec<UInt>>,
    /// Subset data: RxCy#z: Row x, Col y filled with z.
    subset_data: Vec<[UInt; 3]>,
    columns: UInt,
}

#[derive(Clone, Debug)]
pub struct SudokuProblemDlx {
    dlx: Dlx,
    /// Subset data: RxCy#z: Row x, Col y filled with z.
    subset_data: Vec<[UInt; 3]>,
}

impl SudokuProblemDlx {
    /// Get Dlx matrix
    pub fn dlx(&self) -> &Dlx { &self.dlx }

    /// Get additional info for the Dlx rows: triplets of [Rx, Cy, #z]
    /// which correspond to row x, column y being filled with number z
    pub fn dlx_row_info(&self) -> &Dlx { &self.dlx }

    /// Given a Dlx solution, convert into a solved Sudoku
    pub fn to_sudoku(&self, solution: &[UInt]) -> Sudoku {
        Self::sudoku(&self.subset_data, solution)
    }

    fn sudoku(subset_data: &[[UInt; 3]], solution: &[u32]) -> Sudoku {
        let mut solution_data: Vec<_> = solution.iter()
            .map(move |&i| subset_data[i as usize])
            .collect();
        solution_data.sort_by_key(|d| (d[0], d[1]));
        Sudoku {
            values: solution_data.iter().map(|d| d[2] + 1).collect(),
        }
    }

    /// Get all solutions to the sudoku problem
    ///
    /// The problem is unmodified after the end of this method, and could be solved
    /// the same way again.
    pub fn solve_all(&mut self, mut out: impl FnMut(Sudoku)) {
        let subset_data = &self.subset_data;
        dlx::algox(&mut self.dlx, |s| out(Self::sudoku(subset_data, &s.get())));
    }

    /// Get the first found solution to the sudoku problem
    ///
    /// This is faster than `solve_all`, also in the case where there only is one solution:
    /// the extra work in solve all is the part needed to know that the solution is unique,
    /// in that case. This method can not say if the solution is unique or not.
    ///
    /// The problem is unmodified after the end of this method, and could be solved
    /// the same way again.
    pub fn solve_first(&mut self, mut out: impl FnMut(Sudoku)) {
        let subset_data = &self.subset_data;
        let mut config = dlx::AlgoXConfig::default();
        config.stop_at_first = true;
        dlx::algox_config(&mut self.dlx, &mut config, |s| out(Self::sudoku(subset_data, &s.get())));
    }
}

pub fn create_problem(sudoku: &SudokuInput) -> SudokuProblem {
    let n = sudoku.sudoku_size() as usize;
    let nu = sudoku.sudoku_size() as UInt;
    let mut subsets = Vec::<Vec<UInt>>::new();
    let mut subset_data = Vec::new();

    let sudoku: Vec<_> = sudoku.0.iter().enumerate().map(|(i, &v)| {
        Point {
            x: i as UInt / nu,
            y: i as UInt % nu,
            value: v,
        }
    }).collect();

    // create constraints and subsets
    //
    // RxCy: There is exactly 1 number in cell at x, y (N * N)
    // Rx#z: Row x contains number z exactly once (N * N)
    // Cy#z: Col y contains number z exactly once (N * N)
    // Bb#z: Box b contains number z exactly once (N * N)
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

    SudokuProblem {
        sudoku_size: nu,
        subsets,
        subset_data,
        columns: nu * nu * 4,
    }
}

impl SudokuProblem {
    pub fn optimize(self) -> Self {
        todo!()
    }

    pub fn into_dlx(self) -> SudokuProblemDlx {
        let mut dlx = Dlx::new(self.columns);
        for subset in self.subsets {
            dlx.append_row(subset).unwrap();
        }
        let subset_data = self.subset_data;

        SudokuProblemDlx {
            dlx,
            subset_data,
        }
    }
}

/// Displayable version of Sudoku. Can be solved or contain placeholders (as zero).
#[derive(Clone, Debug, PartialEq)]
pub struct Sudoku {
    values: Vec<UInt>,
}

impl Sudoku {
    /// Get the values (row major order)
    pub fn values(&self) -> &[UInt] {
        &self.values
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use dlx::algox;

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
        let mut p = create_problem(&v).into_dlx();
        println!("{:?}", p);
        let mut solution = None;
        algox(&mut p.dlx, |s| solution = Some(s.get()));
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
        let mut p = create_problem(&v).into_dlx();
        println!("{:?}", p);
        let mut solution = None;
        algox(&mut p.dlx, |s| solution = Some(s.get()));
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
        let mut p = create_problem(&v).into_dlx();
        println!("{:?}", p);
        let mut solution = None;
        p.dlx.format(true);
        algox(&mut p.dlx, |s| solution = Some(s.get()));
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

    macro_rules! test_solve {
        (all $input:expr, $($answer:expr),*) => {
            {
                test_solve($input, &[ $( parse($answer).unwrap().to_sudoku() ),* ], false);
            }
        };
        (one $input:expr, $answer:expr) => {
            {
                test_solve($input, &[ parse($answer).unwrap().to_sudoku() ], true);
            }
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
        let mut p = create_problem(&v).into_dlx();
        println!("{:?}", p);
        let mut solutions = Vec::new();
        p.dlx.format(true);
        algox(&mut p.dlx, |s| solutions.push(s.get()));
        println!("{:?}", solutions);
        assert_eq!(solutions.len(), 2);
        println!("{}", v.to_sudoku());
        for s in &solutions {
            println!("{}", p.to_sudoku(&*s));
        }
    }

    #[test]
    fn test_4x4_multi2() {
        test_solve! {
            all
            "
            1 2 . . 
            3 4 1 2
            . 1 . .
            . 3 . .
            "
            ,
            "
            1 2 3 4 
            3 4 1 2 
            2 1 4 3 
            4 3 2 1
            ",
            "
            1 2 3 4 
            3 4 1 2 
            4 1 2 3 
            2 3 4 1
            ",
            "
            1 2 4 3 
            3 4 1 2 
            2 1 3 4 
            4 3 2 1
            "
        }
    }

    fn test_solve(input: &str, solution_answers: &[Sudoku], only_first: bool) {
        let s_input = parse(input).unwrap();
        let mut problem = create_problem(&s_input).into_dlx();
        let mut solutions = Vec::new();
        if only_first {
            problem.solve_first(|s| solutions.push(s));
        } else {
            problem.solve_all(|s| solutions.push(s));
        }
        println!("{}", s_input.to_sudoku());
        for soln in &solutions {
            println!("{}", soln);
            assert!(solution_answers.iter().any(|elt| *elt == *soln),
                "Solution {} does not match answer", soln);
        }
        assert_eq!(solutions.len(), solution_answers.len(), "Wrong number of solutions");
    }

    #[test]
    fn test_9x9_1() {
        test_solve! { all
"
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
",
"
8 4 6 2 3 7 1 9 5
7 1 5 9 6 4 8 3 2
9 3 2 8 5 1 4 7 6
6 5 1 7 4 9 2 8 3
2 9 7 3 8 6 5 1 4
4 8 3 1 2 5 7 6 9
5 2 9 6 7 8 3 4 1
3 6 8 4 1 2 9 5 7
1 7 4 5 9 3 6 2 8
"
        }
    }

    #[test]
    fn test_9x9_2() {
        // http://sw-amt.ws/sudoku/doc/_build/html/worlds-hardest-sudoku.html
        // This one requires 10200 cover-uncover operations and 2600 recursions to solve
        // in DLX ("all solutions", checking it's unique).
        // To just get the first, it's 5460 cover-uncover with 1400 recursions.
        test_solve! {
one
"
 +-------+-------+-------+ 
 | 8 . .   . . .   . . . |
 | . . 3   6 . .   . . . |  
 | . 7 .   . 9 .   2 . . |  
 +-------+-------+-------+  
 | . 5 .   . . 7   . . . |  
 | . . .   . 4 5   7 . . |  
 | . . .   1 . .   . 3 . |  
 +-------+-------+-------+  
 | . . 1   . . .   . 6 8 |  
 | . . 8   5 . .   . 1 . |  
 | . 9 .   . . .   4 . . |  
 +-------+-------+-------+ 
",
"
8 1 2 7 5 3 6 4 9
9 4 3 6 8 2 1 7 5
6 7 5 4 9 1 2 8 3
1 5 4 2 3 7 8 9 6
3 6 9 8 4 5 7 2 1
2 8 7 1 6 9 5 3 4
5 2 1 9 7 4 3 6 8
4 3 8 5 2 6 9 1 7
7 9 6 3 1 8 4 5 2
"
        }
    }

 /*
 +-------+-------+-------+ 
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 +-------+-------+-------+  
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 +-------+-------+-------+  
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 | . . .   . . .   . . . |
 +-------+-------+-------+ 
 */

    #[test]
    fn test_9x9_knuth_29b() {
        // Knuth fasc 5c preprint
        test_solve! {
            one
            "
             +-------+-------+-------+ 
             | . . .   . . .   3 . . |
             | 1 . .   4 . .   . . . |
             | . . .   . . .   1 . 5 |
             +-------+-------+-------+  
             | 9 . .   . . .   . . . |
             | . . .   . . 2   6 . . |
             | . . .   . 5 3   . . . |
             +-------+-------+-------+  
             | . 5 .   8 . .   . . . |
             | . . .   9 . .   . 7 . |
             | . 8 3   . . .   . 4 . |
             +-------+-------+-------+ 
            ",
            "
            5 9 7 2 1 8 3 6 4 
            1 3 2 4 6 5 8 9 7 
            8 6 4 3 7 9 1 2 5 
            9 1 5 6 8 4 7 3 2 
            3 4 8 7 9 2 6 5 1 
            2 7 6 1 5 3 4 8 9 
            6 5 9 8 4 7 2 1 3 
            4 2 1 9 3 6 5 7 8 
            7 8 3 5 2 1 9 4 6 
            "
        }
    }

    #[test]
    fn test_9x9_knuth_29c() {
        // Knuth fasc 5c preprint
        test_solve! {
            all
            "
             +-------+-------+-------+ 
             | . 3 .   . 1 .   . . . |
             | . . .   4 . .   1 . . |
             | . 5 .   . . .   . 9 . |
             +-------+-------+-------+  
             | 2 . .   . . .   6 . 4 |
             | . . .   . 3 5   . . . |
             | 1 . .   . . .   . . . |
             +-------+-------+-------+  
             | 4 . .   6 . .   . . . |
             | . . .   . . .   . 5 . |
             | . 9 .   . . .   . . . |
             +-------+-------+-------+ 
            ",
            "
            9 3 4 5 1 7 2 6 8
            8 6 2 4 9 3 1 7 5
            7 5 1 8 6 2 4 9 3
            2 7 5 9 8 1 6 3 4
            6 4 9 2 3 5 8 1 7
            1 8 3 7 4 6 5 2 9
            4 1 7 6 5 9 3 8 2
            3 2 8 1 7 4 9 5 6
            5 9 6 3 2 8 7 4 1
            ",
            "
            9 3 4 5 1 8 2 6 7
            7 6 2 4 9 3 1 8 5
            8 5 1 7 6 2 4 9 3
            2 8 5 9 7 1 6 3 4
            6 4 9 2 3 5 7 1 8
            1 7 3 8 4 6 5 2 9
            4 1 8 6 5 9 3 7 2
            3 2 7 1 8 4 9 5 6
            5 9 6 3 2 7 8 4 1
            "
        }
    }

}
