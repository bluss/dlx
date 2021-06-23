//! Sudoku solver using the dlx library
/*

sudoku crate for Rust - Sudoku solver
Copyright (C) 2021 Ulrik Sverdrup "bluss"

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

use std::error::Error;
use std::io;
use std::io::Read;

use sudoku::Sudoku;

fn try_main() -> Result<(), Box<dyn Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let sudoku = Sudoku::parse(&input)?;
    println!("{}", sudoku);

    let mut problem = sudoku.to_problem().into_dlx();
    let mut solutions = Vec::new();
    problem.solve_all(|s| solutions.push(s));

    for soln in &solutions {
        println!("{}", soln);
    }

    match solutions.len() {
        0 => println!("No solution"),
        1 => println!("1 solution"),
        n => println!("{} solutions", n),
    }

    Ok(())
}

fn main() {
    if let Err(e) = try_main() {
        eprintln!("Error: {}", e);
        drop(e);
        std::process::exit(1);
    }
}
