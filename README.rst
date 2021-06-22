Dancing Links “dlx”
===================

Dancing Links solver for “algorithm X” by Knuth

This solver solves the exact cover problem using “algorithm X”, implemented using Dancing Links
(“Dlx”). The most common example problem solved by an exact cover solver is sudokus,
and that's included here too.

The Dlx data structure structure corresponds to a sparse binary matrix and it
uses two-dimensional doubly linked and circular lists.

See Knuth for papers about this structure and about “algorithm X”.

Node layout in DLX::

    
               ..    ..    ..
               ||    ||    ||
    :> Head <> C1 <> C2 <> C3 <> ... <:    (Head and column heads)
               ||    ||    ||
            :> R1  <    >  R2  <       ..  (Row items)
               ||    ||          ||
                  :> R3  <    >  R4  < ..
                     ||          ||
                     ..          ..
    
      etc.
      where || are a Up/Down links and <> Prev/Next links.

Head is only linked to the column row.
Note that R1 links directly to R2 and so on, the matrix is sparse.

The lists are doubly linked and circular, in two dimensions: Prev, Next
and Up, Down.


Building
--------

Using stable Rust, at least version Rust 1.53

Running
-------

You can try this, to build and run the sudoku solver that uses dancing links::

    cargo build
    cargo run < examples/hard_sudoku.txt

License
=======

dlx and sudoku crates for Rust
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

