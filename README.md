# calculisto::automatic_differentiation
Forward-mode automatic differentiation with dual numbers, in C++20.

A dual number carries a value and its differentials with respect to `N`
independent variables. The arithmetic operators and the functions of `<cmath>`
compute them together, so that a single evaluation of a function gives its
value and its gradient, exact up to the rounding of the floating-point
operations: there is no step, unlike finite differences.

```cpp
#include <calculisto/automatic_differentiation/automatic_differentiation.hpp>
    using namespace calculisto::automatic_differentiation;

// f (x, y) = x sin (y) + exp (x y) and its gradient, at (1, 2): the
// independent variables number 0 and 1 among 2.
    const auto
x = independent_variable <2> (1., 0);
    const auto
y = independent_variable <2> (2., 1);
    const auto
f = x * sin (y) + exp (x * y);
// f.value is 8.29835, f.differentials[0], df/dx, is 15.6874, and
// f.differentials[1], df/dy, is 6.97291.

// The second derivative of g (t) = t^3, at 2: a dual of duals, whose value is
// itself an independent variable.
    const auto
t = independent_variable <1, dual_t <1>> (independent_variable <1> (2., 0), 0);
    const auto
g = t * t * t;
// g.value.value is 8, g.differentials[0].value, g', is 12, and
// g.differentials[0].differentials[0], g'', is 12.
```

## The dual numbers
- `dual_t <N, T>` is an aggregate, like `std::array`:
  `{ value, differentials }`, with `differentials` a `std::array <T, N>`. `T`
  is a floating-point type, `double` by default, or a dual, for the higher
  derivatives.
  `dual_t <N> {}` is zero, and `dual_t <N> { x }` the constant `x`.
- `independent_variable <N> (x, i)` is the independent variable number `i`
  among `N`, with the value `x`: its differential number `i` is one.
- `dual_cast <U> (d)` converts a dual to another type, as there is no
  converting constructor.

## The operations
- The arithmetic operators and compound assignment combine a dual with a dual
  of the same structure (the same number of differentials, at each level), or
  with a number of an arithmetic type. As with numbers, the result is of the
  type of the result on the values (a dual of `float` times a `double` is a
  dual of `double`), and the arguments are converted to that type first.
- The comparisons compare the values only, so that code branching on a dual
  takes the same branch as with its value.
- The functions of `<cmath>`, with the same rules for their arguments, except
  `tgamma` and `lgamma` (their derivatives need the digamma function, which the
  standard doesn't provide), the mathematical special functions (which libc++
  doesn't provide), and `nextafter`, `nexttoward` and `nan`.
- The arithmetic is `constexpr`, but not the functions of `<cmath>`, like those
  of `std` in C++20. Everything is `noexcept`, except `independent_variable`,
  which throws `std::out_of_range` for an index out of range.

## Where a derivative isn't defined
- A zero differential stays zero, even where a derivative is infinite or NaN,
  as with `sqrt` at 0: a constant doesn't change a derivative.
- The derivative of `abs` and `fabs` at 0 is that of the sign of the zero,
  `copysign (1, x)`.
- The functions constant between their jumps (`floor`, `round`...) have zero
  derivatives, also at the jumps. The integer ones (`lround`...) return an
  integer.
- `fmax` and `fmin` take the derivative of the argument selected: the first
  one where they are equal, the one that isn't NaN where one is. That of `fdim`
  is zero where `x <= y`.
- `pow (a, b)` takes the derivatives as zero where they are mathematically
  (with respect to `b` where `a^b` is zero and `a >= 0`), and NaN with respect
  to `b` where `a < 0`.
- The classification (`isnan`...) and comparison (`isgreater`...) functions
  apply to the values only.

## Accuracy
The formulas of the first derivatives avoid the cancellations of their usual
forms (e.g. `1 / cosh^2 (x)` rather than `1 - tanh^2 (x)` for `tanh`), and the
tests check them within 4 ulps of independent formulas, at points across the
domain of each function. The higher derivatives are those of the formulas of
the first derivatives, computed with their rounding: accurate relative to the
size of the terms they are made of, but not relative to their own value where
they are much smaller than these terms, e.g. the second derivative of
`hypot (x, y)` with respect to `x` where `|x|` is much larger than `|y|`.

With libc++ before LLVM 24, the `long double` constants of `std::numbers` are
only as accurate as a `double`
([llvm/llvm-project#188079](https://github.com/llvm/llvm-project/issues/188079)),
and so are the `long double` derivatives of `exp2`, `log2`, `log10`, `erf` and
`erfc`, which use them, where `long double` is more precise than `double`.

## Dependencies
- A C++20 compiler.
- [CMake](https://cmake.org) 3.25 or later, to install the library or run the
  tests.
- To run the tests, the [doctest](https://github.com/doctest/doctest) library,
  version 2.5 or later. If it isn't installed, CMake downloads it.

## Installation
This is a header-only library. It can be used in any of these ways:
- copy the `include` directory where your compiler can find it;
- from a CMake project, with `add_subdirectory` or `FetchContent`, then link to
  the `calculisto::automatic_differentiation` target;
- install it with CMake:
  ```
  cmake --preset dev
  cmake --install build/dev --prefix <directory>
  ```
  then, from a CMake project:
  ```cmake
  find_package (calculisto_automatic_differentiation 0.1 REQUIRED)
  target_link_libraries (<target> PRIVATE calculisto::automatic_differentiation)
  ```
  adding `<directory>` to `CMAKE_PREFIX_PATH` if it isn't a standard location.

## Tests
To configure, build and run the tests with the default compiler, execute
```
cmake --workflow --preset dev
```
in the root directory of the project. The tests are compiled with warnings as
errors, and run with the address sanitizer and, except with MSVC and clang-cl,
the undefined behavior sanitizer. They also check that the installed library
can be used with `find_package`.

Other presets select a compiler: `gcc`, `clang`, `clang-libcxx`, and on
Windows `msvc` and `clang-cl`. `cmake --list-presets=workflow` lists them.

## License
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

## Affiliation
This material is developed by the Numerical Modelling platform at the
Institut des Sciences de la Terre d'Orléans (https://www.isto-orleans.fr),
a joint laboratory of the University of Orléans (https://www.univ-orleans.fr),
the french National Center For Scientific Research (https://www.cnrs.fr) and
the french Geological Survey (https://www.brgm.fr).

[![logo ISTO](https://calcul-isto.cnrs-orleans.fr/logos/isto-156.png)](https://www.isto-orleans.fr) &emsp;
[![logo CNRS](https://calcul-isto.cnrs-orleans.fr/logos/cnrs-128.png)](https://www.cnrs.fr) &emsp;
[![logo UO](https://calcul-isto.cnrs-orleans.fr/logos/uo-128.png)](https://www.univ-orleans.fr) &emsp;
[![logo BRGM](https://calcul-isto.cnrs-orleans.fr/logos/brgm-256.png)](https://www.brgm.fr)
