// The functions of <cmath>: their values, derivatives and second derivatives,
// with each argument a dual or a number, and at their special points.
#include "test_tools.hpp"

TEST_CASE_TEMPLATE("Functions", Real, float, double, long double)
{
SUBCASE("Functions of one argument")
{
    // 1 / sqrt (1 - x^2), with (1 - x) (1 + x) rather than 1 - x^2, which
    // cancels near |x| = 1.
        const auto
    inverse_sqrt_one_minus_square = [] (Real x)
    {
        return 1 / std::sqrt ((1 - x) * (1 + x));
    };
        const auto
    cube = [] (Real x)
    {
        return x * x * x;
    };
        const auto
    sign = [] (Real x)
    {
        return x < 0 ? Real { -1 } : Real { 1 };
    };
        const auto
    zero = [] (Real)
    {
        return Real {};
    };
    // The constants of std::numbers, which the functions use: with libc++
    // before LLVM 24, those of long double are only as accurate as a double
    // (llvm/llvm-project#188079), and the derivatives with them.
        const auto
    ln2 = std::numbers::ln2_v <Real>;
        const auto
    ln10 = std::numbers::ln10_v <Real>;
        const auto
    inv_sqrtpi = std::numbers::inv_sqrtpi_v <Real>;
    SUBCASE("acos")
    {
        check_unary_function <Real> (FUNCTION (acos)
            , [&] (Real x) { return -inverse_sqrt_one_minus_square (x); }
            , [&] (Real x) { return -x * cube (inverse_sqrt_one_minus_square (x)); }
            , { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("asin")
    {
        check_unary_function <Real> (FUNCTION (asin)
            , inverse_sqrt_one_minus_square
            , [&] (Real x) { return x * cube (inverse_sqrt_one_minus_square (x)); }
            , { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("atan")
    {
        check_unary_function <Real> (FUNCTION (atan)
            , [] (Real x) { return 1 / (1 + x * x); }
            , [] (Real x) { return -2 * x / ((1 + x * x) * (1 + x * x)); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("cos")
    {
        check_unary_function <Real> (FUNCTION (cos)
            , [] (Real x) { return -std::sin (x); }
            , [] (Real x) { return -std::cos (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sin")
    {
        check_unary_function <Real> (FUNCTION (sin)
            , [] (Real x) { return std::cos (x); }
            , [] (Real x) { return -std::sin (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tan")
    {
        check_unary_function <Real> (FUNCTION (tan)
            , [] (Real x) { return 1 / (std::cos (x) * std::cos (x)); }
            , [] (Real x) { return 2 * std::tan (x) / (std::cos (x) * std::cos (x)); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("acosh")
    {
        check_unary_function <Real> (FUNCTION (acosh)
            , [] (Real x) { return 1 / std::sqrt ((x - 1) * (x + 1)); }
            , [&] (Real x) { return -x * cube (1 / std::sqrt ((x - 1) * (x + 1))); }
            , { 1.001, 1.5, 2, 10 });
    }
    SUBCASE("asinh")
    {
        check_unary_function <Real> (FUNCTION (asinh)
            , [] (Real x) { return 1 / std::sqrt (1 + x * x); }
            , [&] (Real x) { return -x * cube (1 / std::sqrt (1 + x * x)); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("atanh")
    {
        check_unary_function <Real> (FUNCTION (atanh)
            , [] (Real x) { return 1 / ((1 - x) * (1 + x)); }
            , [] (Real x) { return 2 * x / ((1 - x) * (1 + x) * (1 - x) * (1 + x)); }
            , { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("cosh")
    {
        check_unary_function <Real> (FUNCTION (cosh)
            , [] (Real x) { return std::sinh (x); }
            , [] (Real x) { return std::cosh (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sinh")
    {
        check_unary_function <Real> (FUNCTION (sinh)
            , [] (Real x) { return std::cosh (x); }
            , [] (Real x) { return std::sinh (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tanh")
    {
        check_unary_function <Real> (FUNCTION (tanh)
            , [] (Real x) { return 1 / (std::cosh (x) * std::cosh (x)); }
            , [] (Real x) { return -2 * std::tanh (x) / (std::cosh (x) * std::cosh (x)); }
            , { -5, -2, -0.5, 0.1, 0.5, 2, 5 });
    }
    SUBCASE("exp")
    {
        check_unary_function <Real> (FUNCTION (exp)
            , [] (Real x) { return std::exp (x); }
            , [] (Real x) { return std::exp (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("exp2")
    {
        check_unary_function <Real> (FUNCTION (exp2)
            , [&] (Real x) { return std::exp2 (x) * ln2; }
            , [&] (Real x) { return std::exp2 (x) * ln2 * ln2; }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("expm1")
    {
        check_unary_function <Real> (FUNCTION (expm1)
            , [] (Real x) { return std::exp (x); }
            , [] (Real x) { return std::exp (x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log")
    {
        check_unary_function <Real> (FUNCTION (log)
            , [] (Real x) { return 1 / x; }
            , [] (Real x) { return -1 / (x * x); }
            , { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log10")
    {
        check_unary_function <Real> (FUNCTION (log10)
            , [&] (Real x) { return 1 / (x * ln10); }
            , [&] (Real x) { return -1 / (x * x * ln10); }
            , { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log1p")
    {
        check_unary_function <Real> (FUNCTION (log1p)
            , [] (Real x) { return 1 / (1 + x); }
            , [] (Real x) { return -1 / ((1 + x) * (1 + x)); }
            , { -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log2")
    {
        check_unary_function <Real> (FUNCTION (log2)
            , [&] (Real x) { return 1 / (x * ln2); }
            , [&] (Real x) { return -1 / (x * x * ln2); }
            , { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("cbrt")
    {
        check_unary_function <Real> (FUNCTION (cbrt)
            , [] (Real x) { return 1 / (3 * std::cbrt (x) * std::cbrt (x)); }
            , [] (Real x) { return -2 / (9 * x * std::cbrt (x) * std::cbrt (x)); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("abs")
    {
        check_unary_function <Real> (FUNCTION (abs), sign, zero
            , { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("fabs")
    {
        check_unary_function <Real> (FUNCTION (fabs), sign, zero
            , { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("erf")
    {
        check_unary_function <Real> (FUNCTION (erf)
            , [&] (Real x) { return 2 * inv_sqrtpi * std::exp (-x * x); }
            , [&] (Real x) { return -4 * x * inv_sqrtpi * std::exp (-x * x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("erf and erfc at large |x|")
    {
        // exp (-x^2) amplifies the rounding of x^2 by x^2: with a float, the
        // reference is computed in double, where x^2 is exact.
        if constexpr (std::same_as <Real, float>)
        {
            for (auto const point: { -5.87, -4.85, 3.3, 5.5 })
            {
                    const auto
                x = opaque <float> (point);
                INFO ("x = ", x);
                    const auto
                expected = static_cast <float> (
                    2 * std::numbers::inv_sqrtpi * std::exp (-static_cast <double> (x) * x)
                );
                    const auto
                variable = independent_variable <1, float> (x, 0);
                CHECK(within_ulps <float> (erf (variable).differentials[0], expected, ulps));
                CHECK(within_ulps <float> (erfc (variable).differentials[0], -expected, ulps));
            }
        }
    }
    SUBCASE("erfc")
    {
        check_unary_function <Real> (FUNCTION (erfc)
            , [&] (Real x) { return -2 * inv_sqrtpi * std::exp (-x * x); }
            , [&] (Real x) { return 4 * x * inv_sqrtpi * std::exp (-x * x); }
            , { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sqrt")
    {
        check_unary_function <Real> (FUNCTION (sqrt)
            , [] (Real x) { return 1 / (2 * std::sqrt (x)); }
            , [] (Real x) { return -1 / (4 * x * std::sqrt (x)); }
            , { 0.1, 0.5, 2, 10 });
    }
}
SUBCASE("abs and fabs at 0")
{
    // The derivative is that of the sign of the zero.
    for (auto const& [x, derivative]: { std::pair { 0., 1. }, { -0., -1. } })
    {
            const auto
        variable = independent_variable <1, Real> (opaque <Real> (x), 0);
        CHECK(abs (variable).differentials[0] == static_cast <Real> (derivative));
        CHECK(fabs (variable).differentials[0] == static_cast <Real> (derivative));
    }
}
SUBCASE("Functions of two arguments")
{
    SUBCASE("atan2")
    {
        for (auto const& [y0, x0]: { std::pair { 3., 2. }, { 0.5, -1. }, { -2., 0.5 } })
        {
                const auto
            y = opaque <Real> (y0);
                const auto
            x = opaque <Real> (x0);
                const auto
            r2 = x * x + y * y;
                const auto
            r4 = r2 * r2;
            check_function (FUNCTION (atan2)
                , std::array { y, x }
                , std::array { x / r2, -y / r2 }
                , std::array {
                      std::array { -2 * x * y / r4, (y * y - x * x) / r4 }
                    , std::array { (y * y - x * x) / r4, 2 * x * y / r4 }
                  }
            );
        }
    }
    SUBCASE("hypot")
    {
        for (auto const& [x0, y0]: { std::pair { 2., 3. }, { -1., 0.5 }, { 0.5, -2. } })
        {
                const auto
            x = opaque <Real> (x0);
                const auto
            y = opaque <Real> (y0);
                const auto
            h = std::hypot (x, y);
                const auto
            h3 = h * h * h;
            check_function (FUNCTION (hypot)
                , std::array { x, y }
                , std::array { x / h, y / h }
                , std::array {
                      std::array { y * y / h3, -x * y / h3 }
                    , std::array { -x * y / h3, x * x / h3 }
                  }
            );
        }
    }
    SUBCASE("pow")
    {
        for (auto const& [x0, y0]: { std::pair { 2., 3. }, { 0.5, -1.5 }, { 3., 0.5 } })
        {
                const auto
            x = opaque <Real> (x0);
                const auto
            y = opaque <Real> (y0);
                const auto
            log_x = std::log (x);
                const auto
            cross = std::pow (x, y - 1) * (1 + y * log_x);
            check_function (FUNCTION (pow)
                , std::array { x, y }
                , std::array { y * std::pow (x, y - 1), std::pow (x, y) * log_x }
                , std::array {
                      std::array { y * (y - 1) * std::pow (x, y - 2), cross }
                    , std::array { cross, std::pow (x, y) * log_x * log_x }
                  }
            );
        }
    }
}
SUBCASE("Functions of two arguments of <cmath>, after pow")
{
        using
    second_partials_t = std::array <std::array <Real, 2>, 2>;
        const auto
    remquo_value = [] (auto const&... arguments)
    {
            using std::remquo;
            int
        quotient;
        return remquo (arguments..., &quotient);
    };
    // Away from the jumps of fmod and remainder, where the quotient changes.
    for (auto const& [x0, y0]: { std::pair { 5.3, 2. }, { -5.3, 2. }, { 5.3, -2. }, { 0.7, 3. }, { 7.9, 2. } })
    {
            const auto
        x = opaque <Real> (x0);
            const auto
        y = opaque <Real> (y0);
        INFO ("x = ", x, ", y = ", y);
            const auto
        truncated = std::trunc (x / y);
            const auto
        nearest = std::nearbyint (x / y);
        check_function (FUNCTION (fmod), std::array { x, y }, std::array { Real { 1 }, -truncated }, second_partials_t {});
        check_function (FUNCTION (remainder), std::array { x, y }, std::array { Real { 1 }, -nearest }, second_partials_t {});
        check_function (remquo_value, std::array { x, y }, std::array { Real { 1 }, -nearest }, second_partials_t {});
    }
    for (auto const& [x0, y0]: { std::pair { 2., 3. }, { -2., 3. }, { 2., -3. }, { -2., -3. } })
    {
            const auto
        x = opaque <Real> (x0);
            const auto
        y = opaque <Real> (y0);
        INFO ("x = ", x, ", y = ", y);
            const auto
        sign = (x < 0) == (y < 0) ? Real { 1 } : Real { -1 };
        check_function (FUNCTION (copysign), std::array { x, y }, std::array { sign, Real {} }, second_partials_t {});
            const auto
        x_larger = x > y;
        check_function (FUNCTION (fdim), std::array { x, y }
            , x_larger ? std::array <Real, 2> { 1, -1 } : std::array <Real, 2> { 0, 0 }
            , second_partials_t {}
        );
        check_function (FUNCTION (fmax), std::array { x, y }
            , x_larger ? std::array <Real, 2> { 1, 0 } : std::array <Real, 2> { 0, 1 }
            , second_partials_t {}
        );
        check_function (FUNCTION (fmin), std::array { x, y }
            , x_larger ? std::array <Real, 2> { 0, 1 } : std::array <Real, 2> { 1, 0 }
            , second_partials_t {}
        );
    }
}
SUBCASE("fmax, fmin and fdim where the arguments are equal, or NaN")
{
    // The derivative is that of the first argument where they are equal, and
    // that of the one that isn't NaN where one is; that of fdim is zero.
        const auto
    nan = std::numeric_limits <Real>::quiet_NaN ();
        const auto
    x = independent_variable <2, Real> (opaque <Real> (2), 0);
        const auto
    y = independent_variable <2, Real> (opaque <Real> (2), 1);
        const auto
    z = independent_variable <2, Real> (nan, 1);
    for (auto const& r: { fmax (x, y), fmin (x, y), fmax (x, z), fmin (x, z), fmax (z, x), fmin (z, x) })
    {
        CHECK(r.value == 2);
        CHECK(r.differentials[0] == 1);
        CHECK(r.differentials[1] == 0);
    }
    CHECK(fdim (x, y).differentials[0] == 0);
    CHECK(fdim (x, y).differentials[1] == 0);
    CHECK(fmax (x, opaque <Real> (2)).differentials[0] == 1);
    CHECK(fmax (opaque <Real> (2), x).differentials[0] == 0);
    CHECK(fmax (opaque <Real> (nan), x).differentials[0] == 1);
}
SUBCASE("Functions of three arguments")
{
    SUBCASE("hypot")
    {
            const auto
        x = std::array { opaque <Real> (2.), opaque <Real> (-1.), opaque <Real> (3.) };
            const auto
        h = std::hypot (x[0], x[1], x[2]);
            const auto
        h3 = h * h * h;
        check_function (FUNCTION (hypot), x
            , std::array { x[0] / h, x[1] / h, x[2] / h }
            , std::array {
                  std::array { (x[1] * x[1] + x[2] * x[2]) / h3, -x[0] * x[1] / h3, -x[0] * x[2] / h3 }
                , std::array { -x[1] * x[0] / h3, (x[0] * x[0] + x[2] * x[2]) / h3, -x[1] * x[2] / h3 }
                , std::array { -x[2] * x[0] / h3, -x[2] * x[1] / h3, (x[0] * x[0] + x[1] * x[1]) / h3 }
              }
        );
    }
}
SUBCASE("Functions of three arguments of <cmath>")
{
        const auto
    x = std::array { opaque <Real> (2.), opaque <Real> (-3.), opaque <Real> (0.5) };
    check_function (FUNCTION (fma), x
        , std::array { x[1], x[0], Real { 1 } }
        , std::array {
              std::array <Real, 3> { 0, 1, 0 }
            , std::array <Real, 3> { 1, 0, 0 }
            , std::array <Real, 3> { 0, 0, 0 }
          }
    );
        const auto
    t = std::array { opaque <Real> (1.), opaque <Real> (3.), opaque <Real> (0.25) };
    check_function (FUNCTION (lerp), t
        , std::array { 1 - t[2], t[2], t[1] - t[0] }
        , std::array {
              std::array <Real, 3> { 0, 0, -1 }
            , std::array <Real, 3> { 0, 0, 1 }
            , std::array <Real, 3> { -1, 1, 0 }
          }
    );
}
SUBCASE("Functions with an exponent, frexp and modf")
{
    for (auto const point: { 0.75, -3.5, 1e-3, 1e3 })
    {
            const auto
        x = opaque <Real> (point);
        INFO ("x = ", x);
            const auto
        variable = independent_variable <2, Real> (x, 0);
        for (auto const n: { 0, 3, -5, 40 })
        {
            INFO ("n = ", n);
                const auto
            scaled = std::ldexp (Real { 1 }, n);
            for (auto const& r: { ldexp (variable, n), scalbn (variable, n), scalbln (variable, long { n }) })
            {
                CHECK(r.value == std::ldexp (x, n));
                CHECK(r.differentials[0] == scaled);
                CHECK(r.differentials[1] == 0);
            }
        }
            int
        exponent;
            int
        expected_exponent;
            const auto
        mantissa = frexp (variable, &exponent);
        CHECK(mantissa.value == std::frexp (x, &expected_exponent));
        CHECK(exponent == expected_exponent);
        CHECK(mantissa.differentials[0] == std::ldexp (Real { 1 }, -exponent));
        CHECK(mantissa.differentials[1] == 0);
            auto
        integral_part = dual_t <2, Real> { 0, { 5, 5 } };
            Real
        expected_integral_part;
            const auto
        fractional_part = modf (variable, &integral_part);
        CHECK(fractional_part.value == std::modf (x, &expected_integral_part));
        CHECK(fractional_part.differentials[0] == 1);
        CHECK(fractional_part.differentials[1] == 0);
        CHECK(integral_part.value == expected_integral_part);
        CHECK(integral_part.differentials[0] == 0);
        CHECK(integral_part.differentials[1] == 0);
    }
    // The quotient of remquo, as that of std::remquo.
        int
    quotient;
        int
    expected_quotient;
    static_cast <void> (remquo (independent_variable <1, Real> (opaque <Real> (7.9), 0), Real { 2 }, &quotient));
    static_cast <void> (std::remquo (opaque <Real> (7.9), Real { 2 }, &expected_quotient));
    CHECK(quotient == expected_quotient);
}
SUBCASE("Functions constant between their jumps")
{
    // The value of std::, with zero differentials, or the integer of std::.
        using
    nested_real_t = dual_t <2, dual_t <2, Real>>;
    for (auto const point: { 2.5, -2.5, 1.3, -0.7, 3., 0.5, 1e3 })
    {
            const auto
        x = opaque <Real> (point);
        INFO ("x = ", x);
            const auto
        variable = independent_variable <2, Real> (x, 0);
            const auto
        nested = independent_variable <2, dual_t <2, Real>> (independent_variable <2, Real> (x, 0), 1);
            const auto
        check = [&] (auto f)
        {
                const auto
            r = f (variable);
            static_assert (std::same_as <decltype (r), dual_t <2, Real> const>);
            CHECK(r.value == f (x));
            CHECK(r.differentials[0] == 0);
            CHECK(r.differentials[1] == 0);
                const auto
            r_nested = f (nested);
            static_assert (std::same_as <decltype (r_nested), nested_real_t const>);
            CHECK(identical (r_nested, nested_real_t { { f (x) } }));
        };
        check (FUNCTION (logb));
        check (FUNCTION (ceil));
        check (FUNCTION (floor));
        check (FUNCTION (nearbyint));
        check (FUNCTION (rint));
        check (FUNCTION (round));
        check (FUNCTION (trunc));
            const auto
        check_integer = [&] (auto f)
        {
                const auto
            r = f (variable);
            static_assert (std::same_as <decltype (f (variable)), decltype (f (x))>);
            CHECK(r == f (x));
            CHECK(f (nested) == f (x));
        };
        check_integer (FUNCTION (ilogb));
        check_integer (FUNCTION (lrint));
        check_integer (FUNCTION (llrint));
        check_integer (FUNCTION (lround));
        check_integer (FUNCTION (llround));
    }
}
SUBCASE("Classification and comparison")
{
    // Of the values, whatever the differentials.
        const auto
    nan = std::numeric_limits <Real>::quiet_NaN ();
        const auto
    inf = std::numeric_limits <Real>::infinity ();
    for (auto const value: { Real { 1 }, Real { -0. }, nan, inf, -inf, std::numeric_limits <Real>::denorm_min () })
    {
        INFO ("value = ", value);
        for (auto const differential: { Real { 0 }, Real { 1 }, nan, inf })
        {
                const auto
            a = dual_t <2, Real> { opaque <Real> (value), { differential, 0 } };
            CHECK(fpclassify (a) == std::fpclassify (value));
            CHECK(isfinite (a) == std::isfinite (value));
            CHECK(isinf (a) == std::isinf (value));
            CHECK(isnan (a) == std::isnan (value));
            CHECK(isnormal (a) == std::isnormal (value));
            CHECK(signbit (a) == std::signbit (value));
            CHECK(isnan (dual_t <2, dual_t <2, Real>> { { a } }) == std::isnan (value));
            for (auto const other: { Real { 1 }, Real { 2 }, nan })
            {
                    const auto
                b = dual_t <2, Real> { opaque <Real> (other), { 0, differential } };
                CHECK(isgreater (a, b) == std::isgreater (value, other));
                CHECK(isgreaterequal (a, other) == std::isgreaterequal (value, other));
                CHECK(isless (value, b) == std::isless (value, other));
                CHECK(islessequal (a, b) == std::islessequal (value, other));
                CHECK(islessgreater (a, other) == std::islessgreater (value, other));
                CHECK(isunordered (value, b) == std::isunordered (value, other));
            }
        }
    }
}
SUBCASE("A zero differential stays zero")
{
    // Where a derivative is infinite, or NaN, a zero differential stays zero
    // rather than giving NaN (0 times infinity), as in pow: a constant
    // doesn't change a derivative. The other differentials follow the
    // mathematics.
        const auto
    inf = std::numeric_limits <Real>::infinity ();
        const auto
    check = [&] (auto f, double point, Real derivative)
    {
        INFO ("At ", point);
            const auto
        r = f (independent_variable <2, Real> (opaque <Real> (point), 0));
        CHECK(r.differentials[0] == derivative);
        CHECK(r.differentials[1] == 0);
    };
    check (FUNCTION (sqrt), 0, inf);
    check (FUNCTION (cbrt), 0, inf);
    check (FUNCTION (log), 0, inf);
    check (FUNCTION (log2), 0, inf);
    check (FUNCTION (log10), 0, inf);
    check (FUNCTION (log1p), -1, inf);
    check (FUNCTION (acos), 1, -inf);
    check (FUNCTION (acos), -1, -inf);
    check (FUNCTION (asin), 1, inf);
    check (FUNCTION (asin), -1, inf);
    check (FUNCTION (atanh), 1, inf);
    check (FUNCTION (atanh), -1, inf);
    check (FUNCTION (acosh), 1, inf);
    // Overflows.
    check (FUNCTION (exp), 1e5, inf);
    check (FUNCTION (exp2), 1e6, inf);
    check (FUNCTION (expm1), 1e5, inf);
    check (FUNCTION (cosh), 1e5, inf);
    check (FUNCTION (sinh), 1e5, inf);

    // At the origin, the derivatives of atan2 and hypot are NaN (0 / 0).
        const auto
    x = independent_variable <3, Real> (opaque <Real> (0), 0);
        const auto
    y = independent_variable <3, Real> (opaque <Real> (0), 1);
        const auto
    zero = opaque <Real> (0);
    for (auto const& r: {
          atan2 (x, y), hypot (x, y), hypot (x, y, zero), hypot (x, zero, y), hypot (zero, x, y)
    }){
        CHECK(std::isnan (r.differentials[0]));
        CHECK(std::isnan (r.differentials[1]));
        CHECK(r.differentials[2] == 0);
    }
    for (auto const& r: {
          atan2 (x, zero), atan2 (zero, x), hypot (x, zero), hypot (zero, x)
        , hypot (x, zero, zero), hypot (zero, x, zero), hypot (zero, zero, x)
    }){
        CHECK(std::isnan (r.differentials[0]));
        CHECK(r.differentials[1] == 0);
        CHECK(r.differentials[2] == 0);
    }
}
SUBCASE("Signed zeros, as with a number")
{
    // A function of duals gives the same zeros as with a number in place of
    // a dual whose differential is zero: the terms whose differential is
    // zero are skipped, not added, which would turn a -0 into +0.
        const auto
    zero = opaque <Real> (0);
    {
        // d hypot (a, b) / db = b / h = 0, times the differential -1.
            const auto
        a = dual_t <2, Real> { opaque <Real> (3), { 1, 0 } };
            const auto
        b = dual_t <2, Real> { zero, { 0, -1 } };
        CHECK(hypot (a, b).differentials[1] == 0);
        CHECK(std::signbit (hypot (a, b).differentials[1]) == std::signbit (hypot (a.value, b).differentials[1]));
        CHECK(std::signbit (hypot (a, b).differentials[1]));
    }
    {
        // d atan2 (y, x) / dx = -y / (x^2 + y^2) = -0.
            const auto
        y = dual_t <2, Real> { zero, { 1, 0 } };
            const auto
        x = dual_t <2, Real> { opaque <Real> (2), { 0, 1 } };
        CHECK(std::signbit (atan2 (y, x).differentials[1]) == std::signbit (atan2 (y.value, x).differentials[1]));
        CHECK(std::signbit (atan2 (y, x).differentials[1]));
    }
}
SUBCASE("pow at the boundaries of its domain")
{
    // A term of the derivative whose differential is zero contributes zero,
    // even where its factor is infinite or NaN, so that a constant exponent
    // or base gives the same result as a scalar one. The other terms follow
    // the mathematics: d/db pow (a, b) is NaN for a < 0, where pow (a, b) is
    // not real around b, even where pow (a, b) underflows to zero; zero for
    // a = 0 and b > 0, and for a = +inf and b < 0, where pow (a, b) is zero
    // around b; and -inf at 0^0 (log (0) times pow (0, 0) = 1).
        const auto
    nan = std::numeric_limits <Real>::quiet_NaN ();
        const auto
    inf = std::numeric_limits <Real>::infinity ();
    // Equal, with the same sign for zeros, or both NaN.
        const auto
    same = [] (Real result, Real expected)
    {
        if (std::isnan (expected))
        {
            return std::isnan (result);
        }
        return result == expected and std::signbit (result) == std::signbit (expected);
    };
        struct
    point_t
    {
        Real a, b, partial_a, partial_b;
    };
    for (auto const& [a0, b0, partial_a, partial_b]: {
          point_t { -2,  0,      0,   nan }
        , point_t { -2,  0.5,    nan, nan }
        , point_t { -2,  1,      1,   nan }
        , point_t { -2,  2,      -4,  nan }
        , point_t { -2,  3,      12,  nan }
        , point_t { -2,  -20000, 0,   nan }
        , point_t { -0., 2,      -0., 0 }
        , point_t { 0,   0,      0,   -inf }
        , point_t { 0,   0.5,    inf, 0 }
        , point_t { 0,   1,      1,   0 }
        , point_t { 0,   2,      0,   0 }
        , point_t { inf, -1,     -0., 0 }
    }){
            const auto
        a = opaque <Real> (a0);
            const auto
        b = opaque <Real> (b0);
        INFO ("a = ", a, ", b = ", b);
        // With an unused differential, which must stay zero.
        {
                const auto
            r = pow (independent_variable <3, Real> (a, 0), independent_variable <3, Real> (b, 1));
            CHECK(same (r.value, std::pow (a, b)));
            CHECK(same (r.differentials[0], partial_a));
            CHECK(same (r.differentials[1], partial_b));
            CHECK(same (r.differentials[2], 0));
        }
        for (auto const& r: {
              pow (independent_variable <2, Real> (a, 0), dual_t <2, Real> { b })
            , pow (independent_variable <2, Real> (a, 0), b)
        }){
            CHECK(same (r.differentials[0], partial_a));
            CHECK(same (r.differentials[1], 0));
        }
        for (auto const& r: {
              pow (dual_t <2, Real> { a }, independent_variable <2, Real> (b, 0))
            , pow (a, independent_variable <2, Real> (b, 0))
        }){
            CHECK(same (r.differentials[0], partial_b));
            CHECK(same (r.differentials[1], 0));
        }
    }
}
SUBCASE("pow with nested duals")
{
    // A differential whose value is zero, but not its own differentials, is
    // not zero: its term must not be skipped. With s the outer variable and t
    // the inner one, both at 0:
        using
    inner_t = dual_t <1, Real>;
        using
    nested_real_t = dual_t <1, inner_t>;
    {
        // a = 2 + s t, whose differential t ds is zero at t = 0, but not its
        // derivative with respect to t: d2/dt ds pow (a, 3) = 3 a^2 = 12.
            const auto
        a = nested_real_t { inner_t { opaque <Real> (2) }, { inner_t { 0, { 1 } } } };
        CHECK(pow (a, 3).differentials[0].differentials[0] == 12);
        CHECK(pow (a, nested_real_t { inner_t { 3 } }).differentials[0].differentials[0] == 12);
    }
    {
        // a = 2 + s and b = t: b is zero, but not its derivative, so that
        // b pow (a, b - 1) is not zero: d2/dt ds pow (a, b) = 1 / a = 0.5.
            const auto
        a = nested_real_t { inner_t { opaque <Real> (2) }, { inner_t { 1 } } };
            const auto
        b = nested_real_t { inner_t { 0, { 1 } } };
        CHECK(pow (a, b).differentials[0].differentials[0] == Real { 0.5 });
    }
}
} // TEST_CASE_TEMPLATE("Functions")
