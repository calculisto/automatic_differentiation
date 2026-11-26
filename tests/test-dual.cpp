#include <doctest/doctest.h>
    using doctest::Approx;
#include <calculisto/automatic_differentiation/automatic_differentiation.hpp>
    using namespace calculisto::automatic_differentiation;
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <utility>
#include <type_traits>
    using calculisto::array::array_t;

// Comparisons exist only with types whose values compare with the value of
// the dual, give a bool (== and !=) or an ordering (<=>), and asking whether
// they exist doesn't cause a compilation error.
    template <class A, class B>
    concept
comparable = requires (A const& a, B const& b)
{
    { a == b } -> std::same_as <bool>;
    { b == a } -> std::same_as <bool>;
    { a != b } -> std::same_as <bool>;
    { b != a } -> std::same_as <bool>;
    { a <=> b } -> std::same_as <std::partial_ordering>;
    { b <=> a } -> std::same_as <std::partial_ordering>;
    { a < b } -> std::same_as <bool>;
    { b < a } -> std::same_as <bool>;
};
    template <class A, class B>
    concept
has_any_comparison = requires (A const& a, B const& b) { a == b; }
    or requires (A const& a, B const& b) { b == a; }
    or requires (A const& a, B const& b) { a != b; }
    or requires (A const& a, B const& b) { a <=> b; }
    or requires (A const& a, B const& b) { b <=> a; }
    or requires (A const& a, B const& b) { a < b; };
    struct
not_comparable
{};
static_assert (comparable <dual_t <1>, dual_t <1>>);
static_assert (comparable <dual_t <1>, dual_t <1, float>>);
static_assert (comparable <dual_t <1>, double>);
static_assert (comparable <dual_t <1>, int>);
static_assert (not has_any_comparison <dual_t <1>, not_comparable>);
static_assert (not has_any_comparison <dual_t <1>, dual_t <2>>);

// Tools to check the functions, with Real the floating-point type of the
// values: the value of the result must be that of the std:: function, and its
// differentials must be the partial derivatives, computed independently in the
// tests, within a few ulps.

// a and b are equal within n ulps (relative to the larger of the two).
    template <class Real>
    bool
within_ulps (Real a, Real b, Real n)
{
    if (a == b)
    {
        return true;
    }
    return std::abs (a - b)
        <= n * std::numeric_limits <Real>::epsilon ()
            * std::max (std::abs (a), std::abs (b));
}

// The tolerance of the derivatives, in ulps.
    constexpr int
ulps = 4;

// The value x converted to Real, which the compiler cannot know: the results
// are computed at run time, as in the std:: functions they are compared with.
    template <class Real, class Number>
    Real
opaque (Number x)
{
        volatile Real
    v = static_cast <Real> (x);
    return v;
}

    namespace
function_checks
{
    // The seeds of the shared differential, one per argument.
        constexpr std::array
    shared_seeds { -3., 0.5, 5. };

    // The argument number I of the function, of K: if bit I of Mask is set, a
    // dual with K + 2 differentials: 2 in slot I, so that slot I of the result
    // is twice the partial derivative I, and a seed in the shared slot K, so
    // that slot K of the result sums the terms of several arguments (the
    // chain rule). Slot K + 1 is zero in every argument, and must stay so,
    // even where a partial derivative is infinite or NaN. The value x
    // otherwise.
        template <unsigned Mask, std::size_t I, std::size_t K, class Real>
        auto
    argument (Real x)
    {
        static_assert (K <= shared_seeds.size ());
        if constexpr ((Mask >> I) & 1u)
        {
                auto
            a = dual_t <K + 2, Real> { x };
            a.differentials[I] = 2;
            a.differentials[K] = static_cast <Real> (shared_seeds[I]);
            return a;
        }
        else
        {
            return x;
        }
    }

    // Checks f at x, with the arguments selected by Mask being duals and the
    // others values: partials are the partial derivatives of f at x.
        template <unsigned Mask, class F, class Real, std::size_t K, std::size_t... I>
        void
    check_function_mask (
          F f
        , std::array <Real, K> const& x
        , std::array <Real, K> const& partials
        , std::index_sequence <I...>
    ){
        INFO ("Duals: mask ", Mask);
            const auto
        r = f (argument <Mask, I, K> (x[I])...);
        static_assert (std::same_as <decltype (r), dual_t <K + 2, Real> const>);
        CHECK(r.value == f (x[I]...));
            auto
        shared = Real {};
            auto
        magnitude = Real {};
        for (std::size_t i = 0; i < K; ++i)
        {
            INFO ("Partial derivative ", i, ": ", r.differentials[i], " expected ", (((Mask >> i) & 1u) ? 2 * partials[i] : Real {}));
            if ((Mask >> i) & 1u)
            {
                CHECK(within_ulps <Real> (r.differentials[i], 2 * partials[i], ulps));
                    const auto
                term = static_cast <Real> (shared_seeds[i]) * partials[i];
                shared += term;
                magnitude += std::abs (term);
            }
            else
            {
                CHECK(r.differentials[i] == 0);
            }
        }
        // The terms may cancel: the error is relative to their magnitude.
        INFO ("Shared differential: ", r.differentials[K], " expected ", shared);
        CHECK(std::abs (r.differentials[K] - shared)
            <= ulps * std::numeric_limits <Real>::epsilon () * magnitude);
        CHECK(r.differentials[K + 1] == 0);
    }

        template <class F, class Real, std::size_t K, std::size_t... M>
        void
    check_function_all_masks (
          F f
        , std::array <Real, K> const& x
        , std::array <Real, K> const& partials
        , std::index_sequence <M...>
    ){
        // Mask 0, where no argument is a dual, is not checked.
        (check_function_mask <M + 1> (
              f
            , x
            , partials
            , std::make_index_sequence <K> {}
        ), ...);
    }
} // namespace function_checks

// Checks f, a function of K arguments, at x, with partials its partial
// derivatives there: with each combination of duals and values as arguments.
    template <class F, class Real, std::size_t K>
    void
check_function (
      F f
    , std::array <Real, K> const& x
    , std::array <Real, K> const& partials
){
    INFO ("At x[0] = ", x[0]);
    function_checks::check_function_all_masks (
          f
        , x
        , partials
        , std::make_index_sequence <(1u << K) - 1> {}
    );
}

// Checks f, a function of one argument, at each of the points converted to
// Real, with derivative its derivative.
    template <class Real, class F, class D>
    void
check_unary_function (F f, D derivative, std::initializer_list <double> points)
{
    for (auto const point: points)
    {
            const auto
        x = opaque <Real> (point);
        check_function (f, std::array { x }, std::array { derivative (x) });
    }
}

// The functions to check, callable with values and with duals.
#define FUNCTION(name) \
    [] (auto const&... arguments) { using std::name; return name (arguments...); }

TEST_CASE("automatic_differentiation.hpp")
{
SUBCASE("Construction")
{
    // An aggregate, like std::array.
    static_assert (std::is_aggregate_v <dual_t <2>>);

    // Value-initialization gives zero, like double {}.
        constexpr auto
    zero = dual_t <2> {};
    static_assert (zero.value == 0);
    static_assert (zero.differentials[0] == 0);
    static_assert (zero.differentials[1] == 0);

    // A constant: the differentials are zero.
        constexpr auto
    a = dual_t <1> { 2 };
    static_assert (a.value == 2);
    static_assert (a.differentials[0] == 0);

    // The value and the differentials.
        constexpr auto
    c = dual_t <2> { 32, { 0, 0.5 } };
    static_assert (c.value == 32);
    static_assert (c.differentials[0] == 0);
    static_assert (c.differentials[1] == 0.5);

    // Class template argument deduction.
        constexpr auto
    d = dual_t { 1.f, array_t <float, 2> { 2.f, 3.f } };
    static_assert (std::same_as <decltype (d), dual_t <2, float> const>);

    // An independent variable: its differential with respect to itself is
    // one.
        const auto
    b = independent_variable <2> (25., 0);
    static_assert (std::same_as <decltype (b), dual_t <2> const>);
    CHECK(b.value == 25);
    CHECK(b.differentials[0] == 1);
    CHECK(b.differentials[1] == 0);
}
SUBCASE("A dual with no differentials behaves normal number")
{
        const auto
    x = independent_variable <1> (3, 0);
        const auto
    p1 = 2 * x;
    CHECK(p1.value == 6);
    CHECK(p1.differentials[0] == 2);

        const auto
    a = dual_t <1> { 2 };
        const auto
    p2 = a * x;
    CHECK(p2.value == 6);
    CHECK(p2.differentials[0] == 2);
}
SUBCASE("Arithmetic, one value")
{
        const auto
    x = independent_variable <1> (2, 0);
    CHECK((+x).value == 2);
    CHECK((-x).value == -2);
    CHECK(x == 2);
    CHECK(2 == x);
    CHECK(x != 3);
    CHECK(3 != x);
    CHECK(x <  3);
    CHECK(1 <  x);
    CHECK(x >  1);
    CHECK(3 >  x);
    CHECK(x <= 3);
    CHECK(1 <= x);
    CHECK(x >= 1);
    CHECK(3 >= x);
    CHECK(x <= 2);
    CHECK(2 <= x);
    CHECK(x >= 2);
    CHECK(2 >= x);
    {
            const auto
        f = x + 1;
        CHECK(f.value == 3);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = 1 + x;
        CHECK(f.value == 3);
        CHECK(f.differentials[0] == 1);
    }
    {
            auto
        f = independent_variable <1> (1, 0);
        f += 1;
        CHECK(f.value == 2);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = x - 1;
        CHECK(f.value == 1);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = 1 - x;
        CHECK(f.value == -1);
        CHECK(f.differentials[0] == -1);
    }
    {
            auto
        f = independent_variable <1> (1, 0);
        f -= 1;
        CHECK(f.value == 0);
        CHECK(f.differentials[0] == 1);
    }
    {
            const auto
        f = x * 2;
        CHECK(f.value == 4);
        CHECK(f.differentials[0] == 2);
    }
    {
            const auto
        f = 2 * x;
        CHECK(f.value == 4);
        CHECK(f.differentials[0] == 2);
    }
    {
            const auto
        f = x / 2;
        CHECK(f.value == 1);
        CHECK(f.differentials[0] == .5);
    }
    {
            const auto
        f = 1 / x;
        CHECK(f.value == .5);
        CHECK(f.differentials[0] == -0.25);
    }
}
SUBCASE("Arithmetic, two values")
{
        const auto
    x = independent_variable <2> (2, 0);
        const auto
    y = independent_variable <2> (3, 1);
    CHECK(x != y);
    CHECK(x <  y);
    CHECK(y >  x);
    CHECK(x <= y);
    CHECK(y >= x);
    {
            const auto
        f = x + y;
        CHECK(f.value == 5);
        CHECK(f.differentials[0] == 1);
        CHECK(f.differentials[1] == 1);
    }
    {
            const auto
        f = x - y;
        CHECK(f.value == -1);
        CHECK(f.differentials[0] == 1);
        CHECK(f.differentials[1] == -1);
    }
    {
            const auto
        f = x * y;
        CHECK(f.value == 6);
        CHECK(f.differentials[0] == 3);
        CHECK(f.differentials[1] == 2);
    }
    {
            const auto
        f = x / y;
        CHECK(f.value == 2. / 3);
        CHECK(f.differentials[0] == 1. / 3);
        CHECK(f.differentials[1] == -2. / 9);
    }
}
SUBCASE("Polynomials")
{
        const auto
    x = independent_variable <3> (2, 0);
        const auto
    y = independent_variable <3> (3, 1);
        const auto
    z = independent_variable <3> (4, 2);
        const auto
    f = 3 * pow (x, 3) + 4 * pow (y, 2) + 5 * z + 3;
    CHECK(f.value == 83);
    CHECK(f.differentials[0] == 36);
    CHECK(f.differentials[1] == 24);
    CHECK(f.differentials[2] == 5);
}
SUBCASE("Unused dimensions")
{
        const auto
    x = independent_variable <3> (2, 0);
        const auto
    y = independent_variable <3> (3, 1);
        const auto
    z = 4.;
        const auto
    f = 3 * pow (x, 3) + 4 * pow (y, 2) + 5 * z + 3;
    CHECK(f.value == 83);
    CHECK(f.differentials[0] == 36);
    CHECK(f.differentials[1] == 24);
    CHECK(f.differentials[2] == 0);
}
    
} // TEST_CASE("automatic_differentiation.hpp")

TEST_CASE_TEMPLATE("Functions", Real, float, double, long double)
{
SUBCASE("Functions of one argument")
{
        const auto
    inverse_sqrt_one_minus_square = [] (Real x)
    {
        return 1 / std::sqrt ((1 - x) * (1 + x));
    };
        const auto
    sign = [] (Real x)
    {
        return x < 0 ? Real { -1 } : Real { 1 };
    };
    SUBCASE("acos")
    {
        check_unary_function <Real> (FUNCTION (acos), [&] (Real x)
        {
            return -inverse_sqrt_one_minus_square (x);
        }, { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("asin")
    {
        check_unary_function <Real> (FUNCTION (asin), inverse_sqrt_one_minus_square
            , { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("atan")
    {
        check_unary_function <Real> (FUNCTION (atan), [] (Real x)
        {
            return 1 / (1 + x * x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("cos")
    {
        check_unary_function <Real> (FUNCTION (cos), [] (Real x)
        {
            return -std::sin (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sin")
    {
        check_unary_function <Real> (FUNCTION (sin), [] (Real x)
        {
            return std::cos (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tan")
    {
        check_unary_function <Real> (FUNCTION (tan), [] (Real x)
        {
            return 1 / (std::cos (x) * std::cos (x));
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("acosh")
    {
        check_unary_function <Real> (FUNCTION (acosh), [] (Real x)
        {
            return 1 / std::sqrt ((x - 1) * (x + 1));
        }, { 1.001, 1.5, 2, 10 });
    }
    SUBCASE("asinh")
    {
        check_unary_function <Real> (FUNCTION (asinh), [] (Real x)
        {
            return 1 / std::sqrt (x * x + 1);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("atanh")
    {
        check_unary_function <Real> (FUNCTION (atanh), [] (Real x)
        {
            return 1 / ((1 - x) * (1 + x));
        }, { -0.999, -0.9, -0.5, 0.1, 0.5, 0.9, 0.999 });
    }
    SUBCASE("cosh")
    {
        check_unary_function <Real> (FUNCTION (cosh), [] (Real x)
        {
            return std::sinh (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sinh")
    {
        check_unary_function <Real> (FUNCTION (sinh), [] (Real x)
        {
            return std::cosh (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tanh")
    {
        check_unary_function <Real> (FUNCTION (tanh), [] (Real x)
        {
            return 1 / (std::cosh (x) * std::cosh (x));
        }, { -5, -2, -0.5, 0.1, 0.5, 2, 5 });
    }
    SUBCASE("exp")
    {
        check_unary_function <Real> (FUNCTION (exp), [] (Real x)
        {
            return std::exp (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("exp2")
    {
        check_unary_function <Real> (FUNCTION (exp2), [] (Real x)
        {
            return std::exp2 (x) * std::numbers::ln2_v <Real>;
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("expm1")
    {
        check_unary_function <Real> (FUNCTION (expm1), [] (Real x)
        {
            return std::exp (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log")
    {
        check_unary_function <Real> (FUNCTION (log), [] (Real x)
        {
            return 1 / x;
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log10")
    {
        check_unary_function <Real> (FUNCTION (log10), [] (Real x)
        {
            return 1 / (x * std::numbers::ln10_v <Real>);
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log1p")
    {
        check_unary_function <Real> (FUNCTION (log1p), [] (Real x)
        {
            return 1 / (1 + x);
        }, { -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log2")
    {
        check_unary_function <Real> (FUNCTION (log2), [] (Real x)
        {
            return 1 / (x * std::numbers::ln2_v <Real>);
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("cbrt")
    {
        check_unary_function <Real> (FUNCTION (cbrt), [] (Real x)
        {
            return 1 / (3 * std::cbrt (x) * std::cbrt (x));
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("abs")
    {
        check_unary_function <Real> (FUNCTION (abs), sign, { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("fabs")
    {
        check_unary_function <Real> (FUNCTION (fabs), sign, { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("sqrt")
    {
        check_unary_function <Real> (FUNCTION (sqrt), [] (Real x)
        {
            return 1 / (2 * std::sqrt (x));
        }, { 0.1, 0.5, 2, 10 });
    }
}
SUBCASE("Arithmetic")
{
    for (auto const& [x0, y0]: { std::pair { 3., 2. }, { -0.5, 4. }, { 2., -0.25 } })
    {
            const auto
        x = opaque <Real> (x0);
            const auto
        y = opaque <Real> (y0);
        check_function ([] (auto const& a, auto const& b) { return a + b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, 1 }
        );
        check_function ([] (auto const& a, auto const& b) { return a - b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, -1 }
        );
        check_function ([] (auto const& a, auto const& b) { return a * b; }
            , std::array { x, y }
            , std::array { y, x }
        );
        check_function ([] (auto const& a, auto const& b) { return a / b; }
            , std::array { x, y }
            , std::array { 1 / y, -x / (y * y) }
        );
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
            check_function (FUNCTION (atan2)
                , std::array { y, x }
                , std::array { x / r2, -y / r2 }
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
            check_function (FUNCTION (hypot)
                , std::array { x, y }
                , std::array { x / h, y / h }
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
            check_function (FUNCTION (pow)
                , std::array { x, y }
                , std::array { y * std::pow (x, y - 1), std::pow (x, y) * std::log (x) }
            );
        }
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
SUBCASE("Functions of three arguments")
{
    SUBCASE("hypot")
    {
            const auto
        x = std::array { opaque <Real> (2.), opaque <Real> (-1.), opaque <Real> (3.) };
            const auto
        h = std::hypot (x[0], x[1], x[2]);
        check_function (FUNCTION (hypot), x
            , std::array { x[0] / h, x[1] / h, x[2] / h }
        );
    }
}
} // TEST_CASE_TEMPLATE("Functions")
