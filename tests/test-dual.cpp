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

// Tools to check the functions: the value of the result must be that of the
// std:: function, and its differentials must be the partial derivatives,
// computed independently in the tests, within a few ulps.

// a and b are equal within n ulps (relative to the larger of the two).
    bool
within_ulps (double a, double b, double n)
{
    if (a == b)
    {
        return true;
    }
    return std::abs (a - b)
        <= n * std::numeric_limits <double>::epsilon ()
            * std::max (std::abs (a), std::abs (b));
}

// The tolerance of the derivatives.
    constexpr double
ulps = 4;

// The value of x, which the compiler cannot know: the results are computed at
// run time, as in the std:: functions they are compared with.
    double
opaque (double x)
{
        volatile double
    v = x;
    return v;
}

    namespace
function_checks
{
    // The argument number I of the function: the independent variable number
    // I if bit I of Mask is set, the double x otherwise.
        template <unsigned Mask, std::size_t I, std::size_t K>
        auto
    argument (double x)
    {
        if constexpr ((Mask >> I) & 1u)
        {
            return independent_variable <K> (x, I);
        }
        else
        {
            return x;
        }
    }

    // Checks f at x, with the arguments selected by Mask being duals and the
    // others doubles: partials are the partial derivatives of f at x.
        template <unsigned Mask, class F, std::size_t K, std::size_t... I>
        void
    check_function_mask (
          F f
        , std::array <double, K> const& x
        , std::array <double, K> const& partials
        , std::index_sequence <I...>
    ){
        INFO ("Duals: mask ", Mask);
            const auto
        r = f (argument <Mask, I, K> (x[I])...);
        CHECK(r.value == f (x[I]...));
        for (std::size_t i = 0; i < K; ++i)
        {
            INFO ("Partial derivative ", i, ": ", r.differentials[i], " expected ", (((Mask >> i) & 1u) ? partials[i] : 0.));
            if ((Mask >> i) & 1u)
            {
                CHECK(within_ulps (r.differentials[i], partials[i], ulps));
            }
            else
            {
                CHECK(r.differentials[i] == 0);
            }
        }
    }

        template <class F, std::size_t K, std::size_t... M>
        void
    check_function_all_masks (
          F f
        , std::array <double, K> const& x
        , std::array <double, K> const& partials
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
// derivatives there: with each combination of duals and doubles as arguments.
    template <class F, std::size_t K>
    void
check_function (
      F f
    , std::array <double, K> const& x
    , std::array <double, K> const& partials
){
    INFO ("At x[0] = ", x[0]);
    function_checks::check_function_all_masks (
          f
        , x
        , partials
        , std::make_index_sequence <(1u << K) - 1> {}
    );
}

// Checks f, a function of one argument, at each of the points, with derivative
// its derivative: with an independent variable as argument, and with a dual
// whose differentials are not all zero or one, for the chain rule.
    template <class F, class D>
    void
check_unary_function (F f, D derivative, std::initializer_list <double> points)
{
    for (auto const point: points)
    {
            const auto
        x = opaque (point);
        check_function (f, std::array { x }, std::array { derivative (x) });
        INFO ("At x = ", x, ", with differentials { 2, -3 }");
            const auto
        r = f (dual_t <2> { x, { 2., -3. } });
        CHECK(r.value == f (x));
        CHECK(within_ulps (r.differentials[0],  2 * derivative (x), ulps));
        CHECK(within_ulps (r.differentials[1], -3 * derivative (x), ulps));
    }
}

// The functions to check, callable with doubles and with duals.
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
SUBCASE("Functions of one argument")
{
        const auto
    inverse_sqrt_one_minus_square = [] (double x)
    {
        return 1 / std::sqrt ((1 - x) * (1 + x));
    };
        const auto
    sign = [] (double x)
    {
        return x < 0 ? -1. : 1.;
    };
    SUBCASE("acos")
    {
        check_unary_function (FUNCTION (acos), [&] (double x)
        {
            return -inverse_sqrt_one_minus_square (x);
        }, { -0.9, -0.5, 0.1, 0.5, 0.9 });
    }
    SUBCASE("asin")
    {
        check_unary_function (FUNCTION (asin), inverse_sqrt_one_minus_square
            , { -0.9, -0.5, 0.1, 0.5, 0.9 });
    }
    SUBCASE("atan")
    {
        check_unary_function (FUNCTION (atan), [] (double x)
        {
            return 1 / (1 + x * x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("cos")
    {
        check_unary_function (FUNCTION (cos), [] (double x)
        {
            return -std::sin (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sin")
    {
        check_unary_function (FUNCTION (sin), [] (double x)
        {
            return std::cos (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tan")
    {
        check_unary_function (FUNCTION (tan), [] (double x)
        {
            return 1 / (std::cos (x) * std::cos (x));
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("acosh")
    {
        check_unary_function (FUNCTION (acosh), [] (double x)
        {
            return 1 / std::sqrt ((x - 1) * (x + 1));
        }, { 1.5, 2, 10 });
    }
    SUBCASE("asinh")
    {
        check_unary_function (FUNCTION (asinh), [] (double x)
        {
            return 1 / std::sqrt (x * x + 1);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("atanh")
    {
        check_unary_function (FUNCTION (atanh), [] (double x)
        {
            return 1 / ((1 - x) * (1 + x));
        }, { -0.9, -0.5, 0.1, 0.5, 0.9 });
    }
    SUBCASE("cosh")
    {
        check_unary_function (FUNCTION (cosh), [] (double x)
        {
            return std::sinh (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("sinh")
    {
        check_unary_function (FUNCTION (sinh), [] (double x)
        {
            return std::cosh (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("tanh")
    {
        check_unary_function (FUNCTION (tanh), [] (double x)
        {
            return 1 / (std::cosh (x) * std::cosh (x));
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("exp")
    {
        check_unary_function (FUNCTION (exp), [] (double x)
        {
            return std::exp (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("exp2")
    {
        check_unary_function (FUNCTION (exp2), [] (double x)
        {
            return std::exp2 (x) * std::numbers::ln2;
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("expm1")
    {
        check_unary_function (FUNCTION (expm1), [] (double x)
        {
            return std::exp (x);
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log")
    {
        check_unary_function (FUNCTION (log), [] (double x)
        {
            return 1 / x;
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log10")
    {
        check_unary_function (FUNCTION (log10), [] (double x)
        {
            return 1 / (x * std::numbers::ln10);
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("log1p")
    {
        check_unary_function (FUNCTION (log1p), [] (double x)
        {
            return 1 / (1 + x);
        }, { -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("log2")
    {
        check_unary_function (FUNCTION (log2), [] (double x)
        {
            return 1 / (x * std::numbers::ln2);
        }, { 0.1, 0.5, 2, 10 });
    }
    SUBCASE("cbrt")
    {
        check_unary_function (FUNCTION (cbrt), [] (double x)
        {
            return 1 / (3 * std::cbrt (x) * std::cbrt (x));
        }, { -2, -0.5, 0.1, 0.5, 2 });
    }
    SUBCASE("abs")
    {
        check_unary_function (FUNCTION (abs), sign, { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("fabs")
    {
        check_unary_function (FUNCTION (fabs), sign, { -2, -0.5, 0.5, 2 });
    }
    SUBCASE("sqrt")
    {
        check_unary_function (FUNCTION (sqrt), [] (double x)
        {
            return 1 / (2 * std::sqrt (x));
        }, { 0.1, 0.5, 2, 10 });
    }
}
SUBCASE("Functions of two arguments")
{
    SUBCASE("atan2")
    {
        for (auto const& [y, x]: { std::pair { 3., 2. }, { 0.5, -1. }, { -2., 0.5 } })
        {
                const auto
            r2 = opaque (x * x + y * y);
            check_function (FUNCTION (atan2)
                , std::array { opaque (y), opaque (x) }
                , std::array { x / r2, -y / r2 }
            );
        }
    }
    SUBCASE("hypot")
    {
        for (auto const& [x, y]: { std::pair { 2., 3. }, { -1., 0.5 }, { 0.5, -2. } })
        {
                const auto
            h = std::hypot (opaque (x), opaque (y));
            check_function (FUNCTION (hypot)
                , std::array { opaque (x), opaque (y) }
                , std::array { x / h, y / h }
            );
        }
    }
    SUBCASE("pow")
    {
        for (auto const& [x, y]: { std::pair { 2., 3. }, { 0.5, -1.5 }, { 3., 0.5 } })
        {
                const auto
            p = std::pow (opaque (x), opaque (y));
            check_function (FUNCTION (pow)
                , std::array { opaque (x), opaque (y) }
                , std::array { y * std::pow (x, y - 1), p * std::log (x) }
            );
        }
    }
}
SUBCASE("pow where the base is not positive")
{
    // A term of the derivative whose differential is zero contributes zero,
    // even where its factor is infinite or NaN, so that a constant exponent
    // or base gives the same result as a double. The other terms follow the
    // mathematics: d/db pow (a, b) is NaN for a < 0, where pow (a, b) is not
    // real around b, zero for a = 0 and b > 0, and -inf at 0^0 (log (0) times
    // pow (0, 0) = 1).
        const auto
    nan = std::numeric_limits <double>::quiet_NaN ();
        const auto
    inf = std::numeric_limits <double>::infinity ();
        const auto
    same = [] (double result, double expected)
    {
        return std::isnan (expected) ? std::isnan (result) : result == expected;
    };
        struct
    point_t
    {
        double a, b, partial_a, partial_b;
    };
    for (auto const& [a0, b0, partial_a, partial_b]: {
          point_t { -2, 0,   0,   nan }
        , point_t { -2, 0.5, nan, nan }
        , point_t { -2, 1,   1,   nan }
        , point_t { -2, 2,   -4,  nan }
        , point_t { -2, 3,   12,  nan }
        , point_t { 0,  0,   0,   -inf }
        , point_t { 0,  0.5, inf, 0 }
        , point_t { 0,  1,   1,   0 }
        , point_t { 0,  2,   0,   0 }
    }){
            const auto
        a = opaque (a0);
            const auto
        b = opaque (b0);
        INFO ("a = ", a, ", b = ", b);
        {
                const auto
            r = pow (independent_variable <2> (a, 0), independent_variable <2> (b, 1));
            CHECK(same (r.value, std::pow (a, b)));
            CHECK(same (r.differentials[0], partial_a));
            CHECK(same (r.differentials[1], partial_b));
        }
        CHECK(same (pow (independent_variable <1> (a, 0), dual_t <1> { b }).differentials[0], partial_a));
        CHECK(same (pow (independent_variable <1> (a, 0), b).differentials[0], partial_a));
        CHECK(same (pow (dual_t <1> { a }, independent_variable <1> (b, 0)).differentials[0], partial_b));
        CHECK(same (pow (a, independent_variable <1> (b, 0)).differentials[0], partial_b));
    }
}
SUBCASE("Functions of three arguments")
{
    SUBCASE("hypot")
    {
            const auto
        x = std::array { opaque (2.), opaque (-1.), opaque (3.) };
            const auto
        h = std::hypot (x[0], x[1], x[2]);
        check_function (FUNCTION (hypot), x
            , std::array { x[0] / h, x[1] / h, x[2] / h }
        );
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
