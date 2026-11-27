#include <doctest/doctest.h>
    using doctest::Approx;
#include <calculisto/automatic_differentiation/automatic_differentiation.hpp>
    using namespace calculisto::automatic_differentiation;
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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

// A dual is made of a floating-point type, or of duals, for the higher
// derivatives: with an integer type, the derivatives would be truncated.
    template <class T>
    concept
is_valid_dual_value_type = requires { typename dual_t <1, T>; };
static_assert (is_valid_dual_value_type <float>);
static_assert (is_valid_dual_value_type <double>);
static_assert (is_valid_dual_value_type <long double>);
static_assert (is_valid_dual_value_type <dual_t <1>>);
static_assert (is_valid_dual_value_type <dual_t <2, dual_t <3, float>>>);
static_assert (not is_valid_dual_value_type <int>);
static_assert (not is_valid_dual_value_type <dual_t <1>&>);

// Arithmetic and the functions of several arguments exist only with duals of
// the same number of differentials, and with numbers of arithmetic types,
// and asking whether they exist doesn't cause a compilation error.
    template <class A, class B>
    concept
has_any_operation = requires (A const& a, B const& b) { a + b; }
    or requires (A const& a, B const& b) { b + a; }
    or requires (A const& a, B const& b) { a - b; }
    or requires (A const& a, B const& b) { b - a; }
    or requires (A const& a, B const& b) { a * b; }
    or requires (A const& a, B const& b) { b * a; }
    or requires (A const& a, B const& b) { a / b; }
    or requires (A const& a, B const& b) { b / a; }
    or requires (A a, B const& b) { a += b; }
    or requires (A a, B const& b) { a -= b; }
    or requires (A a, B const& b) { a *= b; }
    or requires (A a, B const& b) { a /= b; }
    or requires (A const& a, B const& b) { atan2 (a, b); }
    or requires (A const& a, B const& b) { atan2 (b, a); }
    or requires (A const& a, B const& b) { hypot (a, b); }
    or requires (A const& a, B const& b) { hypot (b, a); }
    or requires (A const& a, B const& b) { hypot (a, b, a); }
    or requires (A const& a, B const& b) { hypot (b, a, b); }
    or requires (A const& a, B const& b) { pow (a, b); }
    or requires (A const& a, B const& b) { pow (b, a); }
    or requires (A const& a, B const& b) { fmod (a, b); }
    or requires (A const& a, B const& b) { fmod (b, a); }
    or requires (A const& a, B const& b) { copysign (a, b); }
    or requires (A const& a, B const& b) { fmax (b, a); }
    or requires (A const& a, B const& b) { fma (a, b, a); }
    or requires (A const& a, B const& b) { fma (b, b, a); }
    or requires (A const& a, B const& b) { lerp (a, a, b); }
    or requires (A const& a, B const& b) { isgreater (a, b); }
    or requires (A const& a, B const& b) { isunordered (b, a); };
static_assert (not has_any_operation <dual_t <1>, dual_t <2>>);
static_assert (not has_any_operation <dual_t <1>, std::complex <double>>);
static_assert (not has_any_operation <dual_t <1>, not_comparable>);
static_assert (not has_any_comparison <dual_t <1>, std::complex <double>>);

// Nested duals combine with numbers, and with the duals of the same
// structure: a dual of float with a dual of double, at each level. Not with
// the duals of another structure, whose differentials are with respect to
// other independent variables, at some level.
    template <class A, class B>
    concept
has_all_operations = requires (A const& a, B const& b, A c)
{
    a + b; b + a; a - b; b - a; a * b; b * a; a / b; b / a;
    a == b; b == a; a <=> b; b <=> a;
    atan2 (a, b); atan2 (b, a); hypot (a, b); hypot (b, a); hypot (a, b, a);
    pow (a, b); pow (b, a);
};
    using
nested_t = dual_t <2, dual_t <1>>;
static_assert (has_all_operations <nested_t, double>);
static_assert (has_all_operations <nested_t, int>);
static_assert (has_all_operations <nested_t, nested_t>);
static_assert (has_all_operations <nested_t, dual_t <2, dual_t <1, float>>>);
static_assert (not has_any_operation <nested_t, dual_t <1>>);
static_assert (not has_any_operation <nested_t, dual_t <2>>);
static_assert (not has_any_operation <nested_t, dual_t <2, dual_t <2>>>);
static_assert (not has_any_comparison <nested_t, dual_t <1>>);
static_assert (not has_any_comparison <nested_t, dual_t <2>>);

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

    namespace
second_order_checks
{
    // The argument number I of the function, of K: if bit I of Mask is set,
    // the independent variable number I at both levels of a nested dual, the
    // value x otherwise.
        template <unsigned Mask, std::size_t I, std::size_t K, class Real>
        auto
    argument (Real x)
    {
        if constexpr ((Mask >> I) & 1u)
        {
            return independent_variable <K, dual_t <K, Real>> (
                  independent_variable <K, Real> (x, I)
                , I
            );
        }
        else
        {
            return x;
        }
    }

    // Checks f at x, with the arguments selected by Mask being nested duals
    // and the others values: partials and second_partials are the first and
    // second partial derivatives of f at x.
        template <unsigned Mask, class F, class Real, std::size_t K, std::size_t... I>
        void
    check_function_mask (
          F f
        , std::array <Real, K> const& x
        , std::array <Real, K> const& partials
        , std::array <std::array <Real, K>, K> const& second_partials
        , std::index_sequence <I...>
    ){
        INFO ("Nested duals: mask ", Mask);
            const auto
        r = f (argument <Mask, I, K> (x[I])...);
        static_assert (std::same_as <decltype (r), dual_t <K, dual_t <K, Real>> const>);
        CHECK(r.value.value == f (x[I]...));
        // The second derivatives are accurate relative to the size of the
        // terms they are made of, not to their own value where they are much
        // smaller (see dual_value_type): they are compared within a few ulps
        // of the largest second partial derivative.
            auto
        scale = Real {};
        for (auto const& row: second_partials)
        {
            for (auto const& second_partial: row)
            {
                scale = std::max (scale, std::abs (second_partial));
            }
        }
            constexpr auto
        is_dual = [] (std::size_t i) { return ((Mask >> i) & 1u) != 0; };
        for (std::size_t i = 0; i < K; ++i)
        {
                const auto
            partial = is_dual (i) ? partials[i] : Real {};
            INFO ("Partial derivative ", i);
            CHECK(within_ulps <Real> (r.value.differentials[i], partial, ulps));
            CHECK(within_ulps <Real> (r.differentials[i].value, partial, ulps));
            for (std::size_t j = 0; j < K; ++j)
            {
                    const auto
                second_partial = is_dual (i) and is_dual (j) ? second_partials[i][j] : Real {};
                INFO ("Second partial derivative ", i, ", ", j, ": ", r.differentials[i].differentials[j], " expected ", second_partial);
                if (is_dual (i) and is_dual (j))
                {
                    CHECK(std::abs (r.differentials[i].differentials[j] - second_partial)
                        <= ulps * std::numeric_limits <Real>::epsilon () * scale);
                }
                else
                {
                    CHECK(r.differentials[i].differentials[j] == 0);
                }
            }
        }
    }

        template <class F, class Real, std::size_t K, std::size_t... M>
        void
    check_function_all_masks (
          F f
        , std::array <Real, K> const& x
        , std::array <Real, K> const& partials
        , std::array <std::array <Real, K>, K> const& second_partials
        , std::index_sequence <M...>
    ){
        (check_function_mask <M + 1> (
              f
            , x
            , partials
            , second_partials
            , std::make_index_sequence <K> {}
        ), ...);
    }
} // namespace second_order_checks

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

// The same, with second_partials the second partial derivatives of f at x:
// also with nested duals.
    template <class F, class Real, std::size_t K>
    void
check_function (
      F f
    , std::array <Real, K> const& x
    , std::array <Real, K> const& partials
    , std::array <std::array <Real, K>, K> const& second_partials
){
    check_function (f, x, partials);
    INFO ("At x[0] = ", x[0]);
    second_order_checks::check_function_all_masks (
          f
        , x
        , partials
        , second_partials
        , std::make_index_sequence <(1u << K) - 1> {}
    );
}

// Checks f, a function of one argument, at each of the points converted to
// Real, with derivative and second_derivative its derivatives: with duals,
// and with nested duals for the second derivative.
    template <class Real, class F, class D, class D2>
    void
check_unary_function (
      F f
    , D derivative
    , D2 second_derivative
    , std::initializer_list <double> points
){
    for (auto const point: points)
    {
            const auto
        x = opaque <Real> (point);
        check_function (f, std::array { x }, std::array { derivative (x) });
        INFO ("Second derivative at x = ", x);
            const auto
        r = f (independent_variable <1, dual_t <1, Real>> (independent_variable <1, Real> (x, 0), 0));
        static_assert (std::same_as <decltype (r), dual_t <1, dual_t <1, Real>> const>);
        CHECK(r.value.value == f (x));
        CHECK(within_ulps <Real> (r.value.differentials[0], derivative (x), ulps));
        CHECK(within_ulps <Real> (r.differentials[0].value, derivative (x), ulps));
        CHECK(within_ulps <Real> (r.differentials[0].differentials[0], second_derivative (x), ulps));
    }
}

// The functions to check, callable with values and with duals.
#define FUNCTION(name) \
    [] (auto const&... arguments) { using std::name; return name (arguments...); }

// Arguments of different types, for the tests of the mixed types: the value
// of x, and x converted to Result, a dual with dual_cast and a number to the
// floating-point type at the bottom of Result.
    template <class T>
    struct
bottom
{
        using
    type = T;
};
    template <std::size_t N, class T>
    struct
bottom <dual_t <N, T>>
{
        using
    type = typename bottom <T>::type;
};
    template <class X>
    auto
value_of (X const& x)
{
    if constexpr (requires { x.differentials; })
    {
        return x.value;
    }
    else
    {
        return x;
    }
}
    template <class Result, class X>
    auto
converted (X const& x)
{
    if constexpr (requires { x.differentials; })
    {
        return dual_cast <Result> (x);
    }
    else
    {
        return static_cast <typename bottom <Result>::type> (x);
    }
}

// a and b are equal, and so are their differentials at each level: unlike
// a == b, which compares the values only.
    template <class T>
    bool
identical (T const& a, T const& b)
{
    if constexpr (requires { a.differentials; })
    {
        if (not identical (a.value, b.value))
        {
            return false;
        }
        for (std::size_t i = 0; i < a.differentials.size (); ++i)
        {
            if (not identical (a.differentials[i], b.differentials[i]))
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        return a == b;
    }
}

// dual_cast converts to the types of the same structure: floating-point
// types for a dual of a floating-point type, duals of the same number of
// differentials for a dual of duals.
    template <class D, class U>
    concept
has_dual_cast = requires (D const& d) { dual_cast <U> (d); };

// Checks that f, with arguments of different types, gives a dual of the type
// of f of their values, equal to f of the arguments converted to that type:
// as with numbers, the arguments are converted first.
    template <class F, class... X>
    void
check_mixed (F f, X const&... arguments)
{
        using
    result_t = decltype (f (value_of (arguments)...));
        const auto
    r = f (arguments...);
    static_assert (std::same_as <decltype (r), dual_t <2, result_t> const>);
    CHECK(identical (r, f (converted <result_t> (arguments)...)));
}

TEST_CASE("Mixed types")
{
        const auto
    dual_float = dual_t <2, float> { opaque <float> (0.7), { 2, -3 } };
        const auto
    dual_double = dual_t <2, double> { opaque <double> (1.3), { -0.5, 5 } };
        const auto
    dual_long_double = dual_t <2, long double> { opaque <long double> (0.4), { 1, 0.25 } };
        const auto
    float_number = opaque <float> (0.6);
        const auto
    double_number = opaque <double> (1.7);
        const auto
    long_double_number = opaque <long double> (0.9);
        const auto
    int_number = opaque <int> (2);
SUBCASE("dual_cast")
{
        const auto
    d = dual_cast <double> (dual_float);
    static_assert (std::same_as <decltype (d), dual_t <2, double> const>);
    CHECK(d.value == static_cast <double> (dual_float.value));
    CHECK(d.differentials[0] == 2);
    CHECK(d.differentials[1] == -3);
    static_assert (has_dual_cast <dual_t <2, float>, long double>);
    static_assert (not has_dual_cast <dual_t <2, float>, int>);
}
    const auto
check_all_pairs = [&] (auto f)
{
    check_mixed (f, dual_float, dual_double);
    check_mixed (f, dual_double, dual_float);
    check_mixed (f, dual_float, dual_long_double);
    check_mixed (f, dual_long_double, dual_double);
    check_mixed (f, dual_float, double_number);
    check_mixed (f, double_number, dual_float);
    check_mixed (f, dual_double, int_number);
    check_mixed (f, int_number, dual_double);
    check_mixed (f, dual_double, float_number);
    check_mixed (f, float_number, dual_double);
    check_mixed (f, dual_double, long_double_number);
    check_mixed (f, long_double_number, dual_double);
    check_mixed (f, dual_long_double, int_number);
    check_mixed (f, int_number, dual_long_double);
};
SUBCASE("Arithmetic")
{
    check_all_pairs ([] (auto const& a, auto const& b) { return a + b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a - b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a * b; });
    check_all_pairs ([] (auto const& a, auto const& b) { return a / b; });
}
SUBCASE("Functions of two arguments")
{
    check_all_pairs (FUNCTION (atan2));
    check_all_pairs (FUNCTION (hypot));
    check_all_pairs (FUNCTION (pow));
}
SUBCASE("Functions of two arguments of <cmath>, after pow")
{
        const auto
    remquo_value = [] (auto const&... arguments)
    {
            using std::remquo;
            int
        quotient;
        return remquo (arguments..., &quotient);
    };
    check_all_pairs (FUNCTION (fmod));
    check_all_pairs (FUNCTION (remainder));
    check_all_pairs (remquo_value);
    check_all_pairs (FUNCTION (copysign));
    check_all_pairs (FUNCTION (fdim));
    check_all_pairs (FUNCTION (fmax));
    check_all_pairs (FUNCTION (fmin));
}
SUBCASE("Comparison functions")
{
    // Of the values, converted as for the comparison operators.
    CHECK(isgreater (dual_double, dual_float) == std::isgreater (dual_double.value, dual_float.value));
    CHECK(isless (dual_float, double_number) == std::isless (dual_float.value, double_number));
    CHECK(islessequal (int_number, dual_long_double) == std::islessequal (int_number, dual_long_double.value));
}
SUBCASE("Functions of three arguments")
{
        const auto
    check_combinations = [&] (auto f)
    {
        check_mixed (f, dual_float, dual_double, dual_long_double);
        check_mixed (f, dual_float, double_number, int_number);
        check_mixed (f, int_number, dual_float, dual_double);
        check_mixed (f, float_number, double_number, dual_long_double);
    };
    check_combinations (FUNCTION (fma));
    check_combinations (FUNCTION (lerp));
    check_mixed (FUNCTION (hypot), dual_float, dual_double, dual_long_double);
    check_mixed (FUNCTION (hypot), dual_float, dual_double, float_number);
    check_mixed (FUNCTION (hypot), dual_float, int_number, dual_double);
    check_mixed (FUNCTION (hypot), int_number, dual_float, dual_long_double);
    check_mixed (FUNCTION (hypot), dual_float, double_number, int_number);
    check_mixed (FUNCTION (hypot), int_number, dual_float, double_number);
    check_mixed (FUNCTION (hypot), long_double_number, int_number, dual_float);
}
SUBCASE("Nested duals")
{
        using
    nested_float_t = dual_t <2, dual_t <2, float>>;
        using
    nested_double_t = dual_t <2, dual_t <2, double>>;
        const auto
    nested_float = nested_float_t {
          { opaque <float> (0.7), { 2, -3 } }
        , { dual_t <2, float> { 1, { 0.5, 0 } }, dual_t <2, float> { -2, { 0, 4 } } }
    };
        const auto
    nested_double = nested_double_t {
          { opaque <double> (1.3), { -0.5, 5 } }
        , { dual_t <2> { 3, { 0, -1 } }, dual_t <2> { 0.25, { 2, 0 } } }
    };
        const auto
    d = dual_cast <dual_t <2, double>> (nested_float);
    static_assert (std::same_as <decltype (d), nested_double_t const>);
    CHECK(d.value.value == static_cast <double> (nested_float.value.value));
    CHECK(d.differentials[1].differentials[1] == 4);
    static_assert (has_dual_cast <nested_float_t, dual_t <2, long double>>);
    static_assert (not has_dual_cast <nested_float_t, double>);
    static_assert (not has_dual_cast <nested_float_t, dual_t <1>>);
    static_assert (not has_dual_cast <dual_t <2, float>, dual_t <2>>);

    check_mixed ([] (auto const& a, auto const& b) { return a + b; }, nested_float, nested_double);
    check_mixed ([] (auto const& a, auto const& b) { return a * b; }, nested_double, nested_float);
    check_mixed ([] (auto const& a, auto const& b) { return a / b; }, nested_float, double_number);
    check_mixed ([] (auto const& a, auto const& b) { return a - b; }, int_number, nested_double);
    check_mixed (FUNCTION (atan2), nested_float, nested_double);
    check_mixed (FUNCTION (hypot), double_number, nested_float);
    check_mixed (FUNCTION (hypot), nested_float, int_number, nested_double);
    check_mixed (FUNCTION (pow), nested_float, nested_double);
    check_mixed (FUNCTION (pow), nested_float, int_number);
    check_mixed (FUNCTION (pow), double_number, nested_float);
    check_mixed (FUNCTION (fmod), nested_float, nested_double);
    check_mixed (FUNCTION (copysign), double_number, nested_float);
    check_mixed (FUNCTION (fmax), nested_float, int_number);
    check_mixed (FUNCTION (fma), nested_float, double_number, nested_double);
    check_mixed (FUNCTION (lerp), int_number, nested_double, nested_float);
}
SUBCASE("A dual of float and an int")
{
    // As with numbers, an int gives a float, here without the warning that
    // -Wconversion gives for a float and a non-constant int.
        const auto
    r = dual_float * int_number;
    static_assert (std::same_as <decltype (r), dual_t <2, float> const>);
    CHECK(r.value == dual_float.value * static_cast <float> (int_number));
    CHECK(r.differentials[0] == 2 * static_cast <float> (int_number));
}
SUBCASE("Compound assignment")
{
    // As for numbers, the result is converted to the type of the dual.
        auto
    d = dual_float;
    d += double_number;
    CHECK(d.value == static_cast <float> (dual_float.value + double_number));
    d -= double_number;
    CHECK(d.value == static_cast <float> (static_cast <float> (dual_float.value + double_number) - double_number));
}
}

// Compound assignment exists with the numbers and the duals the arithmetic
// exists with, returns the dual assigned to, and doesn't exist for a
// constant.
    template <class A, class B>
    concept
has_all_compound_assignments = requires (A a, B const& b)
{
    { a += b } -> std::same_as <A&>;
    { a -= b } -> std::same_as <A&>;
    { a *= b } -> std::same_as <A&>;
    { a /= b } -> std::same_as <A&>;
};
static_assert (has_all_compound_assignments <dual_t <2>, dual_t <2>>);
static_assert (has_all_compound_assignments <dual_t <2, float>, dual_t <2>>);
static_assert (has_all_compound_assignments <dual_t <2>, int>);
static_assert (has_all_compound_assignments <dual_t <2, float>, double>);
static_assert (has_all_compound_assignments <nested_t, nested_t>);
static_assert (has_all_compound_assignments <nested_t, double>);
    template <class A, class B>
    concept
has_any_compound_assignment = requires (A a, B const& b) { a += b; }
    or requires (A a, B const& b) { a -= b; }
    or requires (A a, B const& b) { a *= b; }
    or requires (A a, B const& b) { a /= b; };
static_assert (not has_any_compound_assignment <dual_t <2> const, int>);
static_assert (not has_any_compound_assignment <dual_t <2>, dual_t <1>>);
static_assert (not has_any_compound_assignment <nested_t, dual_t <2>>);
static_assert (not has_any_compound_assignment <dual_t <2>, std::complex <double>>);

TEST_CASE("Compound assignment")
{
    // a op= b is a op b, converted to the type of a, and returns a.
        const auto
    check = [] (auto a, auto const& b, auto operation, auto compound_assignment)
    {
            using
        value_t = typename decltype (a)::value_type;
            const auto
        expected = dual_cast <value_t> (operation (a, b));
            auto&
        r = compound_assignment (a, b);
        CHECK(&r == &a);
        CHECK(identical (a, expected));
    };
        const auto
    check_all = [&] (auto const& a, auto const& b)
    {
        check (a, b
            , [] (auto const& u, auto const& v) { return u + v; }
            , [] (auto& u, auto const& v) -> auto& { return u += v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u - v; }
            , [] (auto& u, auto const& v) -> auto& { return u -= v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u * v; }
            , [] (auto& u, auto const& v) -> auto& { return u *= v; }
        );
        check (a, b
            , [] (auto const& u, auto const& v) { return u / v; }
            , [] (auto& u, auto const& v) -> auto& { return u /= v; }
        );
    };
        const auto
    x = dual_t <2> { opaque <double> (1.5), { 2, -3 } };
        const auto
    y = dual_t <2> { opaque <double> (-0.7), { 0.25, 4 } };
        const auto
    x_float = dual_t <2, float> { opaque <float> (0.3), { -1, 0.5 } };
        using
    inner_t = dual_t <2>;
        const auto
    nested_x = dual_t <2, inner_t> { inner_t { opaque <double> (1.5), { 1, 0 } }, { inner_t { 1 }, inner_t { 0, { 0.5, 2 } } } };
        const auto
    nested_y = dual_t <2, inner_t> { inner_t { opaque <double> (-0.7), { 0, 1 } }, { inner_t { 3, { 1, 0 } }, inner_t { -2 } } };
    SUBCASE("With a dual")
    {
        check_all (x, y);
        check_all (x, x_float);
    }
    SUBCASE("With a dual of a wider type: converted to the type of a")
    {
        check_all (x_float, x);
    }
    SUBCASE("With a number")
    {
        check_all (x, opaque <int> (3));
        check_all (x, opaque <double> (0.75));
        check_all (x_float, opaque <double> (0.75));
    }
    SUBCASE("Nested duals")
    {
        check_all (nested_x, nested_y);
        check_all (nested_x, opaque <double> (0.75));
    }
    SUBCASE("With itself")
    {
            auto
        z = independent_variable <1> (opaque <double> (3), 0);
        z *= z;
        CHECK(z.value == 9);
        CHECK(z.differentials[0] == 6);
        z -= z;
        CHECK(z.value == 0);
        CHECK(z.differentials[0] == 0);
    }
}

// The arithmetic is constexpr.
    constexpr auto
rational_function = [] (auto const& x) { return (x * x + 2 * x - 1) / (x + 1); };
// At 3: 14 / 4, and its derivative ((2 x + 2) (x + 1) - (x^2 + 2 x - 1)) /
// (x + 1)^2 = 18 / 16.
static_assert (rational_function (independent_variable <1> (3., 0)).value == 3.5);
static_assert (rational_function (independent_variable <1> (3., 0)).differentials[0] == 1.125);
static_assert ([]
{
        auto
    x = independent_variable <1> (3., 0);
    x *= x;
    x += 1;
    x -= 2;
    x /= 2;
    return x.value == 4 and x.differentials[0] == 3;
} ());
static_assert ((dual_t <1, float> { 1.5f, { 1 } } * 2.).value == 3);
static_assert (dual_cast <float> (dual_t <1> { 0.5, { 2 } }).differentials[0] == 2);
static_assert ([]
{
    // The second derivative of x^3 at 3: 6 x = 18.
        const auto
    x = independent_variable <1, dual_t <1>> (independent_variable <1> (3., 0), 0);
    return (x * x * x).differentials[0].differentials[0] == 18;
} ());
static_assert (independent_variable <2> (1., 1) < 2);

// Everything is noexcept, except independent_variable, which throws for an
// index out of range.
    template <class A, class B>
    concept
all_noexcept = requires (A a, A const& c, B const& b)
{
    requires noexcept (+c) and noexcept (-c);
    requires noexcept (c + b) and noexcept (b + c) and noexcept (c - b) and noexcept (b - c);
    requires noexcept (c * b) and noexcept (b * c) and noexcept (c / b) and noexcept (b / c);
    requires noexcept (a += b) and noexcept (a -= b) and noexcept (a *= b) and noexcept (a /= b);
    requires noexcept (c == b) and noexcept (c <=> b) and noexcept (c < b);
    requires noexcept (atan2 (c, b)) and noexcept (hypot (c, b)) and noexcept (hypot (c, b, c));
    requires noexcept (pow (c, b)) and noexcept (pow (b, c));
    requires noexcept (acos (c)) and noexcept (asin (c)) and noexcept (atan (c));
    requires noexcept (cos (c)) and noexcept (sin (c)) and noexcept (tan (c));
    requires noexcept (acosh (c)) and noexcept (asinh (c)) and noexcept (atanh (c));
    requires noexcept (cosh (c)) and noexcept (sinh (c)) and noexcept (tanh (c));
    requires noexcept (exp (c)) and noexcept (exp2 (c)) and noexcept (expm1 (c));
    requires noexcept (log (c)) and noexcept (log10 (c)) and noexcept (log1p (c)) and noexcept (log2 (c));
    requires noexcept (cbrt (c)) and noexcept (abs (c)) and noexcept (fabs (c)) and noexcept (sqrt (c));
    requires noexcept (dual_cast <float> (c));
    requires noexcept (erf (c)) and noexcept (erfc (c));
    requires noexcept (fmod (c, b)) and noexcept (remainder (b, c)) and noexcept (copysign (c, b));
    requires noexcept (fdim (c, b)) and noexcept (fmax (b, c)) and noexcept (fmin (c, b));
    requires noexcept (fma (c, b, c)) and noexcept (lerp (b, c, b));
    requires noexcept (ceil (c)) and noexcept (floor (c)) and noexcept (trunc (c)) and noexcept (round (c));
    requires noexcept (nearbyint (c)) and noexcept (rint (c)) and noexcept (logb (c)) and noexcept (ilogb (c));
    requires noexcept (lround (c)) and noexcept (llround (c)) and noexcept (lrint (c)) and noexcept (llrint (c));
    requires noexcept (fpclassify (c)) and noexcept (isfinite (c)) and noexcept (isinf (c));
    requires noexcept (isnan (c)) and noexcept (isnormal (c)) and noexcept (signbit (c));
    requires noexcept (isgreater (c, b)) and noexcept (isgreaterequal (b, c)) and noexcept (isless (c, b));
    requires noexcept (islessequal (c, b)) and noexcept (islessgreater (c, b)) and noexcept (isunordered (b, c));
};
static_assert (all_noexcept <dual_t <2>, dual_t <2>>);
static_assert (all_noexcept <dual_t <2>, int>);
static_assert (all_noexcept <dual_t <2>, dual_t <2, float>>);
static_assert (not noexcept (independent_variable <2> (1., 0)));

// Arrays of duals combine with duals element-wise, as with numbers.
    template <class A, class B, class Result>
    concept
gives = requires (A const& a, B const& b)
{
    { a + b } -> std::same_as <Result>;
    { a - b } -> std::same_as <Result>;
    { a * b } -> std::same_as <Result>;
    { a / b } -> std::same_as <Result>;
};
static_assert (gives <array_t <dual_t <1>, 2>, dual_t <1>, array_t <dual_t <1>, 2>>);
static_assert (gives <dual_t <1>, array_t <dual_t <1>, 2>, array_t <dual_t <1>, 2>>);

TEST_CASE("automatic_differentiation.hpp")
{
SUBCASE("Construction")
{
    // An aggregate, like std::array.
    static_assert (std::is_aggregate_v <dual_t <2>>);

    // The types of the value and of the differentials.
    static_assert (std::same_as <dual_t <2, float>::value_type, float>);
    static_assert (std::same_as <dual_t <2, float>::differentials_type, array_t <float, 2>>);

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
SUBCASE("Arithmetic")
{
    for (auto const& [x0, y0]: { std::pair { 3., 2. }, { -0.5, 4. }, { 2., -0.25 } })
    {
            const auto
        x = opaque <Real> (x0);
            const auto
        y = opaque <Real> (y0);
            using
        second_partials_t = std::array <std::array <Real, 2>, 2>;
        check_function ([] (auto const& a, auto const& b) { return a + b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, 1 }
            , second_partials_t {}
        );
        check_function ([] (auto const& a, auto const& b) { return a - b; }
            , std::array { x, y }
            , std::array <Real, 2> { 1, -1 }
            , second_partials_t {}
        );
        check_function ([] (auto const& a, auto const& b) { return a * b; }
            , std::array { x, y }
            , std::array { y, x }
            , second_partials_t {{ { 0, 1 }, { 1, 0 } }}
        );
        check_function ([] (auto const& a, auto const& b) { return a / b; }
            , std::array { x, y }
            , std::array { 1 / y, -x / (y * y) }
            , second_partials_t {{
                  { 0, -1 / (y * y) }
                , { -1 / (y * y), 2 * x / (y * y * y) }
              }}
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
} // TEST_CASE_TEMPLATE("Functions")
