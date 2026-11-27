// Tools for the tests: comparison of numbers in ulps, checks of the
// functions and of their derivatives, and of the mixed types.
#pragma once
#include <doctest/doctest.h>
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
