#pragma once
#include "calculisto/array/array.hpp"
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <type_traits>

    namespace
calculisto::automatic_differentiation
{
    using calculisto::array::array_t;

// The arithmetic (the operators, compound assignment, the comparisons,
// dual_cast and independent_variable) is constexpr. The functions of <cmath>
// are not, like those of std in C++20. Everything is noexcept, except
// independent_variable, which throws std::out_of_range for an index out of
// range: the functions accept every argument, giving NaN where the result
// isn't defined, as for numbers.

    namespace
detail
{
        template <class T>
        constexpr bool
    is_dual = false;
} // namespace detail

// The types a dual can be made of: the floating-point types, and the duals,
// for higher derivatives (dual_t <N, dual_t <M>> gives the second
// derivatives). Not the integer types: the derivatives would be truncated.
// The higher derivatives are the derivatives of the formulas of the first
// derivatives below, computed with their rounding: accurate relative to the
// size of the terms they are made of, but not relative to their own value
// where they are much smaller than these terms. For example, the second
// derivative of hypot (x, y) with respect to x is computed as
// (h^2 - x^2) / h^3, with h = hypot (x, y), which cancels where |x| >> |y|.
    template <class T>
    concept
dual_value_type = std::floating_point <T> or detail::is_dual <T>;

// A number with its differentials with respect to N independent variables.
// It is an aggregate, like std::array: dual_t <N> {} is zero, dual_t <N> { x }
// is the constant x, and an independent variable is made with
// independent_variable. Unlike std::array, the differentials have a default
// member initializer, so that the constant x can be written dual_t <N> { x }
// without a warning about the missing initializer (-Wextra): default
// initialization (dual_t <N> d;) leaves the value indeterminate, like that of
// a double, but the differentials are zero.
    template <std::size_t N, dual_value_type T = double>
    struct
dual_t
{
        using
    value_type = T;
        using
    differentials_type = array_t <T, N>;

        value_type
    value;
        differentials_type
    differentials {};
};

    template <class T, std::size_t N>
dual_t (T, array_t <T, N>) -> dual_t <N, T>;

    namespace
detail
{
        template <std::size_t N, class T>
        constexpr bool
    is_dual <dual_t <N, T>> = true;

    // The floating-point type at the bottom of T: T itself, or that of the
    // value of a dual.
        template <class T>
        struct
    floating_point_type
    {
            using
        type = T;
    };
        template <std::size_t N, class T>
        struct
    floating_point_type <dual_t <N, T>>
    {
            using
        type = typename floating_point_type <T>::type;
    };
        template <class T>
        using
    floating_point_type_t = typename floating_point_type <T>::type;

    // The value of x at the bottom: x itself, or that of the value of a dual.
        template <class T>
        constexpr floating_point_type_t <T>
    floating_point_value (T const& x) noexcept
    {
        if constexpr (is_dual <T>)
        {
            return floating_point_value (x.value);
        }
        else
        {
            return x;
        }
    }

    // T and U are both floating-point types, or both duals with the same
    // number of differentials, of types with the same structure: a dual
    // combines only with the duals of the same independent variables at each
    // level.
        template <class T, class U>
        constexpr bool
    same_structure = std::floating_point <T> and std::floating_point <U>;

        template <std::size_t N, class T, class U>
        constexpr bool
    same_structure <dual_t <N, T>, dual_t <N, U>> = same_structure <T, U>;

    // x is zero, and so are its differentials at each level: unlike x == 0,
    // which compares the value only.
        template <class T>
        constexpr bool
    is_exactly_zero (T const& x) noexcept
    {
        if constexpr (is_dual <T>)
        {
            if (not is_exactly_zero (x.value))
            {
                return false;
            }
            for (auto const& d: x.differentials)
            {
                if (not is_exactly_zero (d))
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            return x == 0;
        }
    }

    // The differentials transformed by f, applied to each of them, except
    // the zero ones, which stay zero, even where f would multiply them by an
    // infinite or NaN factor (e.g. that of sqrt at 0): a constant doesn't
    // change a derivative, as in pow.
        template <class D, std::size_t N, class F>
        constexpr auto
    transform_differentials (array_t <D, N> const& differentials, F f) noexcept
    {
            using
        result_t = decltype (f (differentials[0]));
            auto
        r = array_t <result_t, N> {};
        for (std::size_t i = 0; i < N; ++i)
        {
            if (not is_exactly_zero (differentials[i]))
            {
                r[i] = f (differentials[i]);
            }
        }
        return r;
    }

    // The numbers a dual combines with: those of the arithmetic types. Not any
    // type that combines with the value, such as an array, which would make
    // array_t <dual_t <N>, M> times dual_t <N> ambiguous.
        template <class T>
        concept
    arithmetic = std::is_arithmetic_v <T>;
} // namespace detail

// The independent variable number index (counting from zero) among N, with
// the given value: its differential with respect to itself is one, and the
// others are zero. Throws std::out_of_range if index >= N. As for dual_t, T
// is double unless given: it is not deduced from the value, so that
// independent_variable <N> (2, 0) is not a dual of int. For the second
// derivatives, the value is itself an independent variable:
// independent_variable <N, dual_t <N>> (independent_variable <N> (x, i), i).
    template <std::size_t N, dual_value_type T = double>
    constexpr dual_t <N, T>
independent_variable (std::type_identity_t <T> value, std::size_t index)
{
        auto
    r = dual_t <N, T> { value };
    r.differentials.at (index) = T { 1 };
    return r;
}

// The dual a with its value and differentials converted to U, like
// array_cast: dual_t has no converting constructor. U has the structure of T:
// a floating-point type if T is one, a dual of the same number of
// differentials if T is one.
    template <dual_value_type U, std::size_t N, class T>
    requires detail::same_structure <T, U>
    constexpr dual_t <N, U>
dual_cast (dual_t <N, T> const& a) noexcept
{
        const auto
    convert = [] (T const& x) -> U
    {
        if constexpr (std::floating_point <U>)
        {
            return static_cast <U> (x);
        }
        else
        {
            return dual_cast <typename U::value_type> (x);
        }
    };
        auto
    r = dual_t <N, U> { convert (a.value) };
    for (std::size_t i = 0; i < N; ++i)
    {
        r.differentials[i] = convert (a.differentials[i]);
    }
    return r;
}

// The functions of several arguments below take any combination of duals and
// numbers, with at least one dual.
    namespace
detail
{
    // The value of x: that of a dual, or x itself, a number.
        template <class X>
        constexpr auto
    value_of (X const& x) noexcept
    {
        if constexpr (is_dual <X>)
        {
            return x.value;
        }
        else
        {
            return x;
        }
    }

    // x converted for a function whose result is of type Result: a dual to a
    // dual of Result, a number to the floating-point type at the bottom of
    // Result.
        template <class Result, class X>
        constexpr auto
    convert (X const& x) noexcept
    {
        if constexpr (is_dual <X>)
        {
            return dual_cast <Result> (x);
        }
        else
        {
            return static_cast <floating_point_type_t <Result>> (x);
        }
    }

    // The number of differentials of X: N for a dual_t <N, T>, 0 for a
    // number.
        template <class X>
        constexpr std::size_t
    differentials_count = 0;

        template <std::size_t N, class T>
        constexpr std::size_t
    differentials_count <dual_t <N, T>> = N;

    // X and Y combine as arguments of a function: one of them is a number, or
    // both are duals of the same structure.
        template <class X, class Y>
        constexpr bool
    compatible = not is_dual <X> or not is_dual <Y>;

        template <std::size_t N, class T, class U>
        constexpr bool
    compatible <dual_t <N, T>, dual_t <N, U>> = same_structure <T, U>;

        template <class... X>
        constexpr bool
    all_compatible = true;

        template <class X, class... Y>
        constexpr bool
    all_compatible <X, Y...> = (compatible <X, Y> and ...) and all_compatible <Y...>;

    // The arguments of a function of duals and numbers: at least one dual,
    // numbers of the arithmetic types, and duals of the same structure.
        template <class... X>
        concept
    dual_arguments = (is_dual <X> or ...)
        and ((is_dual <X> or arithmetic <X>) and ...)
        and all_compatible <X...>;

    // The number of differentials of the duals among X, which is the same
    // for all of them.
        template <class... X>
        constexpr std::size_t
    common_differentials_count = (differentials_count <X> | ...);

    // The term of the differentials of a function for one of its arguments,
    // x: its differentials, each transformed by f, or nothing for a number.
        struct
    no_term
    {};

        template <class D, class F>
        struct
    term_t
    {
            D const&
        differentials;
            F
        f;
    };

        template <class X, class F>
        constexpr auto
    term (X const& x, F f) noexcept
    {
        if constexpr (is_dual <X>)
        {
            return term_t <typename X::differentials_type, F> { x.differentials, f };
        }
        else
        {
            return no_term {};
        }
    }

    // The differentials of a function of several arguments, from the terms of
    // its arguments: for each differential, the sum of the terms, skipping
    // those whose differential is zero, as in transform_differentials, and
    // assigning the first term rather than adding it to zero, which would
    // turn a -0 into +0.
        template <class Result, std::size_t N, class... Term>
        constexpr array_t <Result, N>
    combine_terms (Term const&... terms) noexcept
    {
            auto
        r = array_t <Result, N> {};
        for (std::size_t i = 0; i < N; ++i)
        {
                auto
            first = true;
                const auto
            add = [&] <class T> (T const& t)
            {
                if constexpr (not std::same_as <T, no_term>)
                {
                    if (not is_exactly_zero (t.differentials[i]))
                    {
                        if (first)
                        {
                            r[i] = t.f (t.differentials[i]);
                            first = false;
                        }
                        else
                        {
                            r[i] += t.f (t.differentials[i]);
                        }
                    }
                }
            };
            (add (terms), ...);
        }
        return r;
    }

    // The quotient n of x by y, an integer, from the remainder r = x - n y.
        template <class X, class Y, class R>
        auto
    quotient (X const& x, Y const& y, R const& r) noexcept
    {
            using std::round;
            const auto
        x_value = floating_point_value (value_of (x));
            const auto
        y_value = floating_point_value (value_of (y));
        return round ((x_value - floating_point_value (r)) / y_value);
    }
} // namespace detail

// Comparisons compare the values only, like those of a number, so that code
// branching on a dual takes the same branch as with its value. The operators
// with a number on the left are rewritten from these ones. The number is
// converted explicitly, as for numbers, but without the warning about the
// conversion, which would be in this header rather than in the code using it.
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr bool
operator == (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
    return a.value == b.value;
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr bool
operator == (dual_t <N, T> const& a, U const& b) noexcept
{
    return a.value == static_cast <std::common_type_t <detail::floating_point_type_t <T>, U>> (b);
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr auto
operator <=> (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
    return a.value <=> b.value;
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator <=> (dual_t <N, T> const& a, U const& b) noexcept
{
    return a.value <=> static_cast <std::common_type_t <detail::floating_point_type_t <T>, U>> (b);
}

    template <std::size_t N, class T>
    constexpr auto
operator + (dual_t <N, T> a) noexcept
{
    return a;
}
    template <std::size_t N, class T>
    constexpr auto
operator - (dual_t <N, T> const& a) noexcept
{
    return dual_t { -a.value, -a.differentials };
}

// Arithmetic with a dual of another type, or with a number, gives a dual of
// the type of the result of the arithmetic on the values, e.g. a dual of
// float times a double is a dual of double. As in the arithmetic on numbers,
// the arguments are converted to that type first: result_t below.
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator + (dual_t <N, T> const& a, U const& b) noexcept
{
        using
    result_t = decltype (a.value + b);
        const auto
    x = dual_cast <result_t> (a);
    return dual_t { x.value + static_cast <detail::floating_point_type_t <result_t>> (b), x.differentials };
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator + (U const& a, dual_t <N, T> const& b) noexcept
{
        using
    result_t = decltype (a + b.value);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { static_cast <detail::floating_point_type_t <result_t>> (a) + y.value, y.differentials };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr auto
operator + (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using
    result_t = decltype (a.value + b.value);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { 
          x.value + y.value
        , x.differentials + y.differentials 
    };
}

    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator - (dual_t <N, T> const& a, U const& b) noexcept
{
        using
    result_t = decltype (a.value - b);
        const auto
    x = dual_cast <result_t> (a);
    return dual_t { x.value - static_cast <detail::floating_point_type_t <result_t>> (b), x.differentials };
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator - (U const& a, dual_t <N, T> const& b) noexcept
{
        using
    result_t = decltype (a - b.value);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { static_cast <detail::floating_point_type_t <result_t>> (a) - y.value, -y.differentials };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr auto
operator - (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using
    result_t = decltype (a.value - b.value);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { 
          x.value - y.value
        , x.differentials - y.differentials 
    };
}

    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator * (dual_t <N, T> const& a, U const& b) noexcept
{
        using
    result_t = decltype (a.value * b);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    s = static_cast <detail::floating_point_type_t <result_t>> (b);
    return dual_t { x.value * s, x.differentials * s };
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator * (U const& a, dual_t <N, T> const& b) noexcept
{
        using
    result_t = decltype (a * b.value);
        const auto
    s = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { s * y.value, y.differentials * s };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr auto
operator * (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using
    result_t = decltype (a.value * b.value);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { 
          x.value * y.value
        , x.differentials * y.value + y.differentials * x.value 
    };
}

    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator / (dual_t <N, T> const& a, U const& b) noexcept
{
        using
    result_t = decltype (a.value / b);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    s = static_cast <detail::floating_point_type_t <result_t>> (b);
    return dual_t { 
          x.value / s
        , x.differentials / s 
    };
}
    template <std::size_t N, class T, detail::arithmetic U>
    constexpr auto
operator / (U const& a, dual_t <N, T> const& b) noexcept
{
        using
    result_t = decltype (a / b.value);
        const auto
    s = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { 
          s / y.value
        , -y.differentials * s / y.value / y.value
    };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    constexpr auto
operator / (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using
    result_t = decltype (a.value / b.value);
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
    return dual_t { 
          x.value / y.value
        , (
              x.differentials * y.value 
            - y.differentials * x.value
          ) / y.value / y.value 
    };
}

// Compound assignment, with a number or a dual of the same structure: as for
// numbers, a op= b is a op b, computed in the type of the result, converted to
// the type of a. Without the warning about the conversion, which would be in
// this header rather than in the code using it.
    template <std::size_t N, class T, class U>
    requires requires (dual_t <N, T> const& a, U const& b) { dual_cast <T> (a + b); }
    constexpr dual_t <N, T>&
operator += (dual_t <N, T>& a, U const& b) noexcept
{
    a = dual_cast <T> (a + b);
    return a;
}
    template <std::size_t N, class T, class U>
    requires requires (dual_t <N, T> const& a, U const& b) { dual_cast <T> (a - b); }
    constexpr dual_t <N, T>&
operator -= (dual_t <N, T>& a, U const& b) noexcept
{
    a = dual_cast <T> (a - b);
    return a;
}
    template <std::size_t N, class T, class U>
    requires requires (dual_t <N, T> const& a, U const& b) { dual_cast <T> (a * b); }
    constexpr dual_t <N, T>&
operator *= (dual_t <N, T>& a, U const& b) noexcept
{
    a = dual_cast <T> (a * b);
    return a;
}
    template <std::size_t N, class T, class U>
    requires requires (dual_t <N, T> const& a, U const& b) { dual_cast <T> (a / b); }
    constexpr dual_t <N, T>&
operator /= (dual_t <N, T>& a, U const& b) noexcept
{
    a = dual_cast <T> (a / b);
    return a;
}

    template <std::size_t N, class T>
    auto
acos (dual_t <N, T> const& a) noexcept
{
        using std::acos, std::sqrt;
    // (1 - x) (1 + x) rather than 1 - x^2, which cancels near |x| = 1.
        const auto
    root = sqrt ((1 - a.value) * (1 + a.value));
    return dual_t {
          acos (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return -d / root; })
    };
}
    template <std::size_t N, class T>
    auto
asin (dual_t <N, T> const& a) noexcept
{
        using std::asin, std::sqrt;
    // (1 - x) (1 + x) rather than 1 - x^2, which cancels near |x| = 1.
        const auto
    root = sqrt ((1 - a.value) * (1 + a.value));
    return dual_t {
          asin (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / root; })
    };
}
    template <std::size_t N, class T>
    auto
atan (dual_t <N, T> const& a) noexcept
{
        using std::atan;
        const auto
    denominator = 1 + a.value * a.value;
    return dual_t {
          atan (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / denominator; })
    };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    auto
atan2 (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using std::atan2;
        using
    result_t = decltype (atan2 (a.value, b.value));
        const auto
    y = dual_cast <result_t> (a);
        const auto
    x = dual_cast <result_t> (b);
        const auto
    tmp = x.value * x.value + y.value * y.value;
    return dual_t {
          atan2 (y.value, x.value)
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return -d * y.value / tmp; })
        + detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * x.value / tmp; })
    };
}
    template <std::size_t N, class T, detail::arithmetic U>
    auto
atan2 (dual_t <N, T> const& a, U const& b) noexcept
{
        using std::atan2;
        using
    result_t = decltype (atan2 (a.value, b));
        const auto
    y = dual_cast <result_t> (a);
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    tmp = x * x + y.value * y.value;
    return dual_t {
          atan2 (y.value, x)
        , detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * x / tmp; })
    };
}
    template <std::size_t N, class T, detail::arithmetic U>
    auto
atan2 (U const& a, dual_t <N, T> const& b) noexcept
{
        using std::atan2;
        using
    result_t = decltype (atan2 (a, b.value));
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    x = dual_cast <result_t> (b);
        const auto
    tmp = x.value * x.value + y * y;
    return dual_t {
          atan2 (y, x.value)
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return -d * y / tmp; })
    };
}
    template <std::size_t N, class T>
    auto
cos (dual_t <N, T> const& a) noexcept
{
        using std::cos, std::sin;
        const auto
    sine = sin (a.value);
    return dual_t {
          cos (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return -d * sine; })
    };
}
    template <std::size_t N, class T>
    auto
sin (dual_t <N, T> const& a) noexcept
{
        using std::sin, std::cos;
        const auto
    cosine = cos (a.value);
    return dual_t {
          sin (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * cosine; })
    };
}
    template <std::size_t N, class T>
    auto
tan (dual_t <N, T> const& a) noexcept
{
        using std::tan;
        const auto
    tmp = tan (a.value);
        const auto
    factor = 1 + tmp * tmp;
    return dual_t {
          tmp
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * factor; })
    };
}
    template <std::size_t N, class T>
    auto
acosh (dual_t <N, T> const& a) noexcept
{
        using std::acosh, std::sqrt;
    // (x - 1) (x + 1) rather than x^2 - 1, which cancels near x = 1.
        const auto
    root = sqrt ((a.value - 1) * (a.value + 1));
    return dual_t {
          acosh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / root; })
    };
}
    template <std::size_t N, class T>
    auto
asinh (dual_t <N, T> const& a) noexcept
{
        using std::asinh, std::sqrt;
        const auto
    root = sqrt (1 + a.value * a.value);
    return dual_t {
          asinh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / root; })
    };
}
    template <std::size_t N, class T>
    auto
atanh (dual_t <N, T> const& a) noexcept
{
        using std::atanh;
    // (1 - x) (1 + x) rather than 1 - x^2, which cancels near |x| = 1.
        const auto
    denominator = (1 - a.value) * (1 + a.value);
    return dual_t {
          atanh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / denominator; })
    };
}
    template <std::size_t N, class T>
    auto
cosh (dual_t <N, T> const& a) noexcept
{
        using std::cosh, std::sinh;
        const auto
    hyperbolic_sine = sinh (a.value);
    return dual_t {
          cosh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * hyperbolic_sine; })
    };
}
    template <std::size_t N, class T>
    auto
sinh (dual_t <N, T> const& a) noexcept
{
        using std::sinh, std::cosh;
        const auto
    hyperbolic_cosine = cosh (a.value);
    return dual_t {
          sinh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * hyperbolic_cosine; })
    };
}
    template <std::size_t N, class T>
    auto
tanh (dual_t <N, T> const& a) noexcept
{
        using std::tanh, std::cosh;
    // 1 / cosh^2 rather than 1 - tanh^2, which cancels for large |x|, and
    // squaring 1 / cosh rather than cosh, which would overflow first.
        const auto
    tmp = 1 / cosh (a.value);
    return dual_t {
          tanh (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * tmp * tmp; })
    };
}
    template <std::size_t N, class T>
    auto
exp (dual_t <N, T> const& a) noexcept
{
        using std::exp;
        const auto
    tmp = exp (a.value);
    return dual_t {
          tmp
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * tmp; })
    };
}
    template <std::size_t N, class T>
    auto
exp2 (dual_t <N, T> const& a) noexcept
{
        using std::exp2;
        const auto
    tmp = exp2 (a.value);
        const auto
    ln2 = std::numbers::ln2_v <detail::floating_point_type_t <T>>;
    return dual_t {
          tmp
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * tmp * ln2; })
    };
}
    template <std::size_t N, class T>
    auto
expm1 (dual_t <N, T> const& a) noexcept
{
        using std::expm1, std::exp;
        const auto
    exponential = exp (a.value);
    return dual_t {
          expm1 (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * exponential; })
    };
}
    template <std::size_t N, class T>
    auto
log (dual_t <N, T> const& a) noexcept
{
        using std::log;
    return dual_t {
          log (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / a.value; })
    };
}
    template <std::size_t N, class T>
    auto
log10 (dual_t <N, T> const& a) noexcept
{
        using std::log10;
        const auto
    ln10 = std::numbers::ln10_v <detail::floating_point_type_t <T>>;
    return dual_t {
          log10 (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / a.value / ln10; })
    };
}
    template <std::size_t N, class T>
    auto
log1p (dual_t <N, T> const& a) noexcept
{
        using std::log1p;
        const auto
    denominator = a.value + 1;
    return dual_t {
          log1p (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / denominator; })
    };
}
    template <std::size_t N, class T>
    auto
log2 (dual_t <N, T> const& a) noexcept
{
        using std::log2;
        const auto
    ln2 = std::numbers::ln2_v <detail::floating_point_type_t <T>>;
    return dual_t {
          log2 (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / a.value / ln2; })
    };
}
    template <std::size_t N, class T>
    auto
cbrt (dual_t <N, T> const& a) noexcept
{
        using std::cbrt;
        const auto
    tmp = cbrt (a.value);
        const auto
    denominator = 3 * tmp * tmp;
    return dual_t {
          tmp
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / denominator; })
    };
}
// abs and fabs have no derivative at 0: it is taken as that of the sign of
// the zero, copysign (1, x), +1 at +0 and -1 at -0, consistently with the
// signed zeros, and so that the derivative of abs (x) at x = 0 is the limit
// on the side of the sign of the zero.
    template <std::size_t N, class T>
    auto
abs (dual_t <N, T> const& a) noexcept
{
        using std::abs, std::copysign;
        const auto
    sign = copysign (detail::floating_point_type_t <T> { 1 }, detail::floating_point_value (a.value));
    return dual_t {
          abs (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * sign; })
    };
}
    template <std::size_t N, class T>
    auto
fabs (dual_t <N, T> const& a) noexcept
{
        using std::fabs, std::copysign;
        const auto
    sign = copysign (detail::floating_point_type_t <T> { 1 }, detail::floating_point_value (a.value));
    return dual_t {
          fabs (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * sign; })
    };
}
    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    auto
hypot (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b.value));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    tmp = hypot (x.value, y.value);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
        + detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
    };
}
    template <std::size_t N, class T, detail::arithmetic U>
    auto
hypot (dual_t <N, T> const& a, U const& b) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    tmp = hypot (x.value, y);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
    };
}
    template <std::size_t N, detail::arithmetic T, class U>
    auto
hypot (T const& a, dual_t <N, U> const& b) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a, b.value));
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    tmp = hypot (x, y.value);
    return dual_t {
          tmp
        , detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
    };
}
    template <std::size_t N, class T, class U, class V>
    requires detail::same_structure <T, U> and detail::same_structure <T, V>
    auto
hypot (dual_t <N, T> const& a, dual_t <N, U> const& b, dual_t <N, V> const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b.value, c.value));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    z = dual_cast <result_t> (c);
        const auto
    tmp = hypot (x.value, y.value, z.value);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
        + detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
        + detail::transform_differentials (z.differentials, [&] (auto const& d) { return d * z.value / tmp; })
    };
}
    template <std::size_t N, class T, class U, detail::arithmetic V>
    requires detail::same_structure <T, U>
    auto
hypot (dual_t <N, T> const& a, dual_t <N, U> const& b, V const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b.value, c));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    z = static_cast <detail::floating_point_type_t <result_t>> (c);
        const auto
    tmp = hypot (x.value, y.value, z);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
        + detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
    };
}
    template <std::size_t N, class T, detail::arithmetic U, class V>
    requires detail::same_structure <T, V>
    auto
hypot (dual_t <N, T> const& a, U const& b, dual_t <N, V> const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b, c.value));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    z = dual_cast <result_t> (c);
        const auto
    tmp = hypot (x.value, y, z.value);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
        + detail::transform_differentials (z.differentials, [&] (auto const& d) { return d * z.value / tmp; })
    };
}
    template <std::size_t N, detail::arithmetic T, class U, class V>
    requires detail::same_structure <U, V>
    auto
hypot (T const& a, dual_t <N, U> const& b, dual_t <N, V> const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a, b.value, c.value));
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    z = dual_cast <result_t> (c);
        const auto
    tmp = hypot (x, y.value, z.value);
    return dual_t {
          tmp
        , detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
        + detail::transform_differentials (z.differentials, [&] (auto const& d) { return d * z.value / tmp; })
    };
}
    template <std::size_t N, detail::arithmetic T, detail::arithmetic U, class V>
    auto
hypot (T const& a, U const& b, dual_t <N, V> const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a, b, c.value));
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    z = dual_cast <result_t> (c);
        const auto
    tmp = hypot (x, y, z.value);
    return dual_t {
          tmp
        , detail::transform_differentials (z.differentials, [&] (auto const& d) { return d * z.value / tmp; })
    };
}
    template <std::size_t N, class T, detail::arithmetic U, detail::arithmetic V>
    auto
hypot (dual_t <N, T> const& a, U const& b, V const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a.value, b, c));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    z = static_cast <detail::floating_point_type_t <result_t>> (c);
        const auto
    tmp = hypot (x.value, y, z);
    return dual_t {
          tmp
        , detail::transform_differentials (x.differentials, [&] (auto const& d) { return d * x.value / tmp; })
    };
}
    template <std::size_t N, detail::arithmetic T, class U, detail::arithmetic V>
    auto
hypot (T const& a, dual_t <N, U> const& b, V const& c) noexcept
{
        using std::hypot;
        using
    result_t = decltype (hypot (a, b.value, c));
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    z = static_cast <detail::floating_point_type_t <result_t>> (c);
        const auto
    tmp = hypot (x, y.value, z);
    return dual_t {
          tmp
        , detail::transform_differentials (y.differentials, [&] (auto const& d) { return d * y.value / tmp; })
    };
}
// d pow (a, b) = b pow (a, b - 1) da + pow (a, b) log (a) db. A term whose
// differential is zero contributes zero, even where its factor is infinite or
// NaN (log (a) for a <= 0, pow (a, b - 1) for a = 0 and b < 1): a constant
// exponent or base gives the same result as a scalar one, e.g. the derivative
// of pow (x, 2) is -4 at x = -2, and -0 at x = -0. The factors are also taken
// as zero where the mathematical derivative is: b pow (a, b - 1) for b = 0
// (pow (a, 0) is constant), and pow (a, b) log (a) where pow (a, b) is zero
// and a is not negative (a = 0 and b > 0, a = +inf and b < 0: pow (a, b) is
// zero around b). Elsewhere, they follow the mathematics, e.g. d/db pow (a, b)
// is NaN for a < 0, where pow (a, b) is not real around b, even where
// pow (a, b) underflows to zero, and -inf at a = b = 0.
    namespace
detail
{
    // The tests for zero are exact, and at each level of a nested dual: a
    // dual whose value is zero but not its differentials is not a constant.
        template <class Result, class T, class U>
        Result
    pow_partial_base (T const& a, U const& b) noexcept
    {
            using std::pow;
        if (is_exactly_zero (b))
        {
            return {};
        }
        return b * pow (a, b - 1);
    }

        template <class T, class P>
        P
    pow_partial_exponent (T const& a, P const& p) noexcept
    {
            using std::log;
        if (is_exactly_zero (p) and a >= 0)
        {
            return {};
        }
        return p * log (a);
    }
} // namespace detail

    template <std::size_t N, class T, class U>
    requires detail::same_structure <T, U>
    auto
pow (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept
{
        using std::pow;
        using
    result_t = decltype (pow (a.value, b.value));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    p = pow (x.value, y.value);
        const auto
    partial_x = detail::pow_partial_base <result_t> (x.value, y.value);
        const auto
    partial_y = detail::pow_partial_exponent (x.value, p);
        auto
    r = dual_t <N, result_t> { p };
    for (std::size_t i = 0; i < N; ++i)
    {
        // The first term is assigned rather than added to the zero of r, which
        // would turn a -0 into +0.
        if (not detail::is_exactly_zero (x.differentials[i]))
        {
            r.differentials[i] = x.differentials[i] * partial_x;
            if (not detail::is_exactly_zero (y.differentials[i]))
            {
                r.differentials[i] += y.differentials[i] * partial_y;
            }
        }
        else if (not detail::is_exactly_zero (y.differentials[i]))
        {
            r.differentials[i] = y.differentials[i] * partial_y;
        }
    }
    return r;
}
    template <std::size_t N, class T, detail::arithmetic U>
    auto
pow (dual_t <N, T> const& a, U const& b) noexcept
{
        using std::pow;
        using
    result_t = decltype (pow (a.value, b));
        const auto
    x = dual_cast <result_t> (a);
        const auto
    y = static_cast <detail::floating_point_type_t <result_t>> (b);
        const auto
    partial_x = detail::pow_partial_base <result_t> (x.value, y);
        auto
    r = dual_t <N, result_t> { pow (x.value, y) };
    for (std::size_t i = 0; i < N; ++i)
    {
        if (not detail::is_exactly_zero (x.differentials[i]))
        {
            r.differentials[i] = x.differentials[i] * partial_x;
        }
    }
    return r;
}
    template <std::size_t N, detail::arithmetic T, class U>
    auto
pow (T const& a, dual_t <N, U> const& b) noexcept
{
        using std::pow;
        using
    result_t = decltype (pow (a, b.value));
        const auto
    x = static_cast <detail::floating_point_type_t <result_t>> (a);
        const auto
    y = dual_cast <result_t> (b);
        const auto
    p = pow (x, y.value);
        const auto
    partial_y = detail::pow_partial_exponent (x, p);
        auto
    r = dual_t <N, result_t> { p };
    for (std::size_t i = 0; i < N; ++i)
    {
        if (not detail::is_exactly_zero (y.differentials[i]))
        {
            r.differentials[i] = y.differentials[i] * partial_y;
        }
    }
    return r;
}
    template <std::size_t N, class T>
    auto
sqrt (dual_t <N, T> const& a) noexcept
{
        using std::sqrt;
        const auto
    tmp = sqrt (a.value);
        const auto
    denominator = 2 * tmp;
    return dual_t {
          tmp
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d / denominator; })
    };
}

// The functions of <cmath> with a derivative, in its order, after those above.

    namespace
detail
{
    // exp (-x^2), accurately: x^2 rounded would make an error of x^2 times
    // its rounding, so it is split exactly into h + l, with
    // exp (-x^2) = exp (-h) exp (-l) = exp (-h) (1 - l), as l is less than
    // the rounding of h. At each level of a nested dual.
        template <class T>
        T
    exp_minus_square (T const& x) noexcept
    {
        if constexpr (is_dual <T>)
        {
                const auto
            e = exp_minus_square (x.value);
            return T {
                  e
                , transform_differentials (x.differentials, [&] (auto const& d) { return -2 * x.value * e * d; })
            };
        }
        else
        {
                using std::exp, std::fma;
                const auto
            h = x * x;
                const auto
            l = fma (x, x, -h);
            return exp (-h) * (1 - l);
        }
    }
} // namespace detail

    template <std::size_t N, class T>
    auto
erf (dual_t <N, T> const& a) noexcept
{
        using std::erf;
        const auto
    factor = 2 * std::numbers::inv_sqrtpi_v <detail::floating_point_type_t <T>> * detail::exp_minus_square (a.value);
    return dual_t {
          erf (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return d * factor; })
    };
}
    template <std::size_t N, class T>
    auto
erfc (dual_t <N, T> const& a) noexcept
{
        using std::erfc;
        const auto
    factor = 2 * std::numbers::inv_sqrtpi_v <detail::floating_point_type_t <T>> * detail::exp_minus_square (a.value);
    return dual_t {
          erfc (a.value)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return -d * factor; })
    };
}

// The mantissa x 2^-e, whose derivative is 2^-e, with e stored in exponent.
    template <std::size_t N, class T>
    auto
frexp (dual_t <N, T> const& a, int* exponent) noexcept
{
        using std::frexp, std::ldexp;
        const auto
    mantissa = frexp (a.value, exponent);
        const auto
    e = *exponent;
    return dual_t {
          mantissa
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return ldexp (d, -e); })
    };
}
// x 2^n, whose derivative is 2^n: computed exactly, as the value.
    template <std::size_t N, class T>
    auto
ldexp (dual_t <N, T> const& a, int exponent) noexcept
{
        using std::ldexp;
    return dual_t {
          ldexp (a.value, exponent)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return ldexp (d, exponent); })
    };
}
    template <std::size_t N, class T>
    auto
scalbn (dual_t <N, T> const& a, int exponent) noexcept
{
        using std::scalbn;
    return dual_t {
          scalbn (a.value, exponent)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return scalbn (d, exponent); })
    };
}
    template <std::size_t N, class T>
    auto
scalbln (dual_t <N, T> const& a, long exponent) noexcept
{
        using std::scalbln;
    return dual_t {
          scalbln (a.value, exponent)
        , detail::transform_differentials (a.differentials, [&] (auto const& d) { return scalbln (d, exponent); })
    };
}
// The fractional part, whose derivative is 1, with the integral part, a
// constant, stored in integral_part.
    template <std::size_t N, class T>
    auto
modf (dual_t <N, T> const& a, dual_t <N, T>* integral_part) noexcept
{
        using std::modf;
        auto
    integral = T {};
        const auto
    fractional = modf (a.value, &integral);
    *integral_part = dual_t <N, T> { integral };
    return dual_t <N, T> { fractional, a.differentials };
}

// fmod, remainder and remquo: x - n y, with n the quotient x / y rounded
// toward zero (fmod) or to the nearest (remainder, remquo), constant between
// the jumps: the derivative is dx - n dy.
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
fmod (A const& a, B const& b) noexcept
{
        using std::fmod;
        using
    result_t = decltype (fmod (detail::value_of (a), detail::value_of (b)));
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    r = fmod (detail::value_of (x), detail::value_of (y));
        const auto
    n = detail::quotient (x, y, r);
    return dual_t <detail::common_differentials_count <A, B>, result_t> {
          r
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B>> (
              detail::term (x, [] (auto const& d) { return d; })
            , detail::term (y, [&] (auto const& d) { return -d * n; })
          )
    };
}
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
remainder (A const& a, B const& b) noexcept
{
        using std::remainder;
        using
    result_t = decltype (remainder (detail::value_of (a), detail::value_of (b)));
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    r = remainder (detail::value_of (x), detail::value_of (y));
        const auto
    n = detail::quotient (x, y, r);
    return dual_t <detail::common_differentials_count <A, B>, result_t> {
          r
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B>> (
              detail::term (x, [] (auto const& d) { return d; })
            , detail::term (y, [&] (auto const& d) { return -d * n; })
          )
    };
}
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
remquo (A const& a, B const& b, int* quotient) noexcept
{
        using std::remquo;
        using
    result_t = decltype (remquo (detail::value_of (a), detail::value_of (b), quotient));
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    r = remquo (detail::value_of (x), detail::value_of (y), quotient);
        const auto
    n = detail::quotient (x, y, r);
    return dual_t <detail::common_differentials_count <A, B>, result_t> {
          r
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B>> (
              detail::term (x, [] (auto const& d) { return d; })
            , detail::term (y, [&] (auto const& d) { return -d * n; })
          )
    };
}

// The magnitude of x with the sign of y: the derivative with respect to x is
// the product of their signs, and that with respect to y is zero.
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
copysign (A const& a, B const& b) noexcept
{
        using std::copysign;
        using
    result_t = decltype (copysign (detail::value_of (a), detail::value_of (b)));
        using
    floating_point_t = detail::floating_point_type_t <result_t>;
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    sign = copysign (floating_point_t { 1 }, detail::floating_point_value (detail::value_of (x)))
         * copysign (floating_point_t { 1 }, detail::floating_point_value (detail::value_of (y)));
    return dual_t <detail::common_differentials_count <A, B>, result_t> {
          copysign (detail::value_of (x), detail::value_of (y))
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B>> (
              detail::term (x, [&] (auto const& d) { return d * sign; })
          )
    };
}

// The positive difference, max (x - y, 0): the derivative is that of x - y
// where x > y, and zero elsewhere, including where x = y, and where x or y is
// NaN.
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
fdim (A const& a, B const& b) noexcept
{
        using std::fdim;
        using
    result_t = decltype (fdim (detail::value_of (a), detail::value_of (b)));
        constexpr auto
    n = detail::common_differentials_count <A, B>;
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        auto
    r = dual_t <n, result_t> { fdim (detail::value_of (x), detail::value_of (y)) };
    if (detail::value_of (x) > detail::value_of (y))
    {
        r.differentials = detail::combine_terms <result_t, n> (
              detail::term (x, [] (auto const& d) { return d; })
            , detail::term (y, [] (auto const& d) { return -d; })
        );
    }
    return r;
}

// fmax and fmin: the derivative is that of the argument selected, the first
// one where they are equal, and the one that isn't NaN where one is, as for
// the value.
    namespace
detail
{
        template <bool Maximum, class A, class B>
        auto
    select (A const& a, B const& b) noexcept
    {
            using std::fmax, std::fmin, std::isnan;
            using
        result_t = decltype (fmax (value_of (a), value_of (b)));
            constexpr auto
        n = common_differentials_count <A, B>;
            const auto
        x = convert <result_t> (a);
            const auto
        y = convert <result_t> (b);
            const auto
        x_value = floating_point_value (value_of (x));
            const auto
        y_value = floating_point_value (value_of (y));
            const auto
        first = isnan (y_value)
            or (not isnan (x_value) and (Maximum ? x_value >= y_value : x_value <= y_value));
            auto
        r = dual_t <n, result_t> {
            Maximum ? fmax (value_of (x), value_of (y)) : fmin (value_of (x), value_of (y))
        };
        r.differentials = first
            ? combine_terms <result_t, n> (term (x, [] (auto const& d) { return d; }))
            : combine_terms <result_t, n> (term (y, [] (auto const& d) { return d; }));
        return r;
    }
} // namespace detail

    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
fmax (A const& a, B const& b) noexcept
{
    return detail::select <true> (a, b);
}
    template <class A, class B>
    requires detail::dual_arguments <A, B>
    auto
fmin (A const& a, B const& b) noexcept
{
    return detail::select <false> (a, b);
}

// x y + z, with a single rounding of the value: the derivative is
// y dx + x dy + dz.
    template <class A, class B, class C>
    requires detail::dual_arguments <A, B, C>
    auto
fma (A const& a, B const& b, C const& c) noexcept
{
        using std::fma;
        using
    result_t = decltype (fma (detail::value_of (a), detail::value_of (b), detail::value_of (c)));
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    z = detail::convert <result_t> (c);
    return dual_t <detail::common_differentials_count <A, B, C>, result_t> {
          fma (detail::value_of (x), detail::value_of (y), detail::value_of (z))
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B, C>> (
              detail::term (x, [&] (auto const& d) { return d * detail::value_of (y); })
            , detail::term (y, [&] (auto const& d) { return d * detail::value_of (x); })
            , detail::term (z, [] (auto const& d) { return d; })
          )
    };
}

// The linear interpolation a + t (b - a), with the properties of std::lerp
// for the value: the derivative is (1 - t) da + t db + (b - a) dt.
    template <class A, class B, class C>
    requires detail::dual_arguments <A, B, C>
    auto
lerp (A const& a, B const& b, C const& c) noexcept
{
        using std::lerp;
        using
    result_t = decltype (lerp (detail::value_of (a), detail::value_of (b), detail::value_of (c)));
        const auto
    x = detail::convert <result_t> (a);
        const auto
    y = detail::convert <result_t> (b);
        const auto
    t = detail::convert <result_t> (c);
        const auto
    t_value = detail::value_of (t);
    return dual_t <detail::common_differentials_count <A, B, C>, result_t> {
          lerp (detail::value_of (x), detail::value_of (y), t_value)
        , detail::combine_terms <result_t, detail::common_differentials_count <A, B, C>> (
              detail::term (x, [&] (auto const& d) { return d * (1 - t_value); })
            , detail::term (y, [&] (auto const& d) { return d * t_value; })
            , detail::term (t, [&] (auto const& d) { return d * (detail::value_of (y) - detail::value_of (x)); })
          )
    };
}

// The functions constant between their jumps: a dual with zero
// differentials, also at the jumps, so that they compose with the others,
// or, for those whose result is an integer, the integer.
    template <std::size_t N, class T>
    auto
ilogb (dual_t <N, T> const& a) noexcept
{
        using std::ilogb;
    return ilogb (a.value);
}
    template <std::size_t N, class T>
    auto
logb (dual_t <N, T> const& a) noexcept
{
        using std::logb;
    return dual_t <N, T> { logb (a.value) };
}
    template <std::size_t N, class T>
    auto
ceil (dual_t <N, T> const& a) noexcept
{
        using std::ceil;
    return dual_t <N, T> { ceil (a.value) };
}
    template <std::size_t N, class T>
    auto
floor (dual_t <N, T> const& a) noexcept
{
        using std::floor;
    return dual_t <N, T> { floor (a.value) };
}
    template <std::size_t N, class T>
    auto
nearbyint (dual_t <N, T> const& a) noexcept
{
        using std::nearbyint;
    return dual_t <N, T> { nearbyint (a.value) };
}
    template <std::size_t N, class T>
    auto
rint (dual_t <N, T> const& a) noexcept
{
        using std::rint;
    return dual_t <N, T> { rint (a.value) };
}
    template <std::size_t N, class T>
    auto
lrint (dual_t <N, T> const& a) noexcept
{
        using std::lrint;
    return lrint (a.value);
}
    template <std::size_t N, class T>
    auto
llrint (dual_t <N, T> const& a) noexcept
{
        using std::llrint;
    return llrint (a.value);
}
    template <std::size_t N, class T>
    auto
round (dual_t <N, T> const& a) noexcept
{
        using std::round;
    return dual_t <N, T> { round (a.value) };
}
    template <std::size_t N, class T>
    auto
lround (dual_t <N, T> const& a) noexcept
{
        using std::lround;
    return lround (a.value);
}
    template <std::size_t N, class T>
    auto
llround (dual_t <N, T> const& a) noexcept
{
        using std::llround;
    return llround (a.value);
}
    template <std::size_t N, class T>
    auto
trunc (dual_t <N, T> const& a) noexcept
{
        using std::trunc;
    return dual_t <N, T> { trunc (a.value) };
}

// Classification and comparison: of the values, like the comparison
// operators, whatever the differentials.
    template <std::size_t N, class T>
    auto
fpclassify (dual_t <N, T> const& a) noexcept
{
        using std::fpclassify;
    return fpclassify (a.value);
}
    template <std::size_t N, class T>
    auto
isfinite (dual_t <N, T> const& a) noexcept
{
        using std::isfinite;
    return isfinite (a.value);
}
    template <std::size_t N, class T>
    auto
isinf (dual_t <N, T> const& a) noexcept
{
        using std::isinf;
    return isinf (a.value);
}
    template <std::size_t N, class T>
    auto
isnan (dual_t <N, T> const& a) noexcept
{
        using std::isnan;
    return isnan (a.value);
}
    template <std::size_t N, class T>
    auto
isnormal (dual_t <N, T> const& a) noexcept
{
        using std::isnormal;
    return isnormal (a.value);
}
    template <std::size_t N, class T>
    auto
signbit (dual_t <N, T> const& a) noexcept
{
        using std::signbit;
    return signbit (a.value);
}

// The comparison functions are overloads for each combination of duals and
// numbers, rather than templates of any two arguments like the functions of
// two arguments above: the C runtime of Microsoft (UCRT) declares them as
// templates of any two arguments, with which those would be ambiguous.
#define CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(f)  \
    template <std::size_t N, class T, class U>              \
    requires detail::same_structure <T, U>                  \
    bool                                                    \
f (dual_t <N, T> const& a, dual_t <N, U> const& b) noexcept \
{                                                           \
        using std::f;                                       \
    return f (a.value, b.value);                            \
}                                                           \
    template <std::size_t N, class T, detail::arithmetic U> \
    bool                                                    \
f (dual_t <N, T> const& a, U const& b) noexcept             \
{                                                           \
        using std::f;                                       \
    return f (a.value, b);                                  \
}                                                           \
    template <std::size_t N, class T, detail::arithmetic U> \
    bool                                                    \
f (U const& a, dual_t <N, T> const& b) noexcept             \
{                                                           \
        using std::f;                                       \
    return f (a, b.value);                                  \
}
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(isgreater)
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(isgreaterequal)
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(isless)
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(islessequal)
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(islessgreater)
CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON(isunordered)
#undef CALCULISTO_AUTOMATIC_DIFFERENTIATION_COMPARISON
} // namespace calculisto::automatic_differentiation
