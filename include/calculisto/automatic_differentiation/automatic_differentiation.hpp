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
    return dual_t { 
          acos (a.value)
        , -a.differentials / sqrt ((1 - a.value) * (1 + a.value)) };
}
    template <std::size_t N, class T>
    auto
asin (dual_t <N, T> const& a) noexcept
{
        using std::asin, std::sqrt;
    // (1 - x) (1 + x) rather than 1 - x^2, which cancels near |x| = 1.
    return dual_t { 
          asin (a.value)
        , a.differentials / sqrt ((1 - a.value) * (1 + a.value)) };
}
    template <std::size_t N, class T>
    auto
atan (dual_t <N, T> const& a) noexcept
{
        using std::atan;
    return dual_t { 
          atan (a.value)
        , a.differentials / (1 + a.value * a.value) };
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
        , -x.differentials * y.value / tmp 
         + y.differentials * x.value / tmp
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
    return dual_t { 
          atan2 (y.value, x)
        , y.differentials * x / (x * x + y.value * y.value)
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
    return dual_t { 
          atan2 (y, x.value)
        , -x.differentials * y / (x.value * x.value + y * y)
    };
}
    template <std::size_t N, class T>
    auto
cos (dual_t <N, T> const& a) noexcept
{
        using std::cos, std::sin;
    return dual_t { cos (a.value), -a.differentials * sin (a.value) };
}
    template <std::size_t N, class T>
    auto
sin (dual_t <N, T> const& a) noexcept
{
        using std::sin, std::cos;
    return dual_t { sin (a.value), a.differentials * cos (a.value) };
}
    template <std::size_t N, class T>
    auto
tan (dual_t <N, T> const& a) noexcept
{
        using std::tan;
        const auto
    tmp = tan (a.value);
    return dual_t { tmp, a.differentials * (1 + tmp * tmp) };
}
    template <std::size_t N, class T>
    auto
acosh (dual_t <N, T> const& a) noexcept
{
        using std::acosh, std::sqrt;
    // (x - 1) (x + 1) rather than x^2 - 1, which cancels near x = 1.
    return dual_t { 
          acosh (a.value)
        , a.differentials / sqrt ((a.value - 1) * (a.value + 1)) };
}
    template <std::size_t N, class T>
    auto
asinh (dual_t <N, T> const& a) noexcept
{
        using std::asinh, std::sqrt;
    return dual_t { 
          asinh (a.value)
        , a.differentials / sqrt (1 + a.value * a.value) };
}
    template <std::size_t N, class T>
    auto
atanh (dual_t <N, T> const& a) noexcept
{
        using std::atanh;
    // (1 - x) (1 + x) rather than 1 - x^2, which cancels near |x| = 1.
    return dual_t { 
          atanh (a.value)
        , a.differentials / ((1 - a.value) * (1 + a.value)) };
}
    template <std::size_t N, class T>
    auto
cosh (dual_t <N, T> const& a) noexcept
{
        using std::cosh, std::sinh;
    return dual_t { cosh (a.value), a.differentials * sinh (a.value) };
}
    template <std::size_t N, class T>
    auto
sinh (dual_t <N, T> const& a) noexcept
{
        using std::sinh, std::cosh;
    return dual_t { sinh (a.value), a.differentials * cosh (a.value) };
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
    return dual_t { tanh (a.value), a.differentials * tmp * tmp };
}
    template <std::size_t N, class T>
    auto
exp (dual_t <N, T> const& a) noexcept
{
        using std::exp;
        const auto
    tmp = exp (a.value);
    return dual_t { tmp, a.differentials * tmp };
}
    template <std::size_t N, class T>
    auto
exp2 (dual_t <N, T> const& a) noexcept
{
        using std::exp2;
        const auto
    tmp = exp2 (a.value);
    return dual_t { tmp, a.differentials * tmp * std::numbers::ln2_v <detail::floating_point_type_t <T>> };
}
    template <std::size_t N, class T>
    auto
expm1 (dual_t <N, T> const& a) noexcept
{
        using std::expm1, std::exp;
    return dual_t { expm1 (a.value), a.differentials * exp (a.value) };
}
// double frexp(double value, int* exp);
// int ilogb(double x);
// double ldexp(double x, int exp);
    template <std::size_t N, class T>
    auto
log (dual_t <N, T> const& a) noexcept
{
        using std::log;
    return dual_t { log (a.value), a.differentials / a.value };
}
    template <std::size_t N, class T>
    auto
log10 (dual_t <N, T> const& a) noexcept
{
        using std::log10;
    return dual_t { 
          log10 (a.value)
        , a.differentials / a.value / std::numbers::ln10_v <detail::floating_point_type_t <T>> 
    };
}
    template <std::size_t N, class T>
    auto
log1p (dual_t <N, T> const& a) noexcept
{
        using std::log1p;
    return dual_t { log1p (a.value), a.differentials / (a.value + 1) };
}
    template <std::size_t N, class T>
    auto
log2 (dual_t <N, T> const& a) noexcept
{
        using std::log2;
    return dual_t { 
          log2 (a.value)
        , a.differentials / a.value / std::numbers::ln2_v <detail::floating_point_type_t <T>> 
    };
}
// double logb(double x);
// double modf(double value, double* iptr);
// double scalbn(double x, int n);
// double scalbln(double x, long int n);
    template <std::size_t N, class T>
    auto
cbrt (dual_t <N, T> const& a) noexcept
{
        using std::cbrt;
        const auto
    tmp = cbrt (a.value);
    return dual_t { 
          tmp
        , a.differentials / (3 * tmp * tmp)
    };
}
    template <std::size_t N, class T>
    auto
abs (dual_t <N, T> const& a) noexcept
{
        using std::abs, std::copysign;
    return dual_t { 
          abs (a.value)
        , a.differentials * copysign (detail::floating_point_type_t <T> { 1 }, detail::floating_point_value (a.value))
    };
}
    template <std::size_t N, class T>
    auto
fabs (dual_t <N, T> const& a) noexcept
{
        using std::fabs, std::copysign;
    return dual_t { 
          fabs (a.value)
        , a.differentials * copysign (detail::floating_point_type_t <T> { 1 }, detail::floating_point_value (a.value))
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
        , x.differentials * x.value / tmp
        + y.differentials * y.value / tmp
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
        , x.differentials * x.value / tmp
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
        , y.differentials * y.value / tmp
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
        , x.differentials * x.value / tmp
        + y.differentials * y.value / tmp
        + z.differentials * z.value / tmp
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
        , x.differentials * x.value / tmp
        + y.differentials * y.value / tmp
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
        , x.differentials * x.value / tmp
        + z.differentials * z.value / tmp
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
        , y.differentials * y.value / tmp
        + z.differentials * z.value / tmp
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
        , z.differentials * z.value / tmp
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
        , x.differentials * x.value / tmp
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
        , y.differentials * y.value / tmp
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
    return dual_t { 
          tmp
        , a.differentials / (2 * tmp)
    };
}
} // namespace calculisto::automatic_differentiation


// TODO: swap

/*
namespace std {
    double acos(double x);
    double asin(double x);
    double atan(double x);
    double atan2(double y, double x);
    double cos(double x);
    double sin(double x);
    double tan(double x);
    double acosh(double x);
    double asinh(double x);
    double atanh(double x);
    double cosh(double x);
    double sinh(double x);
    double tanh(double x);
    double exp(double x);
    double exp2(double x);
    double expm1(double x);
    double frexp(double value, int* exp);
    int ilogb(double x);
    double ldexp(double x, int exp);
    double log(double x);
    double log10(double x);
    double log1p(double x);
    double log2(double x);
    double logb(double x);
    double modf(double value, double* iptr);
    double scalbn(double x, int n);
    double scalbln(double x, long int n);
    double cbrt(double x);
    // absolute values
    double abs(double j);
    double fabs(double x);
    double hypot(double x, double y);
    // three-dimensional hypotenuse
    double hypot(double x, double y, double z);
    double pow(double x, double y);
    double sqrt(double x);

-----8<----

    double erf(double x);
    double erfc(double x);
    double lgamma(double x);
    double tgamma(double x);
    double ceil(double x);
    double floor(double x);
    double nearbyint(double x);
    double rint(double x);
    long int lrint(double x);
    long long int llrint(double x);
    double round(double x);
    long int lround(double x);
    long long int llround(double x);
    double trunc(double x);
    double fmod(double x, double y);
    double remainder(double x, double y);
    double remquo(double x, double y, int* quo);
    double copysign(double x, double y);
    double nan(const char* tagp);
    double nextafter(double x, double y);
    double fdim(double x, double y);
    double fmax(double x, double y);
    double fmin(double x, double y);
    double fma(double x, double y, double z);
    // linear interpolation
    constexpr double lerp(double a, double b, double t);
    // classification / comparison functions
    int fpclassify(double x);
    bool isfinite(double x);
    bool isinf(double x);
    bool isnan(double x);
    bool isnormal(double x);
    bool signbit(double x);
    bool isgreater(double x, double y);
    bool isgreaterequal(double x, double y);
    bool isless(double x, double y);
    bool islessequal(double x, double y);
    bool islessgreater(double x, double y);
    bool isunordered(double x, double y);
    // mathematical special functions
    // associated Laguerre polynomials
    double       assoc_laguerre(unsigned n, unsigned m, double x);
    // associated Legendre functions
    double       assoc_legendre(unsigned l, unsigned m, double x);
    // beta function
    double       beta(double x, double y);
    // complete elliptic integral of the first kind
    double       comp_ellint_1(double k);
    // complete elliptic integral of the second kind
    double       comp_ellint_2(double k);
    // complete elliptic integral of the third kind
    double       comp_ellint_3(double k, double nu);
    // regular modified cylindrical Bessel functions
    double       cyl_bessel_i(double nu, double x);
    // cylindrical Bessel functions of the first kind
    double       cyl_bessel_j(double nu, double x);
    // irregular modified cylindrical Bessel functions
    double       cyl_bessel_k(double nu, double x);
    // cylindrical Neumann functions;
    // cylindrical Bessel functions of the second kind
    double       cyl_neumann(double nu, double x);
    // incomplete elliptic integral of the first kind
    double       ellint_1(double k, double phi);
    // incomplete elliptic integral of the second kind
    double       ellint_2(double k, double phi);
    // incomplete elliptic integral of the third kind
    double       ellint_3(double k, double nu, double phi);
    // exponential integral
    double       expint(double x);
    // Hermite polynomials
    double       hermite(unsigned n, double x);
    // Laguerre polynomials
    double       laguerre(unsigned n, double x);
    // Legendre polynomials
    double       legendre(unsigned l, double x);
    // Riemann zeta function
    double       riemann_zeta(double x);
    // spherical Bessel functions of the first kind
    double       sph_bessel(unsigned n, double x);
    // spherical associated Legendre functions
    double       sph_legendre(unsigned l, unsigned m, double theta);
    // spherical Neumann functions;
    // spherical Bessel functions of the second kind
    double       sph_neumann(unsigned n, double x);
}
*/
