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

// A number with its differentials with respect to N independent variables.
// It is an aggregate, like std::array: dual_t <N> {} is zero, dual_t <N> { x }
// is the constant x, and an independent variable is made with
// independent_variable. Unlike std::array, the differentials have a default
// member initializer, so that the constant x can be written dual_t <N> { x }
// without a warning about the missing initializer (-Wextra): default
// initialization (dual_t <N> d;) leaves the value indeterminate, like that of
// a double, but the differentials are zero.
    template <std::size_t N, class T = double>
    struct
dual_t
{
        using
    value_type = T;

        T
    value;
        array_t <T, N>
    differentials {};
};

    template <class T, std::size_t N>
dual_t (T, array_t <T, N>) -> dual_t <N, T>;

// The independent variable number index (counting from zero) among N, with
// the given value: its differential with respect to itself is one, and the
// others are zero. Throws std::out_of_range if index >= N. As for dual_t, T
// is double unless given: it is not deduced from the value, so that
// independent_variable <N> (2, 0) is not a dual of int.
    template <std::size_t N, class T = double>
    constexpr dual_t <N, T>
independent_variable (std::type_identity_t <T> value, std::size_t index)
{
        auto
    r = dual_t <N, T> { value };
    r.differentials.at (index) = T { 1 };
    return r;
}

    namespace
detail
{
        template <class T>
        constexpr bool
    is_dual = false;

        template <std::size_t N, class T>
        constexpr bool
    is_dual <dual_t <N, T>> = true;
} // namespace detail

// Comparisons compare the values only, like those of a number, so that code
// branching on a dual takes the same branch as with its value. The operators
// with a scalar on the left are rewritten from these ones.
        template <std::size_t N, class T, class U>
        requires requires (T const& t, U const& u) 
        {
            { t == u } -> std::convertible_to <bool>;
        }
        constexpr bool
    operator == (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return a.value == b.value;
    }
        template <std::size_t N, class T, class U>
        requires (not detail::is_dual <U>) 
        and requires (T const& t, U const& u) 
        {
            { t == u } -> std::convertible_to <bool>;
        }
        constexpr bool
    operator == (dual_t <N, T> const& a, U const& b)
    {
        return a.value == b;
    }
        template <std::size_t N, class T, class U>
        requires requires (T const& t, U const& u) { t <=> u; }
        constexpr auto
    operator <=> (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return a.value <=> b.value;
    }
        template <std::size_t N, class T, class U>
        requires (not detail::is_dual <U>) 
        and requires (T const& t, U const& u) { t <=> u; }
        constexpr auto
    operator <=> (dual_t <N, T> const& a, U const& b)
    {
        return a.value <=> b;
    }

        template <std::size_t N, class T>
        auto
    operator + (dual_t <N, T> a)
    {
        return a;
    }
        template <std::size_t N, class T>
        auto
    operator - (dual_t <N, T> const& a)
    {
        return dual_t { -a.value, -a.differentials };
    }

        template <std::size_t N, class T, class U>
        auto
    operator + (dual_t <N, T> const& a, U const& b)
    {
        return dual_t { a.value + b, a.differentials };
    }
        template <std::size_t N, class T, class U>
        auto
    operator + (U const& a, dual_t <N, T> const& b)
    {
        return dual_t { a + b.value, b.differentials };
    }
        template <std::size_t N, class T, class U>
        auto
    operator + (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return dual_t { 
              a.value + b.value
            , a.differentials + b.differentials 
        };
    }
        template <std::size_t N, class T, class U>
        auto&
    operator += (dual_t <N, T>& a, U const& b)
    {
        a.value += b;
        return a;
    }

        template <std::size_t N, class T, class U>
        auto
    operator - (dual_t <N, T> const& a, U const& b)
    {
        return dual_t { a.value - b, a.differentials };
    }
        template <std::size_t N, class T, class U>
        auto
    operator - (U const& a, dual_t <N, T> const& b)
    {
        return dual_t { a - b.value, -b.differentials };
    }
        template <std::size_t N, class T, class U>
        auto
    operator - (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return dual_t { 
              a.value - b.value
            , a.differentials - b.differentials 
        };
    }
        template <std::size_t N, class T, class U>
        auto&
    operator -= (dual_t <N, T>& a, U const& b)
    {
        a.value -= b;
        return a;
    }

        template <std::size_t N, class T, class U>
        auto
    operator * (dual_t <N, T> const& a, U const& b)
    {
        return dual_t { a.value * b, a.differentials * b };
    }
        template <std::size_t N, class T, class U>
        auto
    operator * (U const& a, dual_t <N, T> const& b)
    {
        return dual_t { a * b.value, b.differentials * a };
    }
        template <std::size_t N, class T, class U>
        auto
    operator * (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return dual_t { 
              a.value * b.value
            , a.differentials * b.value + b.differentials * a.value 
        };
    }

        template <std::size_t N, class T, class U>
        auto
    operator / (dual_t <N, T> const& a, U const& b)
    {
        return dual_t { 
              a.value / b
            , a.differentials / b 
        };
    }
        template <std::size_t N, class T, class U>
        auto
    operator / (U const& a, dual_t <N, T> const& b)
    {
        return dual_t { 
              a / b.value
            , -b.differentials / b.value / b.value
        };
    }
        template <std::size_t N, class T, class U>
        auto
    operator / (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
        return dual_t { 
              a.value / b.value
            , (
                  a.differentials * b.value 
                - b.differentials * a.value
              ) / b.value / b.value 
        };
    }

        template <std::size_t N, class T>
        auto
    acos (dual_t <N, T> const& a)
    {
            using std::acos, std::sqrt;
        return dual_t { 
              acos (a.value)
            , -a.differentials / sqrt (1. - a.value * a.value) };
    }
        template <std::size_t N, class T>
        auto
    asin (dual_t <N, T> const& a)
    {
            using std::asin, std::sqrt;
        return dual_t { 
              asin (a.value)
            , a.differentials / sqrt (1. - a.value * a.value) };
    }
        template <std::size_t N, class T>
        auto
    atan (dual_t <N, T> const& a)
    {
            using std::atan;
        return dual_t { 
              atan (a.value)
            , a.differentials / (1. + a.value * a.value) };
    }
        template <std::size_t N, class T, class U>
        auto
    atan2 (dual_t <N, T> const& y, dual_t <N, U> const& x)
    {
            using std::atan2;
            const auto
        tmp = x.value * x.value + y.value * y.value;
        return dual_t { 
              atan2 (y.value, x.value)
            , -x.differentials * y.value / tmp 
             + y.differentials * x.value / tmp
        };
    }
        template <std::size_t N, class T, class U>
        auto
    atan2 (dual_t <N, T> const& y, U const& x)
    {
            using std::atan2;
        return dual_t { 
              atan2 (y.value, x)
            , y.differentials * x / (x * x + y.value * y.value)
        };
    }
        template <std::size_t N, class T, class U>
        auto
    atan2 (U const& y, dual_t <N, T> const& x)
    {
            using std::atan2;
        return dual_t { 
              atan2 (y, x.value)
            , -x.differentials * y / (x.value * x.value + y * y)
        };
    }
        template <std::size_t N, class T>
        auto
    cos (dual_t <N, T> const& a)
    {
            using std::cos, std::sin;
        return dual_t { cos (a.value), -a.differentials * sin (a.value) };
    }
        template <std::size_t N, class T>
        auto
    sin (dual_t <N, T> const& a)
    {
            using std::sin, std::cos;
        return dual_t { sin (a.value), a.differentials * cos (a.value) };
    }
        template <std::size_t N, class T>
        auto
    tan (dual_t <N, T> const& a)
    {
            using std::tan;
            const auto
        tmp = tan (a.value);
        return dual_t { tmp, a.differentials * (1 + tmp * tmp) };
    }
        template <std::size_t N, class T>
        auto
    acosh (dual_t <N, T> const& a)
    {
            using std::acosh, std::sqrt;
        return dual_t { 
              acosh (a.value)
            , a.differentials / sqrt (a.value * a.value - 1.) };
    }
        template <std::size_t N, class T>
        auto
    asinh (dual_t <N, T> const& a)
    {
            using std::asinh, std::sqrt;
        return dual_t { 
              asinh (a.value)
            , a.differentials / sqrt (1. + a.value * a.value) };
    }
        template <std::size_t N, class T>
        auto
    atanh (dual_t <N, T> const& a)
    {
            using std::atanh;
        return dual_t { 
              atanh (a.value)
            , a.differentials / (1. - a.value * a.value) };
    }
        template <std::size_t N, class T>
        auto
    cosh (dual_t <N, T> const& a)
    {
            using std::cosh, std::sinh;
        return dual_t { cosh (a.value), a.differentials * sinh (a.value) };
    }
        template <std::size_t N, class T>
        auto
    sinh (dual_t <N, T> const& a)
    {
            using std::sinh, std::cosh;
        return dual_t { sinh (a.value), a.differentials * cosh (a.value) };
    }
        template <std::size_t N, class T>
        auto
    tanh (dual_t <N, T> const& a)
    {
            using std::tanh;
            const auto
        tmp = tanh (a.value);
        return dual_t { tmp, a.differentials * (1 - tmp * tmp) };
    }
        template <std::size_t N, class T>
        auto
    exp (dual_t <N, T> const& a)
    {
            using std::exp;
            const auto
        tmp = exp (a.value);
        return dual_t { tmp, a.differentials * tmp };
    }
        template <std::size_t N, class T>
        auto
    exp2 (dual_t <N, T> const& a)
    {
            using std::exp2;
            const auto
        tmp = exp2 (a.value);
        return dual_t { tmp, a.differentials * tmp * std::numbers::ln2 };
    }
        template <std::size_t N, class T>
        auto
    expm1 (dual_t <N, T> const& a)
    {
            using std::expm1, std::exp;
        return dual_t { expm1 (a.value), a.differentials * exp (a.value) };
    }
    // double frexp(double value, int* exp);
    // int ilogb(double x);
    // double ldexp(double x, int exp);
        template <std::size_t N, class T>
        auto
    log (dual_t <N, T> const& a)
    {
            using std::log;
        return dual_t { log (a.value), a.differentials / a.value };
    }
        template <std::size_t N, class T>
        auto
    log10 (dual_t <N, T> const& a)
    {
            using std::log10;
        return dual_t { 
              log10 (a.value)
            , a.differentials / a.value / std::numbers::ln10 
        };
    }
        template <std::size_t N, class T>
        auto
    log1p (dual_t <N, T> const& a)
    {
            using std::log1p;
        return dual_t { log1p (a.value), a.differentials / (a.value + 1.) };
    }
        template <std::size_t N, class T>
        auto
    log2 (dual_t <N, T> const& a)
    {
            using std::log2;
        return dual_t { 
              log2 (a.value)
            , a.differentials / a.value / std::numbers::ln2 
        };
    }
    // double logb(double x);
    // double modf(double value, double* iptr);
    // double scalbn(double x, int n);
    // double scalbln(double x, long int n);
        template <std::size_t N, class T>
        auto
    cbrt (dual_t <N, T> const& a)
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
    abs (dual_t <N, T> const& a)
    {
            using std::abs, std::copysign;
        return dual_t { 
              abs (a.value)
            , a.differentials * copysign (1., a.value)
        };
    }
        template <std::size_t N, class T>
        auto
    fabs (dual_t <N, T> const& a)
    {
            using std::fabs;
        return dual_t { 
              fabs (a.value)
            , a.differentials  * copysign (1., a.value)
        };
    }
        template <std::size_t N, class T, class U>
        auto
    hypot (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b.value);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
            + b.differentials * b.value / tmp
        };
    }
        template <std::size_t N, class T, class U>
        auto
    hypot (dual_t <N, T> const& a, U const& b)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
        };
    }
        template <std::size_t N, class T, class U>
        auto
    hypot (U const& a, dual_t <N, T> const& b)
    {
            using std::hypot;
            const auto
        tmp = hypot (a, b.value);
        return dual_t { 
              tmp
            , b.differentials * b.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (dual_t <N, T> const& a, dual_t <N, U> const& b, dual_t <N, V> const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b.value, c.value);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
            + b.differentials * b.value / tmp
            + c.differentials * c.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (dual_t <N, T> const& a, dual_t <N, U> const& b, V const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b.value, c);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
            + b.differentials * b.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (dual_t <N, T> const& a, U const& b, dual_t <N, V> const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b, c.value);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
            + c.differentials * c.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (T const& a, dual_t <N, U> const& b, dual_t <N, V> const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a, b.value, c.value);
        return dual_t { 
              tmp
            , b.differentials * b.value / tmp
            + c.differentials * c.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (T const& a, U const& b, dual_t <N, V> const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a, b, c.value);
        return dual_t { 
              tmp
            , c.differentials * c.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (dual_t <N, T> const& a, U const& b, V const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a.value, b, c);
        return dual_t { 
              tmp
            , a.differentials * a.value / tmp
        };
    }
        template <std::size_t N, class T, class U, class V>
        auto
    hypot (T const& a, dual_t <N, U> const& b, V const& c)
    {
            using std::hypot;
            const auto
        tmp = hypot (a, b.value, c);
        return dual_t { 
              tmp
            , b.differentials * b.value / tmp
        };
    }
        template <std::size_t N, class T, class U>
        auto
    pow (dual_t <N, T> const& a, dual_t <N, U> const& b)
    {
            using std::pow, std::log;
            const auto
        tmp = pow (a.value, b.value);
        return dual_t { 
              tmp
            , b.differentials * tmp * log (a.value)
            + a.differentials * pow (a.value, b.value - 1) * b.value
        };
    }
        template <std::size_t N, class T, class U>
        auto
    pow (dual_t <N, T> const& a, U const& b)
    {
            using std::pow, std::log;
            const auto
        tmp = pow (a.value, b);
        return dual_t { 
              tmp
            , a.differentials * pow (a.value, b - 1) * b
        };
    }
        template <std::size_t N, class T, class U>
        auto
    pow (T const& a, dual_t <N, U> const& b)
    {
            using std::pow, std::log;
            const auto
        tmp = pow (a, b.value);
        return dual_t { 
              tmp
            , b.differentials * tmp * log (a)
        };
    }
        template <std::size_t N, class T>
        auto
    sqrt (dual_t <N, T> const& a)
    {
            using std::sqrt, std::pow;
            const auto
        tmp = sqrt (a.value);
        return dual_t { 
              tmp
            , a.differentials / 2. / tmp
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
