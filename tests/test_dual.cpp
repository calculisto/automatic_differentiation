// The type dual_t and its interface as a whole: construction, the types it
// can be made of, the operations that exist, constexpr and noexcept.
#include "test_tools.hpp"

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
    or requires (A const& a, B const& b) { lerp (a, a, b); };
static_assert (not has_any_operation <dual_t <1>, dual_t <2>>);
static_assert (not has_any_operation <dual_t <1>, std::complex <double>>);
static_assert (not has_any_operation <dual_t <1>, not_comparable>);
static_assert (not has_any_comparison <dual_t <1>, std::complex <double>>);

// The same for the comparison functions, except with the C runtime of
// Microsoft (UCRT), which declares them as templates of any two arguments,
// without constraints: they exist for any types, including these.
    template <class A, class B>
    concept
has_any_comparison_function = requires (A const& a, B const& b) { isgreater (a, b); }
    or requires (A const& a, B const& b) { isgreater (b, a); }
    or requires (A const& a, B const& b) { isunordered (a, b); }
    or requires (A const& a, B const& b) { isunordered (b, a); };
#ifndef _UCRT
static_assert (not has_any_comparison_function <dual_t <1>, dual_t <2>>);
static_assert (not has_any_comparison_function <dual_t <1>, std::complex <double>>);
static_assert (not has_any_comparison_function <dual_t <1>, not_comparable>);
static_assert (not has_any_comparison_function <dual_t <2, dual_t <1>>, dual_t <1>>);
#endif

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

TEST_CASE("dual_t")
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
}
