from fractions import Fraction

__all__ = [
    'fraction_str',
]


def fraction_str(num: float) -> str:
    frac = Fraction(num).limit_denominator()
    numer, denom = frac.numerator, frac.denominator

    return f'{numer}/{denom}'
