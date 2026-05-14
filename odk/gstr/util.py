from fractions import Fraction


def get_numer_denom_str(num: float) -> str:
    frac = Fraction(num).limit_denominator()
    numer, denom = frac.numerator, frac.denominator

    return f'{numer}/{denom}'
