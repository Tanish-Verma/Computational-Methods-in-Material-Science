#time library
import time

# Get the current time in seconds since the epoch (Unix time, i.e., January 1, 1970)
current_time = time.time()
print(f"Current time (in seconds since epoch): {current_time}")

# Convert seconds since epoch to a human-readable format
local_time = time.ctime(current_time)
print(f"Human-readable time: {local_time}")  # Output: e.g., "Thu Oct 12 10:15:42 2023"

# Sleep for a given number of seconds (useful to delay execution)
print("Sleeping for 2 seconds...")
time.sleep(2)
print("Woke up after 2 seconds!")

# Get the current local time as a structured time object
local_time_struct = time.localtime()
print(f"Local time (struct_time): {local_time_struct}")

# Access specific components of the structured time object
year = local_time_struct.tm_year
month = local_time_struct.tm_mon
day = local_time_struct.tm_mday
print(f"Year: {year}, Month: {month}, Day: {day}")

# Get the current UTC time as a structured time object
utc_time_struct = time.gmtime()
print(f"UTC time (struct_time): {utc_time_struct}")

# Convert a structured time object to seconds since the epoch
seconds_since_epoch = time.mktime(local_time_struct)
print(f"Seconds since epoch (from local time struct): {seconds_since_epoch}")

# Format a structured time object as a string (strftime)
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)
print(f"Formatted local time: {formatted_time}")  # Output: e.g., "2023-10-12 10:15:42"

# Parse a string representing time to a structured time object (strptime)
time_str = "2023-10-12 10:15:42"
parsed_time_struct = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
print(f"Parsed time (struct_time): {parsed_time_struct}")

# Measure execution time using time.perf_counter() for high precision timing
start_time = time.perf_counter()
# Example task: sleeping for 1 second
time.sleep(1)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")  # Output: approximately 1 second

# Get processor time (time spent by the CPU on the current process)
cpu_time = time.process_time()
print(f"Processor time (in seconds): {cpu_time}")

#Math library
import math

# Basic mathematical constants
print(math.pi)     # Output: 3.141592653589793 (π)
print(math.e)      # Output: 2.718281828459045 (Euler's number, e)

# Rounding numbers
print(math.ceil(4.2))   # Output: 5 (rounds up)
print(math.floor(4.8))  # Output: 4 (rounds down)

# Absolute value
print(math.fabs(-3.5))  # Output: 3.5 (returns absolute value as float)

# Exponentiation and power
print(math.pow(2, 3))   # Output: 8.0 (2 raised to the power of 3)
print(math.exp(1))      # Output: 2.718281828459045 (e^1)

# Logarithms
print(math.log(10))           # Output: 2.302585092994046 (natural log, ln(10))
print(math.log10(100))        # Output: 2.0 (log base 10)
print(math.log2(8))           # Output: 3.0 (log base 2)

# Square root
print(math.sqrt(16))   # Output: 4.0

# Trigonometric functions
print(math.sin(math.pi / 2))  # Output: 1.0 (sin(90 degrees))
print(math.cos(math.pi))      # Output: -1.0 (cos(180 degrees))
print(math.tan(math.pi / 4))  # Output: 1.0 (tan(45 degrees))

# Inverse trigonometric functions
print(math.asin(1))   # Output: 1.5707963267948966 (arc sine of 1, in radians)
print(math.acos(0))   # Output: 1.5707963267948966 (arc cosine of 0, in radians)
print(math.atan(1))   # Output: 0.7853981633974483 (arc tangent of 1, in radians)

# Conversion between degrees and radians
print(math.degrees(math.pi))  # Output: 180.0 (convert radians to degrees)
print(math.radians(180))      # Output: 3.141592653589793 (convert degrees to radians)

# Hyperbolic functions
print(math.sinh(1))   # Output: 1.1752011936438014 (hyperbolic sine)
print(math.cosh(1))   # Output: 1.5430806348152437 (hyperbolic cosine)
print(math.tanh(1))   # Output: 0.7615941559557649 (hyperbolic tangent)

# Factorial of a number
print(math.factorial(5))  # Output: 120 (5!)

# Greatest Common Divisor (GCD)
print(math.gcd(24, 36))   # Output: 12 (GCD of 24 and 36)

# Least Common Multiple (LCM) (available in Python 3.9+)
print(math.lcm(24, 36))   # Output: 72 (LCM of 24 and 36)

# Copysign (returns x with the sign of y)
print(math.copysign(3, -1))  # Output: -3.0 (3 with the sign of -1)

# Checking if a number is NaN (Not a Number)
print(math.isnan(float('nan')))  # Output: True

# Checking if a number is finite or infinite
print(math.isfinite(1000))    # Output: True
print(math.isinf(float('inf')))  # Output: True (infinity)

# Remainder of division
print(math.fmod(5, 3))  # Output: 2.0 (5 % 3)

# Sum of an iterable (start value can be provided as the second argument)
print(math.fsum([0.1, 0.2, 0.3]))  # Output: 0.6 (more accurate floating-point sum)

# Euclidean distance between two points
print(math.dist([1, 2], [4, 6]))  # Output: 5.0 (distance between (1, 2) and (4, 6))

# Pythagorean theorem (hypotenuse)
print(math.hypot(3, 4))  # Output: 5.0 (hypotenuse of a right triangle with sides 3 and 4)

# Error functions
print(math.erf(1))    # Output: 0.8427007929497148 (error function)
print(math.erfc(1))   # Output: 0.15729920705028513 (complementary error function)

# Gamma function (generalization of factorial)
print(math.gamma(5))  # Output: 24.0 (gamma function of 5 is 4!, i.e., (5-1)!)

# Log-gamma function (logarithm of gamma function)
print(math.lgamma(5))  # Output: 3.178053830347945 (log-gamma of 5)

# Constants from math module
print(math.inf)  # Output: inf (positive infinity)
print(math.nan)  # Output: nan (not a number)

#random  module
import random

# Generate a random float between 0 and 1
print(random.random())  # Output: Random float between 0 and 1 (e.g., 0.374)

# Generate a random integer between two bounds (inclusive)
print(random.randint(1, 10))  # Output: Random integer between 1 and 10

# Generate a random float between two bounds
print(random.uniform(1, 10))  # Output: Random float between 1 and 10

# Randomly select an element from a sequence
fruits = ['apple', 'banana', 'cherry', 'date']
print(random.choice(fruits))  # Output: Randomly chosen element (e.g., 'banana')

# Randomly select multiple elements (without replacement)
print(random.sample(fruits, 2))  # Output: List of 2 random elements (e.g., ['apple', 'cherry'])

# Shuffle a list (modifies the list in place)
random.shuffle(fruits)
print(fruits)  # Output: Shuffled list (e.g., ['banana', 'date', 'apple', 'cherry'])

# Generate a random float from a normal distribution (mean=0, stddev=1)
print(random.gauss(0, 1))  # Output: Random number from normal distribution (e.g., 0.735)

# Generate a random float from a Gaussian distribution (mean, stddev)
print(random.normalvariate(0, 1))  # Output: Random number from normal distribution

# Generate a random float from an exponential distribution (lambda is rate)
print(random.expovariate(1))  # Output: Random number from an exponential distribution

# Generate a random float from a beta distribution
print(random.betavariate(2, 5))  # Output: Random number from a beta distribution (alpha=2, beta=5)

# Generate a random float from a gamma distribution
print(random.gammavariate(2, 2))  # Output: Random number from a gamma distribution (alpha=2, beta=2)

# Generate a random float from a log-normal distribution
print(random.lognormvariate(0, 1))  # Output: Random number from a log-normal distribution

# Generate a random float from a triangular distribution
print(random.triangular(0, 1, 0.5))  # Output: Random number from a triangular distribution (low, high, mode)

# Set a seed for reproducibility of random numbers
random.seed(42)
print(random.random())  # Output: 0.6394267984578837 (random number with seed 42)

# Generate random Boolean values
print(random.getrandbits(1))  # Output: 0 or 1 (1-bit random number)
print(bool(random.getrandbits(1)))  # Output: True or False

# Generate a random integer with n bits
print(random.getrandbits(8))  # Output: Random integer with 8 bits (e.g., 158)

# Choices with weighted probabilities
items = ['apple', 'banana', 'cherry']
weights = [0.1, 0.7, 0.2]
print(random.choices(items, weights, k=5))  # Output: List of 5 elements with weighted probabilities

# Uniformly distributed random angle in radians
print(random.uniform(0, math.pi))  # Output: Random float between 0 and π

# Generate a random permutation of a list (returns a new list)
print(random.sample(fruits, len(fruits)))  # Output: Random permutation of fruits

# Random floating-point number in a range (inclusive of low, exclusive of high)
print(random.uniform(1.5, 3.5))  # Output: Random float in [1.5, 3.5)

#Decimal library 

import decimal
from decimal import Decimal, getcontext, localcontext

# ----------------------- DECIMAL LIBRARY ADVANTAGES -----------------------
# 1. **High Precision**: Decimal allows you to specify the precision of floating-point arithmetic operations, making it ideal for financial calculations where exact decimal representation is crucial.
# 2. **No Floating-Point Errors**: Unlike `float`, `Decimal` eliminates common floating-point rounding errors, ensuring accurate calculations.
# 3. **Customizable Rounding**: You can specify different rounding modes like `ROUND_HALF_UP`, `ROUND_HALF_EVEN`, etc., to fit your specific use case.
# 4. **Context Management**: You can define global or local contexts (temporary settings) for precision, rounding, etc.
# 5. **Arithmetic Operations**: The module supports precise arithmetic operations, including addition, subtraction, multiplication, division, modulus, powers, square roots, etc.
# 6. **Special Values**: It can handle `NaN`, `Infinity`, and other special values that `float` supports, with better control over these values.

# ----------------------- DECIMAL LIBRARY DISADVANTAGES -----------------------
# 1. **Slower than float**: Decimal is slower than Python's built-in `float` because of its high precision and extra features.
# 2. **Memory Usage**: Decimal can consume more memory compared to `float`, especially with increased precision.
# 3. **More Complex**: Using `Decimal` requires additional code, especially when dealing with rounding, precision, and context settings. This can be overkill for applications that don’t require such high precision.
# 4. **Interoperability**: `Decimal` and `float` types cannot be mixed directly in operations without converting, so you must be cautious about type compatibility.

# ----------------------- DECIMAL CREATION -----------------------

# Creating a Decimal object using string input
d1 = Decimal('0.1')  # decimal object creation
d2 = Decimal('0.3')  # decimal object creation
print(d1, d2)  # Output: 0.1 0.3

# Creating a Decimal object from an integer or float (though float can lead to precision issues)
d3 = Decimal(3)  # decimal from integer
d4 = Decimal(1.5)  # decimal from float (float can cause precision loss)
print(d3, d4)  # Output: 3 1.5

# ----------------------- BASIC OPERATIONS -----------------------

# Addition of two Decimal numbers
d_sum = d1 + d2  # decimal addition
print(d_sum)  # Output: 0.4

# Subtraction of two Decimal numbers
d_diff = d3 - d1  # decimal subtraction
print(d_diff)  # Output: 2.9

# Multiplication of two Decimal numbers
d_prod = d1 * d2  # decimal multiplication
print(d_prod)  # Output: 0.03

# Division of two Decimal numbers
d_div = d3 / d2  # decimal division
print(d_div)  # Output: 10

# Power (raising to a power)
d_pow = d1 ** 2  # decimal power
print(d_pow)  # Output: 0.01

# Modulus operation (remainder of division)
d_mod = d3 % d2  # decimal modulus
print(d_mod)  # Output: 0.1

# ----------------------- ROUNDING -----------------------

# Rounding a Decimal number to a specified number of decimal places
d_rounded = d1.quantize(Decimal('0.00'))  # rounding decimal to 2 places
print(d_rounded)  # Output: 0.10

# Rounding up and down (ROUND_UP, ROUND_DOWN)
d_up = d1.quantize(Decimal('1'), rounding=decimal.ROUND_UP)  # rounding up
print(d_up)  # Output: 1

d_down = d1.quantize(Decimal('1'), rounding=decimal.ROUND_DOWN)  # rounding down
print(d_down)  # Output: 0

# ROUND_HALF_EVEN (default rounding mode, rounds to nearest even number)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_EVEN)  # rounding half even
print(d)  # Output: 1.2

# ROUND_HALF_UP (rounds away from zero when halfway)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_UP)  # rounding half up
print(d)  # Output: 1.3

# ROUND_HALF_DOWN (rounds towards zero when halfway)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_DOWN)  # rounding half down
print(d)  # Output: 1.2

# ----------------------- CONTEXT AND PRECISION -----------------------

# Getting the current context (precision, rounding mode, etc.)
context = getcontext()  # get decimal context
print(context)  # Output: Context settings (precision, rounding)

# Setting precision for decimal arithmetic
getcontext().prec = 10  # set precision to 10 digits
d_prec = Decimal('1.12345678901234') / Decimal('3')
print(d_prec)  # Output rounded to 10 significant digits

# Using localcontext() to temporarily set a precision for certain operations
with localcontext() as ctx:
    ctx.prec = 5  # set precision to 5 digits
    d_temp_prec = Decimal('1.23456789') / Decimal('3')
    print(d_temp_prec)  # Output with precision 5

# ----------------------- COMPARISON OPERATIONS -----------------------

# Comparison of two Decimal numbers
print(d1 == d2)  # equality comparison
print(d1 < d2)   # less-than comparison
print(d3 > d1)   # greater-than comparison

# Minimum and maximum between Decimal numbers
min_d = min(d1, d3)  # finding minimum
max_d = max(d1, d3)  # finding maximum
print(min_d, max_d)  # Output: 0.1 3

# Find maximum and minimum of two decimals
print(Decimal.max(d1, d3))  # decimal max
print(Decimal.min(d1, d3))  # decimal min

# ----------------------- MATH FUNCTIONS -----------------------

# Square root of a Decimal number
d_sqrt = d3.sqrt()  # decimal square root
print(d_sqrt)  # Output: 1.7320508075688772

# Natural logarithm (ln)
d_ln = d3.ln()  # natural logarithm of decimal
print(d_ln)  # Output: 1.0986122886681098

# Exponential (e^x)
d_exp = d3.exp()  # exponential of decimal
print(d_exp)  # Output: 20.085536923187668

# Logarithm to base 10
d_log10 = d3.log10()  # log base 10 of decimal
print(d_log10)  # Output: 0.47712125471966744

# ----------------------- ABSOLUTE VALUE AND SIGN -----------------------

# Getting the absolute value of a Decimal
d_abs = abs(Decimal('-2.5'))  # decimal absolute value
print(d_abs)  # Output: 2.5

# Getting the sign of a Decimal number (-1 for negative, 0 for zero, 1 for positive)
d_sign = Decimal('-2.5').compare(Decimal(0))  # compare with zero for sign
print(d_sign)  # Output: -1

# Copy the sign of one decimal to another
d3 = d1.copy_sign(d2)  # decimal copy sign
print(d3)  # Output: 1.25 (sign of d1 is copied from d2)

# ----------------------- SCALING AND SHIFTING -----------------------

# Shifting a decimal number by moving the decimal point
d_shift = d1.scaleb(2)  # move the decimal point 2 places right
print(d_shift)  # Output: 10

# Normalizing a Decimal number (removes trailing zeros)
d_norm = Decimal('5.00').normalize()  # decimal normalization
print(d_norm)  # Output: 5

# ----------------------- SPECIAL VALUES (NaN, Infinity) -----------------------

# Check if the decimal is NaN
d_nan = Decimal('NaN')
print(d_nan.is_nan())  # decimal is nan

# Check if the decimal is infinite
d_inf = Decimal('Infinity')
print(d_inf.is_infinite())  # decimal is infinite

# Check if the decimal is zero
d_zero = Decimal('0.0')
print(d_zero.is_zero())  # decimal is zero

# ----------------------- INTEGRAL FUNCTIONS -----------------------

# Rounds to the nearest integer without rounding
d = Decimal('5.5')
try:
    integral = d.to_integral_exact()  # decimal to integral exact
except decimal.InvalidOperation:
    print("Rounding required!")

# Rounds to the nearest integer using current rounding mode
d = Decimal('5.5')
integral = d.to_integral_value()  # decimal to integral value
print(integral)  # Output: 6

# ----------------------- OTHER SPECIALIZED FUNCTIONS -----------------------

# Perform a fused multiply-add (FMA) operation (x * y + z)
d1 = Decimal('1.5')
d2 = Decimal('2.0')
d3 = Decimal('0.5')
result = d1.fma(d2, d3)  # decimal fma
print(result)  # Output: 3.5

# Find the remainder closest to zero (modulus operation)
d1 = Decimal('10')
d2 = Decimal('3')
remainder = d1.remainder_near(d2)  # decimal remainder near
print(remainder)  # Output: 1

# Next floating point number towards positive infinity
d = Decimal('1.0')
print(d.next_plus())  # decimal next plus
print(d.next_minus())  # decimal next minus

import decimal
from decimal import Decimal, getcontext, localcontext

# ----------------------- ROUNDING MODES -----------------------

# ROUND_HALF_EVEN (default rounding mode, rounds to nearest even number)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_EVEN)  # rounding half even
print(d)  # Output: 1.2

# ROUND_HALF_UP (rounds away from zero when halfway)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_UP)  # rounding half up
print(d)  # Output: 1.3

# ROUND_HALF_DOWN (rounds towards zero when halfway)
d = Decimal('1.25').quantize(Decimal('1.0'), rounding=decimal.ROUND_HALF_DOWN)  # rounding half down
print(d)  # Output: 1.2

# ----------------------- MAX, MIN, AND COPY SIGN -----------------------

# Find maximum and minimum of two decimals
d1 = Decimal('1.25')
d2 = Decimal('2.5')
print(Decimal.max(d1, d2))  # decimal max
print(Decimal.min(d1, d2))  # decimal min

# Copy the sign of one decimal to another
d3 = d1.copy_sign(d2)  # decimal copy sign
print(d3)  # Output: 1.25 (sign of d1 is copied from d2)

# ----------------------- NEXT PLUS AND NEXT MINUS -----------------------

# Next floating point number towards positive infinity
d = Decimal('1.0')
print(d.next_plus())  # decimal next plus
print(d.next_minus())  # decimal next minus

# ----------------------- NAN, INFINITY, AND ZERO CHECKS -----------------------

# Check if the decimal is NaN
d_nan = Decimal('NaN')
print(d_nan.is_nan())  # decimal is nan

# Check if the decimal is infinite
d_inf = Decimal('Infinity')
print(d_inf.is_infinite())  # decimal is infinite

# Check if the decimal is zero
d_zero = Decimal('0.0')
print(d_zero.is_zero())  # decimal is zero

# ----------------------- MATH FUNCTIONS AND OPERATIONS -----------------------

# Perform a fused multiply-add (FMA) operation (x * y + z)
d1 = Decimal('1.5')
d2 = Decimal('2.0')
d3 = Decimal('0.5')
result = d1.fma(d2, d3)  # decimal fma
print(result)  # Output: 3.5

# Find the remainder closest to zero (modulus operation)
d1 = Decimal('10')
d2 = Decimal('3')
remainder = d1.remainder_near(d2)  # decimal remainder near
print(remainder)  # Output: 1

# ----------------------- INTEGRAL FUNCTIONS -----------------------

# Rounds to the nearest integer without rounding
d = Decimal('5.5')
try:
    integral = d.to_integral_exact()  # decimal to integral exact
except decimal.InvalidOperation:
    print("Rounding required!")

# Rounds to the nearest integer using current rounding mode
d = Decimal('5.5')
integral = d.to_integral_value()  # decimal to integral value
print(integral)  # Output: 6

# ----------------------- ABSOLUTE VALUE AND SIGN -----------------------

# Getting the absolute value of a Decimal
d_abs = abs(Decimal('-2.5'))  # decimal absolute value
print(d_abs)  # Output: 2.5

# Getting the sign of a Decimal number (-1 for negative, 0 for zero, 1 for positive)
d_sign = Decimal('-2.5').compare(Decimal(0))  # compare with zero for sign
print(d_sign)  # Output: -1

# ----------------------- SCALING AND SHIFTING -----------------------

# Shifting a decimal number by moving the decimal point
d_shift = d1.scaleb(2)  # move the decimal point 2 places right
print(d_shift)  # Output: 10

# Normalizing a Decimal number (removes trailing zeros)
d_norm = Decimal('5.00').normalize()  # decimal normalization
print(d_norm)  # Output: 5

# ----------------------- TUPLE REPRESENTATION -----------------------

# Getting the tuple representation (sign, digits, and exponent) of a Decimal
d_tuple = d1.as_tuple()  # decimal tuple representation
print(d_tuple)  # Output: DecimalTuple(sign=0, digits=(1,), exponent=-1)

# ----------------------- CONVERSIONS -----------------------

# Converting a Decimal to an integer
d_to_int = int(d3)  # decimal to integer conversion
print(d_to_int)  # Output: 3

# Converting a Decimal to a float (can lose precision)
d_to_float = float(d1)  # decimal to float conversion
print(d_to_float)  # Output: 0.1

from fractions import Fraction

# ----------------------- FRACTIONS LIBRARY ADVANTAGES -----------------------
# 1. **Exact Arithmetic**: Fraction allows exact representation of numbers as fractions (e.g., 1/3 is represented exactly, not as 0.333...).
# 2. **No Floating-Point Errors**: Fractions avoid the rounding errors that typically occur with floating-point arithmetic.
# 3. **Simple to Use**: Operations on fractions are easy to perform and maintain precision across arithmetic operations.
# 4. **Support for Various Inputs**: You can create fractions from integers, floats, strings, and even another Fraction.
# 5. **Auto-Simplification**: The Fraction class automatically simplifies fractions (e.g., 4/8 becomes 1/2).

# ----------------------- FRACTIONS LIBRARY DISADVANTAGES -----------------------
# 1. **Slower Performance**: Operations on `Fraction` are slower than on integers or floats due to the overhead of maintaining exact representations.
# 2. **Memory Usage**: Fractions can take up more memory than floats, especially for large numerators and denominators.
# 3. **Limited Use Cases**: For most everyday calculations where rounding errors are acceptable, floats may be more appropriate.
# 4. **Mixed-Type Arithmetic**: You must be cautious when mixing `Fraction` with `float` or other types, as explicit conversion may be needed.

# ----------------------- FRACTION CREATION -----------------------

# Creating a Fraction from integers (numerator/denominator)
f1 = Fraction(1, 3)  # fraction from integers
f2 = Fraction(5, 8)  # fraction from integers
print(f1, f2)  # Output: 1/3 5/8

# Creating a Fraction from a float (results in exact fraction representation)
f3 = Fraction(0.5)  # fraction from float (auto-converted to 1/2)
print(f3)  # Output: 1/2

# Creating a Fraction from a string (can handle repeating decimals and rational values)
f4 = Fraction('0.25')  # fraction from string (auto-converted to 1/4)
f5 = Fraction('1.75')  # fraction from string (auto-converted to 7/4)
print(f4, f5)  # Output: 1/4 7/4

# Creating a Fraction from another Fraction
f6 = Fraction(f1)  # fraction from another fraction (copy)
print(f6)  # Output: 1/3

# ----------------------- BASIC OPERATIONS -----------------------

# Addition of fractions
f_add = f1 + f2  # fraction addition
print(f_add)  # Output: 23/24 (1/3 + 5/8)

# Subtraction of fractions
f_sub = f2 - f1  # fraction subtraction
print(f_sub)  # Output: 7/24 (5/8 - 1/3)

# Multiplication of fractions
f_mul = f1 * f2  # fraction multiplication
print(f_mul)  # Output: 5/24 (1/3 * 5/8)

# Division of fractions
f_div = f1 / f2  # fraction division
print(f_div)  # Output: 8/15 (1/3 ÷ 5/8)

# Power (raising a fraction to a power)
f_pow = f1 ** 2  # fraction power
print(f_pow)  # Output: 1/9 (1/3 squared)

# Modulus operation (remainder of division)
f_mod = f2 % f1  # fraction modulus
print(f_mod)  # Output: 5/8 (since 5/8 < 1/3)

# ----------------------- FLOAT CONVERSION -----------------------

# Converting a Fraction to a float
float_val = float(f1)  # fraction to float
print(float_val)  # Output: 0.3333333333333333

# ----------------------- REDUCTION AND SIMPLIFICATION -----------------------

# Simplifying a Fraction (auto-simplified on creation)
f_simplified = Fraction(8, 16)  # auto-simplified to 1/2
print(f_simplified)  # Output: 1/2

# ----------------------- GCD, NUMERATOR, DENOMINATOR -----------------------

# Getting the numerator of a Fraction
numerator = f1.numerator  # fraction numerator
print(numerator)  # Output: 1 (numerator of 1/3)

# Getting the denominator of a Fraction
denominator = f1.denominator  # fraction denominator
print(denominator)  # Output: 3 (denominator of 1/3)

# ----------------------- COMPARISON OPERATIONS -----------------------

# Comparing fractions
print(f1 == f2)  # fraction equality comparison, Output: False
print(f1 > f2)   # fraction greater-than comparison, Output: False
print(f3 < f4)   # fraction less-than comparison, Output: False

# Minimum and maximum of two fractions
f_min = min(f1, f2)  # finding minimum of two fractions
f_max = max(f1, f2)  # finding maximum of two fractions
print(f_min, f_max)  # Output: 1/3 5/8

# ----------------------- MIXED-TYPE OPERATIONS -----------------------

# Adding a fraction and an integer (the integer is automatically converted to a fraction)
f_add_int = f1 + 1  # 1 is converted to 1/1, so 1/3 + 1 = 4/3
print(f_add_int)  # Output: 4/3

# Multiplying a fraction and a float (explicit conversion needed to avoid precision issues)
f_mul_float = f1 * Fraction(0.25)  # multiply fraction by a float
print(f_mul_float)  # Output: 1/12 (1/3 * 1/4)

# ----------------------- GCD (Greatest Common Divisor) -----------------------

# GCD of two fractions (it simplifies the fraction automatically)
f_gcd = Fraction(10, 20)  # GCD automatically simplifies this to 1/2
print(f_gcd)  # Output: 1/2

# ----------------------- LIMIT DENOMINATOR -----------------------

# Limit the denominator to a smaller value for an approximate fraction
f_lim_den = f3.limit_denominator(10)  # approximate fraction with max denominator of 10
print(f_lim_den)  # Output: 1/2 (since 0.5 = 1/2)

# Limit the denominator of an irrational number like π
import math
f_pi_approx = Fraction(math.pi).limit_denominator(100)  # approximate π as a fraction
print(f_pi_approx)  # Output: 311/99 (approximation of π)

# ----------------------- FRACTION OPERATIONS WITH CONTEXT -----------------------

# Fractions can also operate with `Decimal` for increased precision or context-sensitive operations
from decimal import Decimal

f_decimal_add = f1 + Fraction(Decimal('0.25'))  # adding fraction and decimal
print(f_decimal_add)  # Output: 7/12 (1/3 + 1/4)

# ----------------------- SIGN AND ABSOLUTE VALUE -----------------------

# Getting the sign of a fraction (-1, 0, or 1)
sign = f1.__pos__()  # positive sign of a fraction (this is just +1 for f1)
print(sign)  # Output: 1

# Absolute value of a fraction
f_abs = abs(Fraction(-5, 8))  # absolute value of a fraction
print(f_abs)  # Output: 5/8

# ------------------- FRACTION CREATION -------------------

# Creating fractions from float, Decimal, and integers
f1 = Fraction(1, 3)  # From integers
f2 = Fraction.from_float(0.75)  # From float
f3 = Fraction.from_decimal(Decimal('1.25'))  # From Decimal
print(f1, f2, f3)  # Output: 1/3 3/4 5/4

# ------------------- BASIC OPERATIONS -------------------

# Adding fractions and mixed types (integer, float, and Decimal)
result = f1 + 2  # Fraction + integer (auto converts integer)
result_decimal = f2 + Fraction(Decimal('0.75'))  # Fraction + Decimal
print(result, result_decimal)  # Output: 7/3, 3/2

# ------------------- FRACTION FROM FLOAT -------------------

# Exact representation of a float as a fraction
f4 = Fraction.from_float(1.1)  # Exact representation of float 1.1
print(f4)  # Output: 2476979795053773/2251799813685248

# ------------------- LIMIT DENOMINATOR -------------------

# Approximating irrational numbers like π as fractions
f_pi = Fraction(math.pi)
f_pi_approx = f_pi.limit_denominator(100)  # Approximate π
print(f_pi_approx)  # Output: 311/99

# ------------------- HASHING AND FRACTION SETS -------------------

# Using fractions as dictionary keys or set elements
f5 = Fraction(1, 2)
f6 = Fraction(3, 4)
fraction_set = {f5, f6}
print(fraction_set)  # Output: {Fraction(1, 2), Fraction(3, 4)}

# ------------------- NUMERATOR, DENOMINATOR, AND RATIO -------------------

# Getting numerator, denominator, and integer ratio of a fraction
f7 = Fraction(5, 8)
print(f7.numerator)  # Output: 5
print(f7.denominator)  # Output: 8
print(f7.as_integer_ratio())  # Output: (5, 8)

#Datetime library

import datetime


# Advantages of datetime Library:
# Versatile Date and Time Manipulation: Supports operations on dates, times, and both together.
# Built-in Time Zones: Can handle time zone–aware dates and times.
# Arithmetic Support: Can perform operations like adding or subtracting dates and times.
# Supports Parsing: Can parse strings into dates and format dates as strings.
# Disadvantages of datetime Library:
# No Built-in Leap Year Handling: Although datetime handles leap years correctly, it doesn’t automatically adjust for things like leap seconds.
# Limited Formatting for Custom Cases: It can sometimes be tedious to customize formats.
# Time Zones: Handling time zones, although possible, can become complex without external libraries like pytz.

# ----------------------- CURRENT DATE AND TIME -----------------------

# Get the current date and time
current_datetime = datetime.datetime.now()  # current date and time
print(current_datetime)  # Output: e.g., 2024-10-11 15:30:10.123456

# Get only the current date
current_date = datetime.date.today()  # current date
print(current_date)  # Output: e.g., 2024-10-11

# Get only the current time
current_time = datetime.datetime.now().time()  # current time
print(current_time)  # Output: e.g., 15:30:10.123456

# ----------------------- DATE OBJECTS -----------------------

# Create a specific date
custom_date = datetime.date(2023, 10, 12)  # year, month, day
print(custom_date)  # Output: 2023-10-12

# Get individual components of a date
year = custom_date.year  # year of the date
month = custom_date.month  # month of the date
day = custom_date.day  # day of the date
print(year, month, day)  # Output: 2023 10 12

# ----------------------- TIME OBJECTS -----------------------

# Create a specific time
custom_time = datetime.time(14, 30, 45)  # hours, minutes, seconds
print(custom_time)  # Output: 14:30:45

# Get individual components of a time
hour = custom_time.hour  # hour of the time
minute = custom_time.minute  # minute of the time
second = custom_time.second  # second of the time
print(hour, minute, second)  # Output: 14 30 45

# ----------------------- DATETIME OBJECTS -----------------------

# Create a specific datetime
custom_datetime = datetime.datetime(2023, 10, 12, 14, 30, 45)  # year, month, day, hour, minute, second
print(custom_datetime)  # Output: 2023-10-12 14:30:45

# Combine date and time into datetime
combined_datetime = datetime.datetime.combine(custom_date, custom_time)  # combine date and time
print(combined_datetime)  # Output: 2023-10-12 14:30:45

# ----------------------- ARITHMETIC OPERATIONS -----------------------

# Add or subtract time using timedelta
future_date = custom_date + datetime.timedelta(days=10)  # add 10 days
past_date = custom_date - datetime.timedelta(days=5)  # subtract 5 days
print(future_date, past_date)  # Output: 2023-10-22 2023-10-07

# Find the difference between two dates
date_diff = future_date - past_date  # difference between two dates
print(date_diff.days)  # Output: 15

# ----------------------- FORMATTING AND PARSING DATES -----------------------

# Format date and time as a string
formatted_datetime = custom_datetime.strftime("%Y-%m-%d %H:%M:%S")  # format datetime
print(formatted_datetime)  # Output: 2023-10-12 14:30:45

# Parse a string into a datetime object
parsed_datetime = datetime.datetime.strptime("2023-10-12 14:30:45", "%Y-%m-%d %H:%M:%S")  # parse string
print(parsed_datetime)  # Output: 2023-10-12 14:30:45

# ----------------------- TIME ZONES -----------------------

# Get the current time with timezone information (UTC)
utc_time = datetime.datetime.now(datetime.timezone.utc)  # current UTC time
print(utc_time)  # Output: e.g., 2024-10-11 15:30:10.123456+00:00


#MPMATH Library 

import mpmath

# The mpmath library is used for high-precision floating-point arithmetic and symbolic mathematics. It's highly useful for applications where you need precision beyond standard floating-point numbers.

# Advantages of mpmath:
# Arbitrary Precision: Allows calculations with arbitrary precision, making it ideal for scientific computing, cryptography, and more.
# Mathematical Functions: Supports advanced mathematical functions like trigonometry, logarithms, calculus, and complex numbers.
# Efficient for Big Numbers: Handles very large numbers and small numbers with high accuracy.
# Symbolic Computation: Supports symbolic evaluation of expressions.
# Disadvantages of mpmath:
# Slower than Standard Operations: Operations with mpmath are slower compared to native floating-point arithmetic.
# Increased Memory Usage: Due to arbitrary precision, it uses more memory than standard float types.
# Learning Curve: Some of the advanced functions and features may require deeper understanding of mathematical concepts.

# ----------------------- PRECISION CONTROL -----------------------

# Set precision for calculations (in decimal places)
mpmath.mp.dps = 50  # set precision to 50 decimal places

# ----------------------- BASIC OPERATIONS -----------------------

# Addition, subtraction, multiplication, division
a = mpmath.mpf('1.234567890123456789')  # creating a high-precision number
b = mpmath.mpf('3.456789012345678901')
print(a + b)  # Output: 4.69135690246913569 (precision maintained)
print(a - b)  # Output: -2.222221122222222112
print(a * b)  # Output: 4.2684362023456790123
print(a / b)  # Output: 0.35759519299372019117

# ----------------------- MATHEMATICAL FUNCTIONS -----------------------

# Square root
sqrt_val = mpmath.sqrt(2)  # high-precision square root of 2
print(sqrt_val)  # Output: 1.414213562373095048801688724209698078569671875376948

# Exponentiation
exp_val = mpmath.exp(1)  # high-precision value of e (Euler's number)
print(exp_val)  # Output: 2.71828182845904523536028747135266249775724709369995

# Logarithms (natural and base-10)
log_val = mpmath.log(10)  # natural log of 10
log10_val = mpmath.log10(1000)  # base-10 logarithm of 1000
print(log_val)  # Output: 2.302585092994045684017991454684364207601101488628772
print(log10_val)  # Output: 3.0

# ----------------------- TRIGONOMETRY -----------------------

# High-precision sine, cosine, and tangent
sin_val = mpmath.sin(mpmath.pi / 6)  # sin(π/6)
cos_val = mpmath.cos(mpmath.pi / 3)  # cos(π/3)
tan_val = mpmath.tan(mpmath.pi / 4)  # tan(π/4)
print(sin_val)  # Output: 0.5
print(cos_val)  # Output: 0.5
print(tan_val)  # Output: 1.0

# ----------------------- COMPLEX NUMBERS -----------------------

# Working with complex numbers
complex_num = mpmath.mpc('1.5', '2.5')  # complex number with real and imaginary parts
print(complex_num)  # Output: (1.5 + 2.5j)

# Complex exponential and logarithms
complex_exp = mpmath.exp(complex_num)  # e^(1.5 + 2.5j)
complex_log = mpmath.log(complex_num)  # natural log of (1.5 + 2.5j)
print(complex_exp)  # Output: (-3.04934515338309 + 0.833049961066805j)
