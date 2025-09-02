
#welcome 
#this is like a cheat sheet.
#it doesnt contain basics but you can find many things here do explore  and learn.  #python #python3 #pythonprogramming #pythonbasics #pythoncheats


#Arithmetic operators in python

#Addition +
a = 5 + 2
#Subtraction -
b = 5 - 2
#Multiplication *
c = 5 * 2
#Division /
d = 5 / 2
#Floor division //
e = 5 // 2
#Modulus %
f = 5 % 2
#Exponentiation **
g = 5 ** 2

print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, g={g}")

#Assignment operators
x = 2
x += 3  # Equivalent to x = x + 3
print(x)  # Output: 5

x -= 2  # Equivalent to x = x - 2
print(x)  # Output: 3

x *= 4  # Equivalent to x = x * 4
print(x)  # Output: 12

x /= 3  # Equivalent to x = x / 3
print(x)  # Output: 4.0

x **= 2  # Equivalent to x = x ** 2
print(x)  # Output: 16.0

x //= 5  # Equivalent to x = x // 5
print(x)  # Output: 3.0

x %= 2  # Equivalent to x = x % 2
print(x)  # Output: 1.0

type(x)

# Comparison operators
# > (greater than)
# < (less than)
# >= (greater than or equal)
# <= (less than or equal)
# == (equal)
# != (not equal)

# Logical expressions are statements that can either be true or false
# Returns a boolean "True" or "False"

a = 4 == 5
print(a)

b = 5 == 5
print(b)

c = 4 != 5
print(c)

d = 5 != 5
print(d)

# Logical Operators:

# and: Returns True if both statements are True.
print(True and True)   # Output: True
print(True and False)  # Output: False

# or: Returns True if at least one statement is True.
print(True or False)   # Output: True
print(False or False)  # Output: False

# not: Returns the opposite boolean value.
print(not True)    # Output: False
print(not False)   # Output: True

# Examples:

x = 5
y = 10

# Using 'and'
if x > 0 and y < 20:
  print("Both conditions are True")

# Using 'or'
if x < 3 or y > 5:
  print("At least one condition is True")

# Using 'not'
if not x == y:
  print("x is not equal to y")

#floating point error
# print(0.1+0.1+0.1==0.3) false
# print(0.1 + 0.2 == 0.3) false
# print(0.1 + 0.1 == 0.2) true

#Strings and their functions .

a="Hello this is a string" 
print(a)
k=len(a)#gives the lenght of strings
b=str(1233)#converts to a string ;
print(type(b))#type gives the type of variable
# 1. String Concatenation
a = "Hello"
b = "World"
c = a + " " + b  # Concatenates strings using the + operator
print(c)  # Output: Hello World

# 2. String Repetition
a = "Hello"
b = a * 3  # Repeats the string 3 times
print(b)  # Output: HelloHelloHello

# 3. String Indexing
a = "Hello"
print(a[0])  # Accesses the first character of the string (0-based indexing)
print(a[-1])  # Accesses the last character of the string (negative indexing)

# 4. String Slicing
a = "Hello World"
print(a[0:5])  # Slices the string from index 0 to 5 (exclusive)
print(a[6:])  # Slices the string from index 6 to the end
print(a[:5])  # Slices the string from the beginning to index 5 (exclusive)

## Indexing returns a single item,
## but slicing returns a subsequence of items. Thus while trying to index a nonexistent item,
## there is nothing to return and we get error.
## But when we slice a sequence outside of the bounds,
## we can return an empty sequence

# 5. String Length
a = "Hello World"
print(len(a))  # Returns the length of the string

# 6. String Conversion
a = 123
b = str(a)  # Converts an integer to a string
print(type(b))  # Output: <class 'str'>

# 7. String Formatting
a = "Hello {}"
b = a.format("World")  # Formats the string with a variable
print(b)  # Output: Hello World

# 8. String Upper Case
a = "hello"
b = a.upper()  # Converts the string to upper case
print(b)  # Output: HELLO

# 9. String Lower Case
a = "HELLO"
b = a.lower()  # Converts the string to lower case
print(b)  # Output: hello

# 10. String Strip
a = "   Hello World   "
b = a.strip()  # Removes leading and trailing whitespace
print(b)  # Output: Hello World

# 11. String LStrip
a = "   Hello World   "
b = a.lstrip()  # Removes leading whitespace
print(b)  # Output: Hello World  

# 12. String RStrip
a = "   Hello World   "
b = a.rstrip()  # Removes trailing whitespace
print(b)  # Output:    Hello World

# 13. String Split
a = "Hello,World,Python"
b = a.split(",")  # Splits the string into a list using a separator
print(b)  # Output: ['Hello', 'World', 'Python']

# 14. String Join
a = ["Hello", "World", "Python"]
b = ",".join(a)  # Joins a list of strings into a single string
print(b)  # Output: Hello,World,Python

# 15. String Find
a = "Hello World"
b = a.find("World")  # Finds the index of the first occurrence of a substring
print(b)  # Output: 6

# 16. String Index
a = "Hello World"
b = a.index("World")  # Finds the index of the first occurrence of a substring
print(b)  # Output: 6

# 17. String Count
a = "Hello World"
b = a.count("o")  # Counts the number of occurrences of a substring
print(b)  # Output: 2

# 18. String Replace
a = "Hello World"
b = a.replace("World", "Python")  # Replaces a substring with another string
print(b)  # Output: Hello Python

# 19. String Startswith
a = "Hello World"
b = a.startswith("Hello")  # Checks if the string starts with a substring
print(b)  # Output: True

# 20. String Endswith
a = "Hello World"
b = a.endswith("World")  # Checks if the string ends with a substring
print(b)  # Output: True

# 21. String Isalpha
a = "HelloWorld"
b = a.isalpha()  # Checks if the string contains only alphabetic characters
print(b)  # Output: True

# 22. String Isdigit
a = "12345"
b = a.isdigit()  # Checks if the string contains only digits
print(b)  # Output: True

# 23. String Isalnum
a = "Hello123"
b = a.isalnum()  # Checks if the string contains only alphanumeric characters
print(b)  # Output: True

# 24. String Islower
a = "hello"
b = a.islower()  # Checks if the string contains only lowercase characters
print(b)  # Output: True

# 25. String Isupper
a = "HELLO"
b = a.isupper()  # Checks if the string contains only uppercase characters
print(b)  # Output: True

# 26. String Istitle
a = "Hello World"
b = a.istitle()  # Checks if the string is title-cased
print(b)  # Output: True

# 27. String Isnumeric
a = "12345"
b = a.isnumeric()  # Checks if the string contains only numeric characters
print(b)  # Output: True

# 28. String Isdecimal
a = "12345"
b = a.isdecimal()  # Checks if the string contains only decimal characters
print(b)  # Output: True

# 29. String Isidentifier
a = "hello"
b = a.isidentifier()  # Checks if the string is a valid identifier
print(b)  # Output: True

# 30. String Isprintable
a = "Hello World"
b = a.isprintable()  # Checks if the string contains only printable characters
print(b)  # Output: True

# 31. String Isspace
a = "   "
b = a.isspace()  # Checks if the string contains only whitespace characters
print(b)  # Output: True

# 32. String Partition
a = "Hello World"
b = a.partition("World")  # Partitions the string at the first occurrence of a substring
print(b)  # Output: ('Hello ', 'World', '')

# 33. String Rfind
a = "Hello World"
b = a.rfind("World")  # Finds the index of the last occurrence of a substring
print(b)  # Output: 6

# 34. String Rindex
a = "Hello World"
b = a.rindex("World")  # Finds the index of the last occurrence of a substring
print(b)  # Output: 6

# 35. String Rpartition
a = "Hello World"
b = a.rpartition("World")  # Partitions the string at the last occurrence of a substring
print(b)  # Output: ('Hello ', 'World', '')

# 36. String Rsplit
a = "Hello,World,Python"
b = a.rsplit(",", 2)  # Splits the string into a list using a separator
print(b)  # Output: ['Hello', 'World', 'Python']

# 37. String Splitlines
a = "Hello\nWorld\nPython"
b = a.splitlines()  # Splits the string into a list at newline characters
print(b)  # Output: ['Hello', 'World', 'Python']

# 38. String Center
a = "Hello"
b = a.center(10)  # Centers the string in a string of length 10
print(b)  # Output: '   Hello  '

# 39. String Ljust
a = "Hello"
b = a.ljust(10)  # Left-justifies the string in a string of length 10
print(b)  # Output: 'Hello     '

# 40. String Rjust
a = "Hello"
b = a.rjust(10)  # Right-justifies the string in a string of length 10
print(b)  # Output: '     Hello'

# 41. String Zfill
a = "123"
b = a.zfill(5)  # Pads the string with zeros to a minimum width
print(b)  # Output: '00123'

# 42. String Translate
a = "Hello"
b = a.maketrans("H", "J")  # Creates a translation table
c = a.translate(b)  # Translates the string using the translation table
print(c)  # Output: 'Jello'

# 43. String Casefold
a = "HELLO"
b = a.casefold()  # Converts the string to casefolded string
print(b)  # Output: 'hello'

# 44. String Expandtabs
a = "Hello\tWorld"
b = a.expandtabs(8)  # Expands tabs in the string to spaces
print(b)  # Output: 'Hello     World'

# 45. String Encode
a = "Hello"
b = a.encode()  # Encodes the string to bytes
print(b)  # Output: b'Hello'

# 46. String Format
a = "Hello {}"
b = a.format("World")  # Formats the string with a variable
print(b)  # Output: Hello World

# 47. String Format_map
a = "Hello {name}"
b = a.format_map({"name": "World"}) # Formats the string with a mapping
print(b)  # Output: Hello World

# 48. String Capitalize
a = "hello world"
b = a.capitalize()  # Capitalizes the first character of the string
print(b)  # Output: Hello world

# 49. String Swapcase
a = "Hello World"
b = a.swapcase()  # Swaps the case of the string
print(b)  # Output: hELLO wORLD

# 50. String Title
a = "hello world"
b = a.title()  # Converts the string to title case
print(b)  # Output: Hello World

import string

# string.ascii_letters: A string containing all ASCII letters (both lowercase and uppercase)
print("ASCII Letters: ", string.ascii_letters)

# string.ascii_lowercase: A string containing all ASCII lowercase letters
print("ASCII Lowercase Letters: ", string.ascii_lowercase)

# string.ascii_uppercase: A string containing all ASCII uppercase letters
print("ASCII Uppercase Letters: ", string.ascii_uppercase)

# string.digits: A string containing all digit characters
print("Digits: ", string.digits)

# string.hexdigits: A string containing all hexadecimal digit characters
print("Hexadecimal Digits: ", string.hexdigits)

# string.octdigits: A string containing all octal digit characters
print("Octal Digits: ", string.octdigits)

# string.punctuation: A string containing all special characters
print("Punctuation: ", string.punctuation)

# string.printable: A string containing all printable characters
print("Printable Characters: ", string.printable)

# string.whitespace: A string containing all whitespace characters
print("Whitespace Characters: ", string.whitespace)

# Example usage:
# Check if a character is a letter
char = 'a'
if char in string.ascii_letters:
    print(char, "is a letter")

# Check if a character is a digit
char = '5'
if char in string.digits:
    print(char, "is a digit")

# Check if a character is a hexadecimal digit
char = 'A'
if char in string.hexdigits:
    print(char, "is a hexadecimal digit")

# Check if a character is a punctuation
char = '!'
if char in string.punctuation:
    print(char, "is a punctuation")

# Check if a character is a whitespace
char = ' '
if char in string.whitespace:
    print(char, "is a whitespace")


# LISTS
# A list is a collection of items that can be of any data type, including strings, integers
# Lists are denoted by square brackets [] and are ordered, meaning that items have a definite order
# Lists are mutable, meaning that they can be modified after creation
# Creating a list
# Initializing a list
my_list = [1, 2, 3, 4, 5]
print(my_list)  # Output: [1, 2, 3, 4, 5]

# Accessing elements
print(my_list[0])  # Access the first element, Output: 1
print(my_list[-1])  # Access the last element, Output: 5

# Slicing a list
print(my_list[1:3])  # Slice from index 1 to 3 (exclusive), Output: [2, 3]
print(my_list[:3])  # Slice from start to index 3 (exclusive), Output: [1, 2, 3]
print(my_list[2:])  # Slice from index 2 to end, Output: [3, 4, 5]
print(my_list[0::2])#alice from index 0 with step as 2 , Output: [1, 3, 5]

#if you slice a list you will get a list but when you access an element in the list you get that element not a list .
#so there is a type change when getting the number through any of the methods.

# Modifying elements
my_list[0] = 10  # Modify the first element
print(my_list)  # Output: [10, 2, 3, 4, 5]

# Appending elements
my_list.append(6)  # Add an element to the end
print(my_list)  # Output: [10, 2, 3, 4, 5, 6]

# Extending a list
my_list.extend([7, 8])  # Add multiple elements to the end
print(my_list)  # Output: [10, 2, 3, 4, 5, 6, 7, 8]

# Inserting elements
my_list.insert(1, 15)  # Insert 15 at index 1
print(my_list)  # Output: [10, 15, 2, 3, 4, 5, 6, 7, 8]

# Removing elements
my_list.remove(15)  # Remove the first occurrence of 15
print(my_list)  # Output: [10, 2, 3, 4, 5, 6, 7, 8]

# Popping elements
popped_element = my_list.pop()  # Remove and return the last element
print(popped_element)  # Output: 8
print(my_list)  # Output: [10, 2, 3, 4, 5, 6, 7]

popped_element = my_list.pop(0)  # Remove and return the element at index 0
print(popped_element)  # Output: 10
print(my_list)  # Output: [2, 3, 4, 5, 6, 7]

# Finding index of an element
index = my_list.index(4)  # Get the index of the first occurrence of 4
print(index)  # Output: 2

# Counting occurrences
count = my_list.count(3)  # Count occurrences of 3
print(count)  # Output: 1

# Sorting a list
my_list.sort()  # Sort the list in ascending order
print(my_list)  # Output: [2, 3, 4, 5, 6, 7]

my_list.sort(reverse=True)  # Sort the list in descending order
print(my_list)  # Output: [7, 6, 5, 4, 3, 2]

# Reversing a list
my_list.reverse()  # Reverse the elements of the list
print(my_list)  # Output: [2, 3, 4, 5, 6, 7]

# Copying a list
my_list_copy = my_list.copy()  # Create a shallow copy of the list
print(my_list_copy)  # Output: [2, 3, 4, 5, 6, 7]

# Clearing a list
my_list.clear()  # Remove all elements from the list
print(my_list)  # Output: []

# List comprehension
squared = [x**2 for x in my_list_copy]  # Create a new list with squares of elements
print(squared)  # Output: [4, 9, 16, 25, 36, 49]

# Checking for membership
print(5 in my_list_copy)  # Check if 5 is in the list, Output: True
print(10 not in my_list_copy)  # Check if 10 is not in the list, Output: True

# Concatenating lists
concatenated_list = my_list_copy + [8, 9]
print(concatenated_list)  # Output: [2, 3, 4, 5, 6, 7, 8, 9]

# Multiplying a list
multiplied_list = my_list_copy * 2
print(multiplied_list)  # Output: [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7]

# Finding the minimum and maximum elements
print(min(my_list_copy))  # Output: 2
print(max(my_list_copy))  # Output: 7

# Summing all elements in a list
print(sum(my_list_copy))  # Output: 27

# Finding the length of a list
print(len(my_list_copy))  # Output: 6

# Enumerating a list (get index and value)
for index, value in enumerate(my_list_copy):
    print(f"Index {index}, Value {value}")
# Output:
# Index 0, Value 2
# Index 1, Value 3
# Index 2, Value 4
# Index 3, Value 5
# Index 4, Value 6
# Index 5, Value 7

# Sorting using a custom key (sort by absolute value)
my_list_unsorted = [-3, 1, 4, -1, -5]
my_list_unsorted.sort(key=abs)
print(my_list_unsorted)  # Output: [1, -1, -3, 4, -5]

# Sorting without modifying the original list
sorted_list = sorted(my_list_copy)
print(sorted_list)  # Output: [2, 3, 4, 5, 6, 7]
print(my_list_copy)  # Original list remains unchanged

# Iterating over a list in reverse
for item in reversed(my_list_copy):
    print(item)
# Output: 7, 6, 5, 4, 3, 2

# Deleting elements by index
del my_list_copy[0]  # Delete the first element
print(my_list_copy)  # Output: [3, 4, 5, 6, 7]

del my_list_copy[1:3]  # Delete a slice (from index 1 to 2)
print(my_list_copy)  # Output: [3, 6, 7]

# Nested lists
nested_list = [[1, 2], [3, 4], [5, 6]]
print(nested_list[1][0])  # Accessing an element in a nested list, Output: 3

# Flattening a nested list
flat_list = [item for sublist in nested_list for item in sublist]
print(flat_list)  # Output: [1, 2, 3, 4, 5, 6]

# Using 'any()' and 'all()' on lists
print(any([0, 1, 2]))  # Output: True (at least one element is True)
print(all([1, 2, 3]))  # Output: True (all elements are True)
print(all([0, 1, 2]))  # Output: False (0 is considered False)

# Zipping two lists
list_a = [1, 2, 3]
list_b = ['a', 'b', 'c']
zipped = list(zip(list_a, list_b))
print(zipped)  # Output: [(1, 'a'), (2, 'b'), (3, 'c')]

# Unpacking a list into variables
first, *middle, last = [2, 3, 4, 5, 6, 7]
print(first)  # Output: 2
print(middle)  # Output: [3, 4, 5, 6]
print(last)   # Output: 7

#can be used to swap variable  values
a=5
b=10
a,b=b,a
print(a)  # Output: 10

#Packing
a = 10; b=20; c=(0,1,2); d=(3,4,5,6);
z = a,b,c,d
print(z)   # Output: (10, 20, (0, 1, 2), (3,4,5,6))
type(z)   # Output: tuple


# Multiplying corresponding elements of two lists (element-wise multiplication)
list_x = [1, 2, 3]
list_y = [4, 5, 6]
multiplied = [a * b for a, b in zip(list_x, list_y)]
print(multiplied)  # Output: [4, 10, 18]

# Using filter() to filter elements from a list
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4, 6]

# Using map() to transform elements of a list
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # Output: [2, 4, 6, 8, 10, 12]

# Using reduce() to reduce a list to a single value
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 720

# Using itertools combinations() to get combinations of list elements
from itertools import combinations
comb = list(combinations([1, 2, 3], 2))
print(comb)  # Output: [(1, 2), (1, 3), (2, 3)]

# Using itertools permutations() to get permutations of list elements
from itertools import permutations
perms = list(permutations([1, 2, 3]))
print(perms)  # Output: [(1, 2, 3), (1, 3, 2), (2, 1, 3), ...]

# Using zip to transpose a matrix (list of lists)
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = list(zip(*matrix))
print(transposed)  # Output: [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

from itertools import permutations

# Using permutations with a list
my_list = [1, 2, 3]
perm_set = set(permutations(my_list, 2))

print(perm_set)  # Output: {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}
#can be used for all datatypes which are iterable

# Using chain() from itertools to combine multiple lists
from itertools import chain
list1 = [1, 2]
list2 = [3, 4]
combined = list(chain(list1, list2))
print(combined)  # Output: [1, 2, 3, 4]

#counting the repitions in a list
from collections import Counter
multi_set = Counter([1, 1, 2, 2, 2, 3])
print(multi_set)  # Output: Counter({2: 3, 1: 2, 3: 1})

# Using deque from collections as a double-ended queue
from collections import deque
d = deque([1, 2, 3])
d.appendleft(0)  # Efficient O(1) operation at the start
print(d)  # Output: deque([0, 1, 2, 3])

# Using heapq to manage a priority queue (min-heap)
import heapq
heap = [3, 2, 1]
heapq.heapify(heap)  # Transform into a heap
heapq.heappush(heap, 0)
print(heap)  # Output: [0, 2, 1, 3]

# Removing all occurrences of an element using list comprehension
my_list = [1, 2, 2, 3, 2, 4]
my_list = [x for x in my_list if x != 2]
print(my_list)  # Output: [1, 3, 4]

# Using list as a stack
stack = []
stack.append(10)  # Push onto stack
stack.append(20)
print(stack.pop())  # Pop from stack, Output: 20
print(stack)  # Output: [10]

# Using list as a queue (inefficient, consider deque)
queue = []
queue.append(10)  # Enqueue
queue.append(20)
print(queue.pop(0))  # Dequeue, Output: 10
print(queue)  # Output: [20]

# Warning example about list references in Python

# Original list of prime numbers
prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Assigning n2 to reference the same list as prime_numbers
n2 = prime_numbers

# This shows the initial state of both lists
print("Before modification:")
print("prime_numbers:", prime_numbers)  # Output: original list
print("n2:", n2)                          # Output: original list

# Modifying n2 (which is a reference to prime_numbers)
n2[0] = 99  # Changing the first element to 99

# Now let's see the effect of the modification
print("\nAfter modification:")
print("prime_numbers:", prime_numbers)  # Output: [99, 3, 5, 7, 11, 13, 17, 19, 23, 29]
print("n2:", n2)                          # Output: [99, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Warning message about mutable objects
print("\nWARNING: Both 'n2' and 'prime_numbers' reference the same list.")
print("Modifying 'n2' affects 'prime_numbers' since they point to the same memory location.")

# To avoid this issue, use the copy method
n1 = prime_numbers.copy()  # Create a separate copy

# Now let's see the effect of modifying n1
n1[1] = 88  # Changing the second element to 88

# Checking the values after modification
print("\nAfter modifying n1:")
print("prime_numbers:", prime_numbers)  # Should remain unchanged
print("n1:", n1)                          # Output: [99, 88, 5, 7, 11, 13, 17, 19, 23, 29]

#Tuples
# Tuples are immutable, meaning their contents cannot be modified after creation.
# They are defined by enclosing elements in parentheses, ().
# Tuples are faster and more memory-efficient than lists, especially for large datasets.
# Initializing a tuple
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple)  # Output: (1, 2, 3, 4, 5)

# Accessing elements in a tuple
print(my_tuple[0])  # Access the first element, Output: 1
print(my_tuple[-1])  # Access the last element, Output: 5

# Slicing a tuple
print(my_tuple[1:3])  # Slice from index 1 to 3 (exclusive), Output: (2, 3)
print(my_tuple[:3])  # Slice from start to index 3 (exclusive), Output: (1, 2, 3)

# Concatenating tuples
concat_tuple = my_tuple + (6, 7)
print(concat_tuple)  # Output: (1, 2, 3, 4, 5, 6, 7)

# Repeating a tuple
repeated_tuple = my_tuple * 2
print(repeated_tuple)  # Output: (1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

# Checking for membership
print(3 in my_tuple)  # Check if 3 is in the tuple, Output: True

# Finding the length of a tuple
print(len(my_tuple))  # Output: 5

# Counting occurrences of an element
count = my_tuple.count(3)  # Count occurrences of 3
print(count)  # Output: 1

# Finding the index of an element
index = my_tuple.index(4)  # Get the index of the first occurrence of 4
print(index)  # Output: 3

# Tuple unpacking (destructuring)
a, b, c, d, e = my_tuple
print(a, b, c, d, e)  # Output: 1 2 3 4 5

# Extended unpacking
first, *middle, last = my_tuple
print(first)   # Output: 1
print(middle)  # Output: [2, 3, 4] (middle is a list)
print(last)    # Output: 5

# Nested tuple access
nested_tuple = (1, (2, 3), (4, (5, 6)))
print(nested_tuple[1])  # Output: (2, 3)
print(nested_tuple[2][1][0])  # Accessing nested element, Output: 5

# Converting tuple to a list (to perform mutable operations)
tuple_to_list = list(my_tuple)
tuple_to_list.append(6)
print(tuple_to_list)  # Output: [1, 2, 3, 4, 5, 6]

# Converting a list back to a tuple
list_to_tuple = tuple(tuple_to_list)
print(list_to_tuple)  # Output: (1, 2, 3, 4, 5, 6)

# Sorting elements of a tuple (requires conversion to a list)
sorted_tuple = tuple(sorted(my_tuple))
print(sorted_tuple)  # Output: (1, 2, 3, 4, 5)

# Reversing elements of a tuple (requires conversion to a list)
reversed_tuple = tuple(reversed(my_tuple))
print(reversed_tuple)  # Output: (5, 4, 3, 2, 1)

# Using 'any()' and 'all()' on tuples
print(any((0, 1, 2)))  # Output: True (at least one element is True)
print(all((1, 2, 3)))  # Output: True (all elements are True)

# Zipping two tuples (creates pairs)
tuple_a = (1, 2, 3)
tuple_b = ('a', 'b', 'c')
zipped_tuples = tuple(zip(tuple_a, tuple_b))
print(zipped_tuples)  # Output: ((1, 'a'), (2, 'b'), (3, 'c'))

# Finding the minimum and maximum elements in a tuple
print(min(my_tuple))  # Output: 1
print(max(my_tuple))  # Output: 5

# Summing elements of a tuple
print(sum(my_tuple))  # Output: 15

# Creating a tuple from a generator expression
generator = (x**2 for x in range(5))
generated_tuple = tuple(generator)
print(generated_tuple)  # Output: (0, 1, 4, 9, 16)

# Tuple with one element (requires a trailing comma)
singleton_tuple = (1,)
print(singleton_tuple)  # Output: (1,)

# Empty tuple
empty_tuple = ()
print(empty_tuple)  # Output: ()

# Using itertools to create combinations from tuples
from itertools import combinations
tuple_combinations = list(combinations(my_tuple, 2))
print(tuple_combinations)  # Output: [(1, 2), (1, 3), (1, 4), ...]

# Using itertools to create permutations from tuples
from itertools import permutations
tuple_permutations = list(permutations(my_tuple))
print(tuple_permutations)  # Output: [(1, 2, 3, 4, 5), ...]

# Using zip() to transpose a tuple (of tuples)
matrix_tuple = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
transposed_tuple = tuple(zip(*matrix_tuple))
print(transposed_tuple)  # Output: ((1, 4, 7), (2, 5, 8), (3, 6, 9))

# Tuples as keys in dictionaries (since tuples are hashable and immutable)
my_dict = {(1, 2): 'a', (3, 4): 'b'}
print(my_dict[(1, 2)])  # Output: 'a'

# Creating a tuple from a string
string_tuple = tuple("hello")
print(string_tuple)  # Output: ('h', 'e', 'l', 'l', 'o')

# Namedtuple example (from collections)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(2, 3)
print(p.x, p.y)  # Output: 2 3

# Using tuple() constructor to create an empty tuple
empty_tuple = tuple()
print(empty_tuple)  # Output: ()

# Function returning multiple values using a tuple
def min_max(numbers):
    return min(numbers), max(numbers)

result = min_max([3, 5, 7, 2, 8])
print(result)  # Output: (2, 8)

# Tuple unpacking from function return values
min_val, max_val = min_max([3, 5, 7, 2, 8])
print(min_val, max_val)  # Output: 2 8

# Comparing tuples lexicographically
tuple1 = (1, 2, 3)
tuple2 = (1, 2, 4)
print(tuple1 < tuple2)  # Output: True (compares element by element)

# Hashing tuples and using them as dictionary keys
my_dict = {('a', 'b'): 10, ('c', 'd'): 20}
print(my_dict[('a', 'b')])  # Output: 10

# Tuple packing and unpacking
packed_tuple = 1, 2, 3  # Packing
print(packed_tuple)  # Output: (1, 2, 3)

a, b, c = packed_tuple  # Unpacking
print(a, b, c)  # Output: 1 2 3

# Nested tuple unpacking
nested_tuple = (1, (2, 3), 4)
a, (b, c), d = nested_tuple
print(a, b, c, d)  # Output: 1 2 3 4

# Memory efficiency comparison (tuple vs. list)
import sys
my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3, 4, 5)
print(sys.getsizeof(my_list))   # Output: Memory size of list
print(sys.getsizeof(my_tuple))  # Output: Memory size of tuple (less than list)

# Enumerating over a tuple (index and value)
my_tuple = ('a', 'b', 'c')
for index, value in enumerate(my_tuple):
    print(f"Index {index}, Value {value}")
# Output:
# Index 0, Value a
# Index 1, Value b
# Index 2, Value c

# Tuples with *args in functions
def sum_all(*args):
    return sum(args)

result = sum_all(1, 2, 3, 4)
print(result)  # Output: 10


#Dictionary
# Initializing a dictionary
my_dict = {'name': 'John', 'age': 25, 'city': 'New York'}
print(my_dict)  # Output: {'name': 'John', 'age': 25, 'city': 'New York'}
# Accessing elements
print(my_dict['name'])  # Access value by key, Output: 'John'

# Adding new key-value pairs
my_dict['email'] = 'john@example.com'
print(my_dict)  # Output: {'name': 'John', 'age': 25, 'city': 'New York', 'email': 'john@example.com'}

# Modifying values
my_dict['age'] = 26  # Update value for the key 'age'
print(my_dict)  # Output: {'name': 'John', 'age': 26, 'city': 'New York', 'email': 'john@example.com'}

# Removing elements
del my_dict['email']  # Remove a key-value pair
print(my_dict)  # Output: {'name': 'John', 'age': 26, 'city': 'New York'}

# Using pop() to remove a key and return its value
age = my_dict.pop('age')
print(age)  # Output: 26
print(my_dict)  # Output: {'name': 'John', 'city': 'New York'}

# popitem() removes and returns the last inserted key-value pair (in Python 3.7+)
last_item = my_dict.popitem()
print(last_item)  # Output: ('city', 'New York')
print(my_dict)  # Output: {'name': 'John'}


# Create a new dictionary using the dict() constructor
my_dict = dict([('a', 1), ('b', 2), ('c', 3)])

# Get the number of key-value pairs in the dictionary
print(len(my_dict))  # Output: 3

# Check if a key is present in the dictionary
print('a' in my_dict)  # Output: True
print('d' in my_dict)  # Output: False

# Check if a key is not present in the dictionary
print('a' not in my_dict)  # Output: False
print('d' not in my_dict)  # Output: True

# Retrieve a value from the dictionary with a default value
print(my_dict.get('a', 0))  # Output: 1
print(my_dict.get('d', 0))  # Output: 0

# Set a value for a key if it is not already present in the dictionary
my_dict.setdefault('d', 4)
print(my_dict)  # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Update the dictionary with an iterable of key-value pairs
my_dict.update([('e', 5), ('f', 6)])
print(my_dict)  # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}

# Clearing a dictionary (removing all key-value pairs)
my_dict.clear()
print(my_dict)  # Output: {}

# Reinitializing a dictionary
my_dict = {'name': 'Alice', 'age': 30, 'city': 'Paris'}

# Accessing with get() (returns None if the key doesn't exist, avoids KeyError)
email = my_dict.get('email', 'Not Available')  # Default value if key is missing
print(email)  # Output: 'Not Available'

# Checking for a key in a dictionary
print('name' in my_dict)  # Output: True
print('email' in my_dict)  # Output: False

# Iterating over a dictionary
# Iterating over keys
for key in my_dict:
    print(key)
# Output:
# name
# age
# city

# Iterating over values
for value in my_dict.values():
    print(value)
# Output:
# Alice
# 30
# Paris

# Iterating over key-value pairs
for key, value in my_dict.items():
    print(f'{key}: {value}')
# Output:
# name: Alice
# age: 30
# city: Paris

# Dictionary comprehension (creating a new dictionary from an iterable)
squared_numbers = {x: x**2 for x in range(5)}
print(squared_numbers)  # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Merging two dictionaries using update()
my_dict.update({'email': 'alice@example.com', 'country': 'France'})
print(my_dict)  # Output: {'name': 'Alice', 'age': 30, 'city': 'Paris', 'email': 'alice@example.com', 'country': 'France'}

# Merging dictionaries using dictionary unpacking (Python 3.5+)
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged_dict = {**dict1, **dict2}
print(merged_dict)  # Output: {'a': 1, 'b': 3, 'c': 4}

# Merging dictionaries with dict union operator (Python 3.9+)
merged_dict = dict1 | dict2
print(merged_dict)  # Output: {'a': 1, 'b': 3, 'c': 4}

# Dictionary with tuple keys (tuples are hashable and can be used as keys)
tuple_key_dict = {('x', 'y'): 10, ('a', 'b'): 20}
print(tuple_key_dict[('x', 'y')])  # Output: 10

# Creating a dictionary from keys with a default value using fromkeys()
keys = ['name', 'age', 'city']
default_dict = dict.fromkeys(keys, 'unknown')
print(default_dict)  # Output: {'name': 'unknown', 'age': 'unknown', 'city': 'unknown'}

# Using setdefault() to get a value and set it if the key is missing
my_dict.setdefault('gender', 'Female')
print(my_dict)  # Output: {'name': 'Alice', 'age': 30, 'city': 'Paris', 'email': 'alice@example.com', 'country': 'France', 'gender': 'Female'}

# Sorting dictionary by keys
sorted_by_keys = dict(sorted(my_dict.items()))
print(sorted_by_keys)  # Output: {'age': 30, 'city': 'Paris', 'country': 'France', 'email': 'alice@example.com', 'gender': 'Female', 'name': 'Alice'}

# Sorting dictionary by values
sorted_by_values = dict(sorted(my_dict.items(), key=lambda item: item[1]))
print(sorted_by_values)  # Output: {'age': 30, 'city': 'Paris', 'name': 'Alice', 'country': 'France', 'email': 'alice@example.com', 'gender': 'Female'}

# Copying a dictionary (shallow copy)
copy_dict = my_dict.copy()
print(copy_dict)  # Output: {'name': 'Alice', 'age': 30, 'city': 'Paris', 'email': 'alice@example.com', 'country': 'France', 'gender': 'Female'}

# Nested dictionaries
nested_dict = {
    'person1': {'name': 'John', 'age': 25},
    'person2': {'name': 'Alice', 'age': 30}
}
print(nested_dict['person1']['name'])  # Output: 'John'

# Accessing deeply nested keys using get() to avoid KeyError
print(nested_dict.get('person3', {}).get('name', 'Not Found'))  # Output: 'Not Found'

# Handling missing keys with defaultdict (from collections)
from collections import defaultdict
dd = defaultdict(int)  # int sets a default of 0 for missing keys
dd['x'] += 1
print(dd['x'])  # Output: 1
print(dd['y'])  # Output: 0 (default value provided by int)

# Counting items using defaultdict
from collections import defaultdict
count_dict = defaultdict(int)
for char in 'hello world':
    count_dict[char] += 1
print(count_dict)  # Output: defaultdict(<class 'int'>, {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# Dictionary with multiple values per key using defaultdict(list)
multi_value_dict = defaultdict(list)
multi_value_dict['fruits'].append('apple')
multi_value_dict['fruits'].append('banana')
print(multi_value_dict)  # Output: defaultdict(<class 'list'>, {'fruits': ['apple', 'banana']})

# Using ChainMap to combine multiple dictionaries (from collections)
from collections import ChainMap
dict_a = {'a': 1, 'b': 2}
dict_b = {'b': 3, 'c': 4}
combined = ChainMap(dict_a, dict_b)  # ChainMap searches from dict_a to dict_b
print(combined['b'])  # Output: 2 (takes value from the first dictionary)
print(combined['c'])  # Output: 4

# Dictionary memory size comparison (dict vs. list)
import sys
print(sys.getsizeof(my_dict))  # Output: Memory size of the dictionary
print(sys.getsizeof(['Alice', 30, 'Paris', 'alice@example.com', 'France', 'Female']))  # Output: Memory size of an equivalent list

# Enumerating through dictionary items
for index, (key, value) in enumerate(my_dict.items()):
    print(f"Index {index}: {key} = {value}")
# Output:
# Index 0: name = Alice
# Index 1: age = 30
# Index 2: city = Paris
# Index 3: email = alice@example.com
# Index 4: country = France
# Index 5: gender = Female

# Dictionary keys, values, and items (view objects)
print(my_dict.keys())    # Output: dict_keys(['name', 'age', 'city', 'email', 'country', 'gender'])
print(my_dict.values())  # Output: dict_values(['Alice', 30, 'Paris', 'alice@example.com', 'France', 'Female'])
print(my_dict.items())   # Output: dict_items([('name', 'Alice'), ('age', 30), ...])


# Sets
# A set is an unordered collection of unique elements.
# Sets are used to store multiple items in a single variable.
# They are mutable, meaning that they can be changed after creation.
# They do not allow duplicate values.

# Initializing a set
my_set = {1, 2, 3, 4, 5}
print(my_set)  # Output: {1, 2, 3, 4, 5}

# Creating a set from a list
my_list = [1, 2, 2, 3, 4, 4, 5]
my_set = set(my_list)
print(my_set)  # Output: {1, 2, 3, 4, 5}

# Creating a set from a string
my_string = "hello"
my_set = set(my_string)
print(my_set)  # Output: {'h', 'e', 'l', 'o'}

# Accessing elements in a set
print(2 in my_set)  # Output: True
print(6 in my_set)  # Output: False

# Adding elements to a set
my_set.add(6)
print(my_set)  # Output: {1, 2, 3, 4, 5, 6}

# Adding multiple elements to a set
my_set.update([7, 8, 9])
print(my_set)  # Output: {1, 2, 3, 4, 5, 6, 7, 8, 9}

# Removing elements from a set
my_set.remove(6)
print(my_set)  # Output: {1, 2, 3, 4, 5, 7, 8, 9}

# Removing multiple elements from a set
my_set.discard(7)
my_set.discard(8)
print(my_set)  # Output: {1, 2, 3, 4, 5, 9}

# Removing an arbitrary element from a set
removed_element = my_set.pop()
print(removed_element)  # Output: 9 (the actual output may vary)
print(my_set)  # Output: {1, 2, 3, 4, 5}

# Clearing a set
my_set.clear()
print(my_set)  # Output: set()

# Checking the length of a set
my_set = {1, 2, 3, 4, 5}
print(len(my_set))  # Output: 5

# Creating an empty set
empty_set = set()
print(empty_set)  # Output: set()

# Checking if a set is a subset of another set
set_a = {1, 2, 3}
set_b = {1, 2, 3, 4, 5}
print(set_a.issubset(set_b))  # Output: True

# Checking if a set is a superset of another set
print(set_b.issuperset(set_a))  # Output: True

# Checking if two sets are disjoint (have no elements in common)
set_c = {6, 7, 8}
print(set_a.isdisjoint(set_c))  # Output: True

# Finding the union of two sets
set_d = {4, 5, 6}
print(set_a.union(set_d))  # Output: {1, 2, 3, 4, 5, 6}

# Finding the intersection of two sets
print(set_a.intersection(set_d))  # Output: set() (empty set, since there are no common elements)

# Finding the difference of two sets
print(set_a.difference(set_d))  # Output: {1, 2, 3}

# Finding the symmetric difference of two sets
print(set_a.symmetric_difference(set_d))  # Output: {1, 2, 3, 6}

# Using set comprehension to create a new set
squared_numbers = {x**2 for x in range(5)}
print(squared_numbers)  # Output: {0, 1, 4, 9, 16}

# Filtering with set comprehension
even_numbers = {x for x in range(10) if x % 2 == 0}
print(even_numbers)  # Output: {0, 2, 4, 6, 8}

# Using set operations to find common elements between multiple sets
set_e = {1, 2, 3, 4}
set_f = {3, 4, 5, 6}
set_g = {3, 4, 7, 8}
common_elements = set_e.intersection(set_f).intersection(set_g)
print(common_elements)  # Output: {3, 4}

# Using set operations to find unique elements between multiple sets
unique_elements = set_e.union(set_f).union(set_g)
print(unique_elements)  # Output: {1, 2, 3, 4, 5, 6, 7, 8}

# Copying a set
my_set_copy = my_set.copy()
print(my_set_copy)  # Output: {} (or original set elements if my_set was not cleared)

# Frozenset - immutable version of a set
frozen_set = frozenset([1, 2, 3])
print(frozen_set)  # Output: frozenset({1, 2, 3})

# Using comparison operators with sets
set_h = {1, 2, 3}
set_i = {1, 2, 3, 4}
print(set_h < set_i)   # Output: True (set_h is a proper subset of set_i)
print(set_h <= set_i)  # Output: True (set_h is a subset of set_i)
print(set_h > set_i)   # Output: False (set_h is not a proper superset of set_i)
print(set_h >= set_i)  # Output: False (set_h is not a superset of set_i)

# Using `symmetric_difference_update` to update the set with the symmetric difference
set_a.symmetric_difference_update(set_d)
print(set_a)  # Output: {1, 2, 3, 4, 5, 6}

# Using `intersection_update` to keep only the elements that are in both sets
set_a.intersection_update(set_d)
print(set_a)  # Output: {4, 5}

# Using `difference_update` to remove elements found in another set
set_a.difference_update(set_d)
print(set_a)  # Output: set() (or remaining elements after the difference operation)

# Using `update` with different iterable types
set_a.update(range(5))  # Adding elements from range
print(set_a)  # Output: {0, 1, 2, 3, 4}

# Using `pop` and handling potential KeyError when the set is empty
try:
    print(set_a.pop())  # Will remove and return an arbitrary element
except KeyError:
    print("Set is empty, cannot pop an element.")

# Checking if a set is empty
print("Is set_a empty?", len(set_a) == 0)  # Output: False

# Comparing two sets for equality
set_h = {1, 2, 3}
set_i = {3, 2, 1}
print(set_h == set_i)  # Output: True (order does not matter in sets)



#LAMBDA FUNCTION
# Example 1: Lambda function to add two numbers
# This is a simple lambda function that takes two arguments and returns their sum.
add = lambda x, y: x + y
# Calling the lambda function
print(add(5, 3))  # Output: 8

# Example 2: Using lambda with map() to square all elements in a list
numbers = [1, 2, 3, 4, 5]
# map() applies the lambda function (which squares a number) to each element in the list
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)  # Output: [1, 4, 9, 16, 25]

# Example 3: Using lambda with filter() to get even numbers from a list
numbers = [1, 2, 3, 4, 5, 6]
# filter() applies the lambda function (which returns True for even numbers) to each element
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # Output: [2, 4, 6]

# Example 4: Using lambda with sorted() to sort a list of tuples by the second element
points = [(2, 3), (1, 2), (4, 1)]
# sorted() uses the lambda function to sort by the second element in each tuple
sorted_points = sorted(points, key=lambda x: x[1])
print(sorted_points)  # Output: [(4, 1), (1, 2), (2, 3)]

# Example 5: Using lambda with reduce() to multiply all elements in a list
from functools import reduce
numbers = [1, 2, 3, 4]
# reduce() applies the lambda function (which multiplies two numbers) across the list
product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 24

# Example 6: Using lambda for quick inline function (e.g., event handler or callback)
button_click = lambda: print("Button clicked!")
button_click()  # Output: Button clicked!


# --------- Advantages of lambda functions ---------
# 1. Concise and Simple: Lambda functions allow for quick, inline function definitions, saving space in the code.
#    This is useful for simple, one-liner functions that do not require a full function definition.
#
# 2. Anonymous: Lambda functions do not require a name, so they are perfect for short-lived, throwaway functions.
#    This is helpful when you need a function only once, like in map(), filter(), or sorted().
#
# 3. Functional Programming: Lambda functions fit naturally in functional programming where functions are passed as arguments,
#    such as map(), filter(), reduce(), etc.

# --------- Disadvantages of lambda functions ---------
# 1. Limited to Single Expressions: Lambda functions are limited to only a single expression.
#    They cannot contain multiple statements, loops, or complex logic, making them unsuitable for more detailed operations.
#
# 2. Reduced Readability: Overuse of lambda functions can make the code harder to read, especially for those unfamiliar with lambdas.
#    Named functions (using def) often provide better readability and understanding of the code's purpose.
#
# 3. Lack of Documentation: Since lambda functions are anonymous, they cannot be easily documented or reused.
#    Defining a regular function allows for docstrings and better clarity in larger projects.


#some interesting problems
# 1. Modifying a list while iterating over it
# Problem: Removing elements while iterating can lead to unexpected results.

numbers = [1, 2, 3, 4, 5]
for num in numbers:
  if num % 2 == 0:
    numbers.remove(num)  # Incorrect way to remove elements AS the elements index changes but the loops iteration value doesn't.

print(numbers)  #(Incorrect, should be [1, 3, 5])

# Solution: Create a new list or use list comprehension.
numbers = [1, 2, 3, 4, 5]
new_numbers = [num for num in numbers if num % 2 != 0]
print(new_numbers)  # Output: [1, 3, 5] (Correct)
# 2. Modifying a tuple (immutability)
# Problem: Tuples are immutable, so you can't change their elements directly.

my_tuple = (1, 2, 3)
my_tuple[0] = 5  # This will raise a TypeError

# Solution: Create a new tuple with the modified element.
new_tuple = (5,) + my_tuple[1:]
print(new_tuple)  # Output: (5, 2, 3)

# 3. Duplicate keys in dictionaries
# Problem: Dictionaries cannot have duplicate keys. If you try to assign a value to an existing key, it will overwrite the previous value.

my_dict = {"a": 1, "b": 2, "a": 3}
print(my_dict)  # Output: {'a': 3, 'b': 2} (The first value for 'a' is overwritten)

# Solution: If you need to store multiple values for a key, use a list or another dictionary as the value.
my_dict = {"a": [1, 3], "b": 2}
print(my_dict)  # Output: {'a': [1, 3], 'b': 2}

# 4. Unhashable types as dictionary keys
# Problem: Dictionary keys must be hashable (immutable). Using mutable types like lists as keys will raise a TypeError.

my_dict = {[1, 2]: 3}  # This will raise a TypeError

# Solution: Use tuples instead of lists for keys if you need a collection as a key.
my_dict = {(1, 2): 3}
print(my_dict)  # Output: {(1, 2): 3}

# 5. Unexpected behavior with default values in dictionaries
# Problem: Using mutable objects as default values in the `get()` method can lead to unexpected behavior.

my_dict = {}
my_dict.setdefault("a", []).append(1)
my_dict.setdefault("a", []).append(2)
print(my_dict)  # Output: {'a': [1, 2]} (The same list is used as the default value)

# Solution: Use a lambda function to create a new default value each time.
my_dict = {}
my_dict.setdefault("a", lambda: []).append(1)
my_dict.setdefault("a", lambda: []).append(2)
print(my_dict)  # Output: {'a': [1], 'a': [2]} (Separate lists are created)
numbers = [2,4,6,8,10]

for num in numbers:
  print(numbers[0])
  if num % 2 == 0:
    numbers.remove(num)

print(numbers)
# gives wrong results
# why?
# First Iteration: num = 2 (index 0). 2 is even, so it’s removed. The list becomes [4, 6, 8, 10].
# Second Iteration: Now, num = 6 (index 1). 4 is even, so it’s removed. The list becomes [4, 8, 10].
# Third Iteration: Now, num = 10 (index 2). 10 is even, so it’s removed. The list becomes [4, 8].
# Loop Ends: The loop ends because there are no more elements to iterate
numbers = [2,4,6,8,10]
numbers[:]
numbers = [2,4,6,8,10]
for num in numbers[:]:
  print(numbers[0])    #Iterates over a copy of the list.
  if num % 2 == 0:
    numbers.remove(num)

print(numbers)
# Gives correct results. Why?
#as the number gets removed the copy also changes hence it always uses the 0 th index of the list

# Always works, reliable
numbers = [2, 2, 2, 2, 2]
index = 0

while index < len(numbers):
    if numbers[index] == 2:
        numbers.remove(2)
    index += 1
    print(index)
print(numbers)
# First Iteration:  index = 0, numbers[0] = 2. 2 is removed. The list becomes [2, 2, 2, 2]. The index is incremented to 1.
# Second Iteration: index = 1, numbers[1] = 2. 2 is removed. The list becomes [2, 2, 2]. The index is incremented to 2.
# Third Iteration:  index = 2, numbers[2] = 2. 2 is removed. The list becomes [2, 2]. The index is incremented to 3.
# Fourth Iteration: index = 3, but now the length of numbers is 2, so the loop exits
# Iterating backward
numbers = [2,2,2,2,2,2,2,2,2,2]
index = len(numbers) - 1

while index >= 0:
    if numbers[index] == 2:
        numbers.remove(2)
    index -= 1
    print(index)
print(numbers)
#Unreliable coding - yet another example
def append_to_list(value, my_list=[]):
    my_list.append(value)
    return my_list

# Call the function multiple times
result1 = append_to_list(1)
print(result1)
result2 = append_to_list(2)
print(result2)
result3 = append_to_list(3)
print(result3)
# print(result1)  # Expected: [1]  but got [1]
# print(result2)  # Expected: [2]  but got [1, 2]
# print(result3)  # Expected: [3]  but got [1, 2, 3]

#True way
def append_to_list(value,my_list=None):
    if my_list is None:
        my_list = []
    my_list.append(value)
    return my_list

# Call the function multiple times
result1 = append_to_list(1)
result2 = append_to_list(2)
result3 = append_to_list(3)

print(result1)  # Expected: [1]
print(result2)  # Expected: [2]
print(result3)  # Expected: [3]

#this is it from my side
#I hope it is helpful
#thank you

