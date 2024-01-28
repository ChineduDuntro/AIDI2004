# This program calculates the factorial of a non-negative integer

# Function to calculate factorial using recursion
def factorial(n):
    # Base case: if n is 0 or 1, the factorial is 1
    if n in (0, 1):
        return 1
    # Recursive case: n! = n * (n-1)!
    else:
        return n * factorial(n - 1)

# Main program
if __name__ == "__main__":
    while True:
        # Taking user input
        try:
            number = int(input("Enter a non-negative integer to calculate its factorial: "))
            if number < 0:
                print("Please enter a non-negative integer.")
                continue

            # Calculating the factorial
            result = factorial(number)

            # Printing the result
            print(f"The factorial of {number} is: {result}")
            break  # Exit the loop if everything went well

        except ValueError:
            print("Invalid input! Please enter an integer.")

