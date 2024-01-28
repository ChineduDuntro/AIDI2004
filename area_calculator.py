# This program calculates the area of a rectangle

# Function to calculate area of rectangle
def calculate_area(length, width):
    return length * width

# Main program
if __name__ == "__main__":
    # Taking user input for length and width of the rectangle
    length = float(input("Enter the length of the rectangle: "))
    width = float(input("Enter the width of the rectangle: "))
    
    # Calculating the area
    area = calculate_area(length, width)
    
    # Printing the result
    print(f"The area of the rectangle is: {area} square units")