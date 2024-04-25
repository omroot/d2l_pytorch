from abc import ABC, abstractmethod
from typing import List, Iterator


class Measurable(ABC):
    """An abstract base class representing a measurable object."""

    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the object."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the object."""
        pass


class Drawable(ABC):
    """An abstract base class representing a drawable object."""

    @abstractmethod
    def draw(self) -> None:
        """Draw the object."""
        pass


class Shape(Measurable, Drawable):
    """An abstract base class representing a shape."""

    @abstractmethod
    def area(self) -> float:
        """Calculate the area of the shape."""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate the perimeter of the shape."""
        pass

    @abstractmethod
    def draw(self) -> None:
        """Draw the shape."""
        pass


class Circle(Shape):
    """A concrete implementation of a circle."""

    def __init__(self, radius: float) -> None:
        """Initialize the Circle with a given radius."""
        self.__radius: float = radius

    @property
    def radius(self) -> float:
        """Get the radius of the circle."""
        return self.__radius

    def area(self) -> float:
        """Calculate the area of the circle."""
        return 3.14159 * self.__radius ** 2

    def perimeter(self) -> float:
        """Calculate the perimeter of the circle."""
        return 2 * 3.14159 * self.__radius

    def draw(self) -> None:
        """Draw the circle."""
        print("Drawing a circle with radius:", self.__radius)


class Rectangle(Shape):
    """A concrete implementation of a rectangle."""

    def __init__(self, length: float, width: float) -> None:
        """Initialize the Rectangle with given length and width."""
        self.__length: float = length
        self.__width: float = width

    @property
    def length(self) -> float:
        """Get the length of the rectangle."""
        return self.__length

    @property
    def width(self) -> float:
        """Get the width of the rectangle."""
        return self.__width

    def area(self) -> float:
        """Calculate the area of the rectangle."""
        return self.__length * self.__width

    def perimeter(self) -> float:
        """Calculate the perimeter of the rectangle."""
        return 2 * (self.__length + self.__width)

    def draw(self) -> None:
        """Draw the rectangle."""
        print("Drawing a rectangle with length:", self.__length, "and width:", self.__width)


class ShapeGroup(Measurable, Drawable):
    """A group of shapes."""

    def __init__(self) -> None:
        """Initialize the ShapeGroup."""
        self.shapes: List[Shape] = []

    def add_shape(self, shape: Shape) -> None:
        """Add a shape to the ShapeGroup."""
        self.shapes.append(shape)

    def area(self) -> float:
        """Calculate the total area of all shapes in the ShapeGroup."""
        return sum(shape.area() for shape in self.shapes)

    def perimeter(self) -> float:
        """Calculate the total perimeter of all shapes in the ShapeGroup."""
        return sum(shape.perimeter() for shape in self.shapes)

    def draw(self) -> None:
        """Draw all shapes in the ShapeGroup."""
        for shape in self.shapes:
            shape.draw()

    def __iter__(self) -> Iterator[Shape]:
        """Iterator for the ShapeGroup."""
        return iter(self.shapes)

    def __len__(self) -> int:
        """Return the number of shapes in the ShapeGroup."""
        return len(self.shapes)


# Unit tests
import unittest


class TestShapeGroup(unittest.TestCase):
    """Unit tests for ShapeGroup class."""

    def test_area(self) -> None:
        """Test calculation of total area."""
        circle = Circle(5)
        rectangle = Rectangle(3, 4)
        group = ShapeGroup()
        group.add_shape(circle)
        group.add_shape(rectangle)
        self.assertAlmostEqual(group.area(), 5 ** 2 * 3.14159 + 3 * 4, places=5)

    def test_perimeter(self) -> None:
        """Test calculation of total perimeter."""
        circle = Circle(5)
        rectangle = Rectangle(3, 4)
        group = ShapeGroup()
        group.add_shape(circle)
        group.add_shape(rectangle)
        self.assertAlmostEqual(group.perimeter(), 2 * 3.14159 * 5 + 2 * (3 + 4), places=5)

    def test_draw(self) -> None:
        """Test drawing of shapes."""
       
