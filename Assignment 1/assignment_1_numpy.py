"""Assignment 1 - NumPy solutions.

Run:
    python3 assignment_1_numpy.py
"""

from pathlib import Path

import numpy as np
from PIL import Image


def heading(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def question_1():
    heading("Q1: Questions on Basic NumPy Array")

    print("\nQ1(a): Reverse the NumPy array")
    arr = np.array([1, 2, 3, 6, 4, 5])
    print("Original:", arr)
    print("Reversed:", arr[::-1])

    print("\nQ1(b): Flatten the NumPy array using two methods")
    array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
    print("Using flatten():", array1.flatten())
    print("Using ravel():", array1.ravel())

    print("\nQ1(c): Compare two NumPy arrays")
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[1, 2], [3, 4]])
    print("Element-wise comparison:\n", arr1 == arr2)
    print("Arrays are equal:", np.array_equal(arr1, arr2))

    print("\nQ1(d): Most frequent value and indices")
    for name, values in {
        "x": np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1]),
        "y": np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3]),
    }.items():
        unique, counts = np.unique(values, return_counts=True)
        most_frequent = unique[np.argmax(counts)]
        indices = np.where(values == most_frequent)[0]
        print(f"{name}: most frequent value = {most_frequent}, indices = {indices}")

    print("\nQ1(e): Matrix sums")
    gfg = np.matrix("[4, 1, 9; 12, 3, 1; 4, 5, 6]")
    print("Matrix:\n", gfg)
    print("Sum of all elements:", np.sum(gfg))
    print("Row-wise sum:", np.sum(gfg, axis=1))
    print("Column-wise sum:", np.sum(gfg, axis=0))

    print("\nQ1(f): Matrix operations")
    n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])
    print("Matrix:\n", n_array)
    print("Sum of diagonal elements:", np.trace(n_array))
    eigen_values, eigen_vectors = np.linalg.eig(n_array)
    print("Eigen values:", eigen_values)
    print("Eigen vectors:\n", eigen_vectors)
    print("Inverse:\n", np.linalg.inv(n_array))
    print("Determinant:", np.linalg.det(n_array))

    print("\nQ1(g): Matrix multiplication and covariance")
    p1 = np.array([[1, 2], [2, 3]])
    q1 = np.array([[4, 5], [6, 7]])
    print("Case i product:\n", np.matmul(p1, q1))
    print("Case i covariance:\n", np.cov(p1.ravel(), q1.ravel()))

    p2 = np.array([[1, 2], [2, 3], [4, 5]])
    q2 = np.array([[4, 5, 1], [6, 7, 2]])
    print("Case ii product:\n", np.matmul(p2, q2))
    print("Case ii covariance:\n", np.cov(p2.ravel(), q2.ravel()))

    print("\nQ1(h): Inner, outer and cartesian product")
    x = np.array([[2, 3, 4], [3, 2, 9]])
    y = np.array([[1, 5, 0], [5, 10, 3]])
    print("Inner product:", np.inner(x, y))
    print("Outer product:\n", np.outer(x, y))
    cartesian = np.array(np.meshgrid(x.ravel(), y.ravel())).T.reshape(-1, 2)
    print("Cartesian product:\n", cartesian)


def question_2():
    heading("Q2: Based on NumPy Mathematics and Statistics")

    print("\nQ2(a): Absolute, percentile, mean, median and standard deviation")
    array = np.array([[1, -2, 3], [-4, 5, -6]])
    print("Array:\n", array)
    print("Absolute values:\n", np.abs(array))
    print("Percentiles of flattened array:", np.percentile(array, [25, 50, 75]))
    print("Percentiles column-wise:\n", np.percentile(array, [25, 50, 75], axis=0))
    print("Percentiles row-wise:\n", np.percentile(array, [25, 50, 75], axis=1))
    print("Mean flattened:", np.mean(array))
    print("Mean column-wise:", np.mean(array, axis=0))
    print("Mean row-wise:", np.mean(array, axis=1))
    print("Median flattened:", np.median(array))
    print("Median column-wise:", np.median(array, axis=0))
    print("Median row-wise:", np.median(array, axis=1))
    print("Standard deviation flattened:", np.std(array))
    print("Standard deviation column-wise:", np.std(array, axis=0))
    print("Standard deviation row-wise:", np.std(array, axis=1))

    print("\nQ2(b): Floor, ceiling, truncated and rounded values")
    a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
    print("Original:", a)
    print("Floor:", np.floor(a))
    print("Ceiling:", np.ceil(a))
    print("Truncated:", np.trunc(a))
    print("Rounded:", np.round(a))


def question_3():
    heading("Q3: Based on Searching and Sorting")

    print("\nQ3(a): Sorting")
    array = np.array([10, 52, 62, 16, 16, 54, 453])
    print("Original:", array)
    print("Sorted array:", np.sort(array))
    print("Indices of sorted array:", np.argsort(array))
    print("4 smallest elements:", np.sort(array)[:4])
    print("5 largest elements:", np.sort(array)[-5:])

    print("\nQ3(b): Integer and float elements")
    array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
    integer_elements = array[array == array.astype(int)]
    float_elements = array[array != array.astype(int)]
    print("Integer elements only:", integer_elements)
    print("Float elements only:", float_elements)


def img_to_array(path):
    """Read an RGB or grayscale image and save its pixel array as a text file."""
    image_path = Path(path)
    image = Image.open(image_path)
    image_array = np.asarray(image)
    output_path = image_path.with_suffix(".txt")

    if image.mode == "L":
        np.savetxt(output_path, image_array, fmt="%d")
    elif image.mode in {"RGB", "RGBA"}:
        flattened = image_array.reshape(-1, image_array.shape[-1])
        np.savetxt(output_path, flattened, fmt="%d")
    else:
        converted = image.convert("RGB")
        image_array = np.asarray(converted)
        flattened = image_array.reshape(-1, 3)
        np.savetxt(output_path, flattened, fmt="%d")

    return image_array, output_path


def question_4():
    heading("Q4: Image to array and loading saved file")

    demo_dir = Path(__file__).parent / "demo_images"
    demo_dir.mkdir(exist_ok=True)

    rgb_path = demo_dir / "rgb_demo.png"
    gray_path = demo_dir / "gray_demo.png"

    Image.fromarray(
        np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    ).save(rgb_path)
    Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8), mode="L").save(gray_path)

    for image_path in [rgb_path, gray_path]:
        image_array, text_path = img_to_array(image_path)
        loaded_array = np.loadtxt(text_path, dtype=int)
        print(f"Image: {image_path.name}")
        print("Saved text file:", text_path)
        print("Original array shape:", image_array.shape)
        print("Loaded array from text file:\n", loaded_array)


def main():
    question_1()
    question_2()
    question_3()
    question_4()


if __name__ == "__main__":
    main()
