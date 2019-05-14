import sys
import argparse
from pathlib import Path

import numpy as np


def create_triangle(im, p0, p1, p2, color=None):
    """ Draw a triangle using the floating-point version of Bresenham's line-drawing algorithm to determine how to step.

    Parameters
    ----------
    im : numpy.ndarray, shape=(R, C, K)
        The image on which to draw a line.

    p0 : Tuple[int, int]
        The (x, y) position of the first vertex.

    p1 : Tuple[int, int]
        The (x, y) position of the second vertex.

    p2 : Tuple[int, int]
        The (x, y) position of the third vertex.

    color : ArrayLike, shape=(K,) or None, optional (default=None)
        The color of the rectangle, in RGB, or None to randomly generate a color.

    Returns
    -------
    Tuple[int, int, int, int]
        (left, top, right, bottom) of the rectangle.
    """
    color = np.random.rand(3) if color is None else color

    xs = [p0[0], p1[0], p2[0]]
    ys = [p0[1], p1[1], p2[1]]
    x_order = np.argsort(xs)

    leftmost = (xs[x_order[0]], ys[x_order[0]])
    middle = (xs[x_order[1]], ys[x_order[1]])
    rightmost = (xs[x_order[2]], ys[x_order[2]])

    # we'll step from leftmost to middle along the lines leftmost->middle and
    # leftmost->rightmost, moving one pixel horizontally and however many
    # pixels vertically we need to in order to maintain the line, according to
    # Bresenham's line-drawing algorithm. Once we've moved vertically as much as
    # necessary, we'll draw a vertical line connecting the two lines along which we're
    # walking, which will fill the first part of the triangle
    lr_dx = rightmost[0] - leftmost[0]
    lm_dx = middle[0] - leftmost[0]
    lr_dy = rightmost[1] - leftmost[1]
    lm_dy = middle[1] - leftmost[1]
    lr_derr = abs(lr_dy / lr_dx)
    lm_derr = abs(lm_dy / lm_dx)
    lr_err = 0
    lm_err = 0
    lr_y = leftmost[1]
    lm_y = leftmost[1]
    for x in range(leftmost[0], middle[0]):
        if lr_y < lm_y:
            im[lr_y:lm_y+1, x] = color
        else:
            im[lm_y:lr_y+1, x] = color

        lr_err += lr_derr
        lm_err += lm_derr
        while lm_err >= 0.5:
            lm_y += 1 if lm_dy > 0 else -1
            lm_err -= 1
        while lr_err >= 0.5:
            lr_y += 1 if lr_dy > 0 else -1
            lr_err -= 1

    # now that we have this, we'll recompute our y step and go middle->rightmost and
    # continue along the path we've been drawing from leftmost->rightmost, again
    # taking a single horizontal step and as many vertical steps as we need to,
    # connecting these lines to fill the second part of the triangle
    mr_dx = rightmost[0] - middle[0]
    mr_dy = rightmost[1] - middle[1]
    mr_derr = abs(mr_dy / mr_dx)
    mr_err = 0
    mr_y = middle[1]
    for x in range(middle[0], rightmost[0]):
        if lr_y < mr_y:
            im[lr_y:mr_y+1, x] = color
        else:
            im[mr_y:lr_y+1, x] = color

        lr_err += lr_derr
        mr_err += mr_derr
        while lr_err >= 0.5:
            lr_y += 1 if lr_dy > 0 else -1
            lr_err -= 1
        while mr_err >= 0.5:
            mr_y += 1 if mr_dy > 0 else -1
            mr_err -= 1

    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)


def create_rectangle(image, *, center=(0, 0), width=10, height=10, color=None):
    """ Generate a rectangle at the given position and size with a specified color on the given image.

    Parameters
    ----------
    image : numpy.ndarray, shape=(R, C, 3)
        The image on which to create and add a rectangle.

    center : Tuple[int, int], optional (default=(0, 0))
        The (x, y) position of the center of the rectangle.

    width : int, optional (default=10)
        The width of the rectangle.

    height : int, optional (default=10)
        The height of the rectangle.

    color : ArrayLike[Real, Real, Rela] or None, optional (default=None)
        The color of the rectangle, in RGB, or None to randomly generate a color.

    Returns
    -------
    Tuple[int, int, int, int]
        (left, top, right, bottom) of the rectangle.
    """
    color = np.random.rand(3) if color is None else color

    x, y = center
    left, right = x - width // 2, x + width // 2
    top, bottom = y - height // 2, y + height // 2

    image[top:bottom, left:right] = color
    return left, top, right, bottom


def create_circle(image, *, center=(0, 0), radius=10, color=None):
    """ Create a circle at a given position and size with a specified color on the given image.

    Parameters
    ----------
    image : numpy.ndarray, shape=(R, C, 3)
        The image on which to create and add a circle.

    center : Tuple[int, int], optional (default=(0, 0))
        The (x, y) position of the center of the circle.

    radius : int, optional (default=10)
        The radius of the circle.

    color : ArrayLike[Real, Real, Real] or None, optional (default=None)
        The color of the circle, in RGB, or None to randomly generate a color.

    Returns
    -------
    Tuple[int, int, int, int]
        (left, top, right, bottom) of the circle.

    Notes
    -----
    This uses the midpoint circle algorithm (an extension of Bresenham's algorithm) to put a circle on the image.
    """
    color = np.random.rand(3) if color is None else color

    x0, y0 = center
    left, right = x0 - radius, x0 + radius
    top, bottom = y0 - radius, y0 + radius

    f = 1 - radius
    dx = 0
    dy = -2 * radius
    x, y = 0, radius
    while x < y:
        if f >= 0:
            y -= 1
            dy += 2
            f += dy

        x += 1
        dx += 2
        f += dx + 1
        image[y0-x:y0+x+1, x0-y:x0+y+1] = color
        image[y0-y:y0+y+1, x0-x:x0+x+1] = color

    return left, top, right, bottom


def generate_image(*, num_objects=3, image_shape=(256, 256), object_minimum_distance=20,
                   object_minimum_size=10, object_maximum_size=20):
    """ Generate a single image and metadata with the provided constraints.

    Parameters
    ----------
    num_objects : int, optional (default=3)
        The number of objects to have in the image.

    image_shape : Tuple[int, in], optional (default=(64, 64))
        The height and width of the image.

    object_minimum_distance : int, optional (default=20)
        The minimum distance between the centers of any two objects in the scene.

    object_minimum_size : int, optional (default=10)
        The minimum size of an object.

    object_maximum_size : int, optional (default=20)
        The maximum size of an object.

    Returns
    -------
    Tuple[numpy.ndarray shape=image_shape, List[Tuple[str, int, int, int, int]]]
        The image and its corresponding detections.
    """
    # initialize the metadata lists
    labels, boxes = [], []
    centers = []  # the center of each object in the scene, for comparisons only

    # first, generate a background
    background_color = np.random.rand(3) * 0.5
    image = np.ones((*image_shape, 3)) * background_color + np.random.randn(*image_shape, 3) * 0.05
    image = (image - image.min()) / (image.max() - image.min())

    # next, generate the number of things to put on the background
    object_labels = np.random.randint(1, 4, num_objects)  # rectangle, square, circle

    # place the objects in the scene
    min_x, min_y = object_maximum_size, object_maximum_size
    max_x, max_y = image_shape[1] - object_maximum_size, image_shape[0] - object_maximum_size
    for obj_label in object_labels:
        x = np.random.randint(min_x, max_x)
        y = np.random.randint(min_y, max_y)

        # if there is an object in the scene
        # make sure the distance from this object to the others is at least the minimum
        if centers:
            while np.linalg.norm(np.array([x, y]) - np.array(centers), axis=1).min() < object_minimum_distance:
                x = np.random.randint(min_x, max_x)
                y = np.random.randint(min_y, max_y)  # this is inefficient but works fine for a small example

        # generate the color
        color = np.random.rand(3)

        # we now have a position, a type, and a color
        # place the object!
        if obj_label == 1:
            w, h = np.random.randint(object_minimum_size, object_maximum_size, 2)
            bounds = create_rectangle(image, center=(x, y), width=w, height=h, color=color)
        elif obj_label == 2:
            low = object_minimum_size - object_maximum_size
            high = object_maximum_size - object_minimum_size
            p0 = np.random.randint(low, high, 2)
            p1 = np.random.randint(low, high, 2)
            while p1[0] == p0[0] or p1[1] == p0[1]:
                p1 = np.random.randint(low, high, 2)
            p2 = np.random.randint(low, high, 2)
            while p2[0] == p0[0] or p2[0] == p1[0] or p2[1] == p1[1] or p2[1] == p0[1]:
                p2 = np.random.randint(low, high, 2)

            center = np.array((x, y))
            bounds = create_triangle(image, p0 + center, p1 + center, p2 + center, color=color)
        elif obj_label == 3:
            radius = np.random.randint(object_minimum_size // 2, object_maximum_size // 2)
            bounds = create_circle(image, center=(x, y), radius=radius, color=color)

        centers.append((x, y))
        labels.append(obj_label)
        boxes.append(bounds)

    return image, np.array(labels, dtype=np.int32), np.array(boxes, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser('generate-dataset')
    parser.add_argument('-d', '--destination', type=str, default='./data',
                        help='The base path in which to save the generated dataset')
    parser.add_argument('-n', '--num_images', type=int, default=1000,
                        help='The number of images to generate')
    parser.add_argument('--min_objects', type=int, default=3,
                        help='The minimum number of objects per image')
    parser.add_argument('--max_objects', type=int, default=8,
                        help='The maximum number of objects per image')
    parser.add_argument('--min_object_size', type=int, default=24,
                        help='The minimum pixel size of each object')
    parser.add_argument('--max_object_size', type=int, default=40,
                        help='The maximum pixel size of each object')

    args = parser.parse_args()

    min_dist = np.sqrt(2 * args.max_object_size**2)
    images, labels, boxes = [], [], []
    for _ in range(args.num_images):
        num_objects = np.random.randint(args.min_objects, args.max_objects + 1)
        img, lbls, bxs = generate_image(num_objects=num_objects,
                                        object_minimum_size=args.min_object_size,
                                        object_maximum_size=args.max_object_size,
                                        object_minimum_distance=min_dist)
        images.append(img)
        labels.append(lbls)
        boxes.append(bxs)

    path = Path(args.destination)
    np.save(path / 'images.npy', np.array(images, dtype=np.float32))
    np.save(path / 'labels.npy', labels)  # rectangle triangle circle
    np.save(path / 'boxes.npy', boxes)


if __name__ == '__main__':
    sys.exit(main())
