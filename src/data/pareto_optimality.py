###Eddie's code
from scipy.optimize import fmin_cobyla
from sympy import *
import numpy as np
import pandas as pd
import math


def paretoFront(df):
    # This function calculates the pareto frontier given a group of points, and outputs them as a, b, c
    # in the quadratic equation ax^2+bx+c

    # Alycia's section
    y1 = df["nGaps"].min()
    x1 = df[df["nGaps"] == y1]["score"].max()
    x3 = df["score"].max()
    y3 = df[df["score"] == x3]["nGaps"].min()
    l = list(df[df["nGaps"] <= y3]["nGaps"].unique())
    l.sort()
    y2 = l[(len(l) - 1) // 2]
    x2 = df[df["nGaps"] == y2]["score"].max()

    unique_points = get_unique_points(y1, x1, x3, y3, y2, x2)

    if len(unique_points) == 3:  # Best case scenario, can use original algorithm
        pass  # proceed to the original algorithm below

    elif len(unique_points) == 2:  # Take the midpoint between these two points
        x1, y1 = unique_points[0]
        x2, y2 = unique_points[1]
        x3 = (
            x2 - x1
        ) / 2 + x1  # Take the x-coordinate of the point in between x1 and x2
        y3 = (
            y2 - y1
        ) / 2 + y1  # Take the y-coordinate of the point in between y1 and y2

    elif len(unique_points) == 1:  # Take the line from (0, 0) to this point
        x1 = 0
        y1 = 0
        x2, y2 = unique_points[0]
        x3 = (
            x2 - x1
        ) / 2 + x1  # Take the x-coordinate of the point in between x1 and x2
        y3 = (
            y2 - y1
        ) / 2 + y1  # Take the y-coordinate of the point in between y1 and y2

    else:  # if some weird shit happens, continue to next puzzle
        return [-1]  # In the main method, this will make this puzzle be skipped over

    a, b, c = symbols("a b c")

    # important, flipped x and y so that x is estimating score
    return solve(
        [
            Eq(a * y1**2 + b * y1 + c, x1),
            Eq(a * y2**2 + b * y2 + c, x2),
            Eq(a * y3**2 + b * y3 + c, x3),
        ],
        [a, b, c],
    )


def get_unique_points(y1, x1, x3, y3, y2, x2):
    # Jenny's section (added to make sure pareto frontiers can be found for all puzzles)
    if (
        x1 == x2
    ):  # These few if statements are to make sure that there aren't points with same x and different y (parabola can't be drawn this way)
        if y1 < y2:
            y1 = y2
        else:
            y2 = y1

    if x2 == x3:
        if y2 < y3:
            y2 = y3
        else:
            y3 = y2

    if x1 == x3:
        if y1 < y3:
            y1 = y3
        else:
            y3 = y1

    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = (x3, y3)

    unique_points = list(set([p1, p2, p3]))
    return unique_points


def process_puzzle_solutions_pareto_dist(sol):

    result_df = pd.DataFrame(sol, columns=["score", "nGaps"])
    minGaps = result_df["nGaps"].max()
    maxGaps = result_df["nGaps"].min()

    min_score = result_df["score"].min()

    # estimate the pareto frontier
    ans = paretoFront(
        result_df
    )  # ans holds the values of a, b, c that defines the pareto frontier,
    if len(ans) != 3:
        return False

    # calculate shortest distance to pareto frontier for every solution
    for i, p in enumerate(sol):

        p["paretoDist"] = get_shortest_distance_to_pareto(
            maxGaps, minGaps, p["score"], p["nGaps"], ans
        )
        p["horizontalDist"] = get_score_distance_to_pareto(p["score"], p["nGaps"], ans)
        p["horizontalProportion"] = get_horizontal_proportion(
            p["score"], p["nGaps"], ans, min_score
        )

    return True


def get_horizontal_proportion(score, nGaps, ans, min_score):
    a, b, c = symbols("a b c")

    def f(x):
        return ans[a] * x**2 + ans[b] * x + ans[c]

    paretoFrontScore = abs(f(nGaps))

    if paretoFrontScore == min_score:
        return 1
    return (score - min_score) / (paretoFrontScore - min_score)


def get_numpy_roots_pareto_dist(score, nGaps, ans):
    a, b, c = symbols("a b c")

    p = []
    p[0] = 2 * (ans[a] ** 2)
    p[1] = 3 * ans[a] * ans[b]
    p[2] = ans[b] ** 2 + 2 * ans[a] * ans[c] - 2 * ans[a] * nGaps + 1
    p[3] = ans[b] * ans[c] - ans[b] * nGaps - score
    roots = np.roots(p)
    dist = math.inf
    for x in roots:
        new_dist = np.sqrt(
            (x - score) ** 2 + (ans[a] * x**2 + ans[b] * x + ans[c] - nGaps) ** 2
        )
        if new_dist < dist:
            dist = new_dist
    return new_dist


def get_shortest_distance_to_pareto(maxGaps, minGaps, score, nGaps, ans):
    a, b, c = symbols("a b c")

    def f(x):
        return ans[a] * x**2 + ans[b] * x + ans[c]

    def distance(X):
        x, y = X
        return np.sqrt((x - nGaps) ** 2 + (y - score) ** 2)

    def c1(X):
        x, y = X
        return f(x) - y

    def c2(X):
        x, y = X
        return y - f(x)

    X = fmin_cobyla(
        distance, x0=[minGaps, maxGaps], cons=[c2, c1], rhobeg=10, rhoend=0.5, maxfun=20
    )
    return distance(X)


def get_score_distance_to_pareto(score, nGaps, ans):
    a, b, c = symbols("a b c")

    def f(x):
        return ans[a] * x**2 + ans[b] * x + ans[c]

    return abs(f(nGaps) - score)
