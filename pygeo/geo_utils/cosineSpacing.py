import numpy as np
from pygeo import geo_utils
from pyspline import Curve


def getCosineIntersections(surf, leList, teList, nSpan, nChord):
    """
    A variant of _getSurfaceIntersections in DVCon that uses cosine spacing instead of TFI.

    """

    # Get points from surface list
    p0, p1, p2 = surf

    # Create leading and trailing edge curves from the provided lists
    le_s = Curve(X=leList, k=2)
    te_s = Curve(X=teList, k=2)

    # Generate spanwise parametric distances
    if isinstance(nSpan, int):
        # Use equal spacing along the curve
        le_span_s = te_span_s = np.linspace(0.0, 1.0, nSpan)
    elif isinstance(nSpan, list):
        # Use equal spacing within each segment defined by leList and teList

        # We use the same nSpan for the leading and trailing edges, so check that the lists are the same size
        if len(leList) != len(teList):
            raise ValueError("leList and teList must be the same length if nSpan is provided as a list.")

        # Also check that nSpan is the correct length
        numSegments = len(leList) - 1
        if len(nSpan) != numSegments:
            raise ValueError(f"nSpan must be of length {numSegments}.")

        # Find the parametric distances of the break points that define each segment
        le_breakPoints = le_s.projectPoint(leList)[0]
        te_breakPoints = te_s.projectPoint(teList)[0]

        # Initialize empty arrays for the full spanwise parameteric distances
        le_span_s = np.array([])
        te_span_s = np.array([])

        for i in range(numSegments):
            # Only include the endpoint if this is the last segment to avoid double counting points
            if i == numSegments - 1:
                endpoint = True
            else:
                endpoint = False

            # Interpolate over this segment and append to the parametric distance array
            le_span_s = np.append(
                le_span_s, np.linspace(le_breakPoints[i], le_breakPoints[i + 1], nSpan[i], endpoint=endpoint)
            )
            te_span_s = np.append(
                te_span_s, np.linspace(te_breakPoints[i], te_breakPoints[i + 1], nSpan[i], endpoint=endpoint)
            )
        # Get the total number of spanwise sections
        nSpanTotal = np.sum(nSpan)

    elif not nSpan:
        if len(leList) != len(teList):
            raise ValueError("leList and teList must be the same length if they define the exact projection location")
        le_span_s = le_s.projectPoint(leList)[0]
        te_span_s = te_s.projectPoint(teList)[0]
        # Get the total number of spanwise sections
        nSpanTotal = len(le_span_s)

    else:
        raise ValueError("nSpan must be either None, int, or list")

    # Generate a 2D region of intersections based on cosine spacing

    X = np.zeros((nSpanTotal, nChord, 3))

    le_coord = le_s(le_span_s)
    te_coord = te_s(te_span_s)

    for i in range(nSpanTotal):
        # Set x and z based on cosine spacing along the chord
        X[i, :, 0] = le_coord[i, 0] + (1 - np.cos(np.linspace(0, np.pi, nChord))) / 2.0 * (
            te_coord[i, 0] - le_coord[i, 0]
        )
        X[i, :, 2] = le_coord[i, 2] + (1 - np.cos(np.linspace(0, np.pi, nChord))) / 2.0 * (
            te_coord[i, 2] - le_coord[i, 2]
        )

    coords = np.zeros((nSpanTotal, nChord, 2, 3))
    for i in range(nSpanTotal):
        for j in range(nChord):
            # Generate the 'up_vec' from taking the cross product
            # across a quad
            if i == 0:
                uVec = X[i + 1, j] - X[i, j]
            elif i == nSpanTotal - 1:
                uVec = X[i, j] - X[i - 1, j]
            else:
                uVec = X[i + 1, j] - X[i - 1, j]

            if j == 0:
                vVec = X[i, j + 1] - X[i, j]
            elif j == nChord - 1:
                vVec = X[i, j] - X[i, j - 1]
            else:
                vVec = X[i, j + 1] - X[i, j - 1]

            upVec = np.cross(uVec, vVec)
            # Project actual node:
            up, down, fail = geo_utils.projectNode(X[i, j], upVec, p0, p1 - p0, p2 - p0)

            if fail == 0:
                coords[i, j, 0] = up
                coords[i, j, 1] = down
            elif fail == -1:
                # More than 2 solutions. Returned in sorted distance.
                coords[i, j, 0] = down
                coords[i, j, 1] = up
            else:
                raise ArithmeticError(
                    "There was an error projecting a node at (%f, %f, %f) with normal (%f, %f, %f)."
                    % (X[i, j, 0], X[i, j, 1], X[i, j, 2], upVec[0], upVec[1], upVec[2])
                )

    return coords
