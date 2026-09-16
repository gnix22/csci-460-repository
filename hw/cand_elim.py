import math
# positive and negative points
positive_points = [(0, 0), (1, 0), (0.5, -0.5)]
negative_points = [(2, 3), (4, -1), (-1, 4), (-3, 1)]
# the allowed intervals
a_values = range(-5, 6)
b_values = range(-5, 6)
r_values = range(1, 11)
consistent_hypotheses = []
# just loop through every permutation (a, b, r) in the given intervals
for a in a_values:
    for b in b_values:
        for r in r_values:
            # check every positive point is inside the circle
            covers_all_positives = True
            for (x, y) in positive_points:
                distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)
                # distance of candidate circle larger than radius means points aren't covered
                if distance > r:
                    covers_all_positives = False
                    #print(f"[({a},{b}),{r}] doesn't cover all points, inconsistent hypothesis.")
            # check every negative point is outside the circle
            avoids_all_negatives = True
            for (x, y) in negative_points:
                distance = math.sqrt((x - a) ** 2 + (y - b) ** 2)
                if distance <= r:
                    avoids_all_negatives = False
                    #print(f"[({a},{b}),{r}] doesn't avoid all points, inconsistent hypothesis.")
            if covers_all_positives and avoids_all_negatives:
                consistent_hypotheses.append((a, b, r))
print("out of: ", len(a_values)* len(b_values) * len(r_values), "hypotheses, consistent circles found:", len(consistent_hypotheses))
print(consistent_hypotheses)
# now find the "smallest" circles: ones not fully contained in another
most_specific = []
for (a1, b1, r1) in consistent_hypotheses:
    is_smallest = True
    for (a2, b2, r2) in consistent_hypotheses:
        if (a1, b1, r1) != (a2, b2, r2):
            center_distance = math.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)
            # is circle 1 bigger than circle 2? (circle 2 fits entirely inside circle 1)
            circle1_contains_circle2 = center_distance + r2 <= r1
            circle2_contains_circle1 = center_distance + r1 <= r2
            if circle1_contains_circle2 and not circle2_contains_circle1:
                is_smallest = False
    if is_smallest:
        most_specific.append((a1, b1, r1))
print("most specific circles (S):", most_specific)
# now find the "biggest" circles: ones not fully contained inside another
most_general = []
for (a1, b1, r1) in consistent_hypotheses:
    is_biggest = True
    for (a2, b2, r2) in consistent_hypotheses:
        if (a1, b1, r1) != (a2, b2, r2):
            center_distance = math.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)
            # is circle 2 bigger than circle 1? (circle 1 fits entirely inside circle 2)
            circle2_contains_circle1 = center_distance + r1 <= r2
            circle1_contains_circle2 = center_distance + r2 <= r1
            if circle2_contains_circle1 and not circle1_contains_circle2:
                is_biggest = False
    if is_biggest:
        most_general.append((a1, b1, r1))

print("most general circles (G):", most_general)
