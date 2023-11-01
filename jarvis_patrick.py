def euclidean(x1, x2):
    if len(x1) != len(x2):
        raise AssertionError

    distance = 0
    for index in range(len(x1)):
        distance += (x1[index] - x2[index]) ** 2

    return distance ** 0.5


def jarvis_patrick(data, k, k_min, function):
    data = set(data)
    if k >= len(data) or k_min > k:
        raise ValueError
    if not callable(function):
        raise AssertionError

    knn = dict()
    for x in data:
        temp_data = list(data)
        temp_data.remove(x)
        sorted_data = sorted(temp_data, key=lambda y1: function(x, y1))
        knn[x] = set(sorted_data[:k])

    clusters = list()
    clustered = dict()
    for x in data:
        clustered[x] = False

    for x in data:
        if clustered[x]:
            continue
        cluster = list()
        cluster.append(x)
        clustered[x] = True
        for y in data:
            if clustered[y]:
                continue
            if y in knn[x] and x in knn[y] and len(knn[x] & knn[y]) >= k_min:
                cluster.append(y)
                clustered[y] = True

        clusters.append(cluster)

    return clusters


data_file = open('data.csv', 'r')
lines = data_file.readlines()

d = list()
for line in lines:
    values = str(line).split(',')
    d.append(tuple([float(x) for x in values]))


clusters_ = jarvis_patrick(d, 3, 2, euclidean)
for n, c in enumerate(clusters_):
    print('Cluster {} - '.format(n), c)