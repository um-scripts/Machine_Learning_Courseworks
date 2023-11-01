def euclidean(x1, x2):
    if len(x1) != len(x2):
        raise AssertionError

    distance = 0
    for index in range(len(x1)):
        distance += (x1[index] - x2[index]) ** 2

    return distance ** 0.5


def leader_clustering(data, dist_fun, threshold):
    if not callable(dist_fun):
        raise AssertionError

    groups = list()
    for x in data:
        flag = False
        for i, g in enumerate(groups):
            leader = g[0]
            if dist_fun(leader, x) <= threshold:
                groups[i].append(x)
                flag = True
                break
        if not flag:
            groups.append([x])

    return groups


data_file = open('data.csv', 'r')
lines = data_file.readlines()
print(lines)

d = list()
for line in lines:
    values = str(line).split(',')
    d.append(tuple([float(i) for i in values]))


clusters_ = leader_clustering(d, euclidean, 3 )
for n, c in enumerate(clusters_):
    print('Cluster {} - '.format(n), c)

