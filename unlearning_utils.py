def get_unlearning_data(datalist):
    min = 99999999
    idx = 0
    for i, data in enumerate(datalist):
        if min > len(data[0]):
            idx = i
    res = []
    idx = 4
    for i, data in enumerate(datalist):
        if i == idx:
            continue
        else:
            res.append(data)
    return datalist[idx], res