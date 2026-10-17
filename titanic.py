import numpy as np

x = 5
name = 'Аня'
nums = [3, 1, 4, 1, 5]
person = {'name': 'Аня', 'age': 19}
print(type(x), type(name), type(nums), type(person))
print(nums[0], nums[-1], len(nums), sum(nums) / len(nums))
print(person['name'])


print('-------------------------------------------------')

def mean(lst):
    total = 0
    for v in lst:
        total += v
    return total / len(lst)
 
def minmax(lst):
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo: lo = v
        if v > hi: hi = v
    return lo, hi
 
print(mean([3, 1, 4]), minmax([3, 1, 4]))

print('-------------------------------------------------')

with open('titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]
print(header)
print(len(rows), 'строк')
print(rows[0])

print('-------------------------------------------------')

cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)
print(cols['age'][:10])

print('-------------------------------------------------')

for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')

print('-------------------------------------------------')

ages = np.array(cols['age'])
stats = ages.mean(), ages.min(), ages.max()
print('NUMPY TITANIC!!!!!!!!!!!!!!!', stats)
