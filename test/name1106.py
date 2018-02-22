girls=['alice', 'ber', 'clarice']
boys=['chris', 'arno', 'bob']
letterGirls={}
for girl in girls:
    letterGirls.setdefault(girl[0], [girl])
print('hello')
print([b+'+'+g for b in boys for g in letterGirls[b[0]]])
