import pymania as mn
import pickle as pk

_id = 'vdense'

subs = '''126426
135124
137431
144125
146735
152427
153227
177140
180533
186545
188145
192237
206323
227533
248238
360030
361234
362034
368753
401422
413934
453542
463040
468050
481042
825654
911849
917558
992673
558960
569965
644246
654552
680452
701535
804646
814548'''.split('\n')

subs = [int(xx.strip()) for xx in subs]


a = mn.create_project('Constantine',_id)
a.backend.connect()
for roi in ['L'+str(xx) for xx in range(1,181)]:
    a.add_roi(roi)


for subject in subs:
    a.add_subject(subject)
print('loading...')
a.load()
print('loaded...')
a.run()

with open(f'{_id}.pk','wb') as f:
    pk.dump(a,f)
