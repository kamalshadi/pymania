import pymania as mn
import pickle as pk

_id = 'vdense'

a = mn.create_project('Constantine',_id)
a.backend.connect()
for roi in ['R'+str(xx) for xx in range(1,181)]:
    a.add_roi(roi)


for subject in :
    a.add_subject(146735)
a.load()
a.run()

with open(f'{_id}.pk','wb') as f:
    pk.dump(a,f)
