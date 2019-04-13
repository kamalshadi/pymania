```
counter = 0
for sub in subjects:
    counter = 0
    for p in pairs:
        if sub[p].is_distance_corrected:
            counter+=1
        else:
            continue
        if 0.5-delta<f(p)<0.5+delta:
            Continue #we cannot decouple variability sources
        if f(p)>0.5+delta:
            if sub.has_the_connection(p)
                true_positive[sub]+=1
            else:
                false_negative[sub]+=1
        else:
            if sub.has_the_connection(p)
                false_positive[sub]+=1
            else:
                true_negative[sub]+=1

    TP[sub] =  true_positive/counter # reported true positive per subject
    FP[sub] =  false_positive/counter # reported false negative per subject
```
