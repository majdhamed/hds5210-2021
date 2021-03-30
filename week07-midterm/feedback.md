Score: 45/50

In your Framingham function, you ended up cutting out one of the calculations for men who were <= 70 and women who were <= 78.  In the blocks of code that looked like this, you needed to have some "else" conditions:

```
        if smoker:
            L=L+12.096316 *1
            if age >70:
                L += -2.84367*math.log(70)
```


Nice attempt on part 4 as well.  Using the csv.DictReader was a good approach.

