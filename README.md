# Disease Prediction
Gives the probable diagnosis after entering the symptoms

## Model
The data set has 132 symptoms which lead to the cause of 41 diseases.
### RFC
```
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)
```
### Accuracy
The model has an accuracy of **97.62%**


## Flask
### POST method
```
    if request.method=='POST':
        col=x_test.columns
        inputt = [str(x) for x in request.form.values()]

        b=[0]*132
        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
       
```
## HTML
### Jinja
```
{{pred}}
```
### Page 🍕 
> ![wp blank](https://user-images.githubusercontent.com/68746915/99890950-ab8a2b00-2c8a-11eb-8181-6f02811030d8.png)

> ![wp entered](https://user-images.githubusercontent.com/68746915/99891453-cc08b400-2c8f-11eb-8f2f-1ef1513aa681.png)

> ![wb diag](https://user-images.githubusercontent.com/68746915/99891551-20f8fa00-2c91-11eb-9a68-eb18800563d3.png)

## Try it Yourself!
https://disease-prediction-pip.herokuapp.com/
