
### dataset.py
Observation & Live Thinking  
1. Data present in ../input/digit-recognizer/train.csv
2. Data is saved as pixel values in columns
3. Read in numpy - numerical analysis of data? or keras - nice naming in nn ? pytorch - just nn, low level library? pandas - no column wrangling?
4. keras would make nn writing via proper modules & training easy.. could override methods & use keras just as pytorch. 
5. pytorch, easiest debugging. keras makes coding easier, but actual nn code is executing in backend. makes custom debugging harder. but for simple architectures, keras would be better.
- [ ] code same code in keras & pytorch. to compare. also use skorch. will read data in numpy. and then send to keras & pytorch accordingly.

### models
1. wrote first architecture in keras
2. could write different models in folder.
3. dir(keras.get*) (important function always)
4. `keras.model.get_config()` and `keras.model.get`
