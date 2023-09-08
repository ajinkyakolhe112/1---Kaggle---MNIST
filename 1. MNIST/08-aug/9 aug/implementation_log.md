
Problem
- Digit Classification Ability
Data
- 8bit Black & White Data in 28*28 pixels
- data in csv file


**Steps to Complete**  
- [x] Data in csv -> 
- [x] Data in Dataloader ready to be trained -> not increamental coding, and 1 logical error , np matrix didn't visualize properly for one hot encoding
- [x] Model Architecture - Simple architecture
- [x] Model Training - pytorch lightning makes it easier
- [x] Model Monitoring - 
- [ ] Model Prediction
- [ ] Submission

**Parametric Experiment**
1. Input -> Dense(**13**) -> Dense(10) -> Output
2. 15 neurons. max: 75% accuracy. still more than 50% accuracy.

Experiment: 11 neurons
Observation
1. still accuracy 70+ percent

Experiment: 8 neurons
1. still reaches 80% accuracy. 
2. **ignore outliers in chart scaling** very important option in tensorboard.

- 5 neurons. Still 70% accuracy. and sharp increase in accuracy in first few batches.
- 3 neurons. 50% accuracy by 1st epoch and 1182 steps/batches
- 2 neurons. 30% by epoch 1. platue at 60% accuracy. **JUST LEARNING**
- 100 neurons. almost 100% accuracy
- 100 neurons and 1/2 Data. almost 95% accuracy
- 100 neurons and 1/4 Data. almost 90% accuracy
- 50 neurons  and 1/4 Data, 70% accuracy platue
- 50 neurons  and 1/2 Data, almost 95% accuracy