
### Dataset
1. saw data. in csv with pixel values.
2. log (min(data) - max(data)) / log 2 -> give the power raised to 2 for this index.
   1. 8 bits per color. 24 bits for 3 colors.
   2. 256^3 = 16.77 million colors. Human vision sensitive to 10 million colors
   3. Color Blind. (8% men are color blind. 0.5% women are color blind). 256^2 colors. 
   4. ![](https://www.colourblindawareness.org/wp-content/themes/cba/images/slider/types/eye.jpg)
   5. 4 color cones. 256^4. They can see 256 shades of this any single color we see. for every such color we can see, they can see 256 more shades of it..
3. memorize: powers of 2 & values. will be useful


color blindness & its consequence in modern life. 
mirror neurons & autism. its impact on brain's NN learning. 
neurotransmitter - dopamine & adhd. its imact on brain's NN activation.
neurotransmitter - dopamine & serotonine & depression. its impact on brain's NN activation. 
eye floaters
blindspot
same smell being cancelled out (what's happening in brain)
same audio doesn't cancel out. (not a feature built in brain)

----
keras, easier to build nn. & iterate on it. just num of neurons for layer. **faster iteration, changing neuron configuration in multiple layers, needs change only at 1 place**
keras also needed because of tf javascript
pytorch. in_channels & out_channels for layer. need to always change two numbers. iterations have one more step change. but in keras, you can change number easily. 
so iterate on keras and then move to pytorch. 
- in keras, lots of things are happening in the background. batch = 32. what is loss function, what is optimizer. what is actually happening. hence we need to move to pytorch, which is a low level framework.
even in a low level framework, you can simplify the common repeated codes as library, pytorch lightning, torchsummary, torchinfo

with pytorch lightning, pytorch code is as easy as keras. also control over forward loop. can get details during each step. 

- [ ] doing, mnist & cifar at same time. also learn cnn-explainer. combined with keras & pytorch
- [ ] alexnet size but for 10 categories like cifar. This will allow to do good experiments & understanding of real life data
----

Code Design Questions
Reshape in pandas vs numpy vs torch
- reshape during data processing stage should be done with pandas or numpy
- reshape once in DL modelling, should be done with torch
  

num_classes vector encoding in pandas vs numpy vs torch
- pandas: `pd.get_dummies`
- pytorch: `torch.nn.functional.one_hot`

### Model
1. Check documentation of each layer as transformation function. And go deeper each time. (documentation -> source code -> your own code -> maths latex)