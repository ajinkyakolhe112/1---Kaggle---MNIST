

Summary of Yesterday  
- `keras` good for easy coding. but subtle regression bugs or design bugs, lead to lots of time wasting
- `keras` has good functions, which are useful. Need specific libraries in pytorch
- `pytorch + skorch + lightning`: do same thing as keras
- `pytorch`: least api change. but everything in one place, unlike seperate modules in keras


Design Doc  
1. `model architecture + training + monitoring`
2. `dataloader`. needed to look up, `torch.utils.data.Dataset` & `torch.utils.data.DataLoader`
   1. used TensorDataset. Dataset is an abstract class
3. `models.py`. training with help of pytorch lighning. almost same as skorch



