Libraries
- Keras 2.0 (Mar 2017), TF 2.0 (Sep 2019 + keras), 
	- Keras, TF vs Pytorch Fight
	- TF Pros: Compile & Speedup, Distributed Tensor
	- Keras 3.0 (July 2023, FUTURE FOCUS) with Jax, Pytorch, TF cross functional
- Pytorch 2.0 (Mar 2023) with compile
	- API has been stable
	- mps backend, scaled dot product attention 2.0, torch.func like jax vmap
	- Pytorch lighning makes training easier


Datasets
- `torchvision.datasets`: a few important benchmark datasets
- `tf.keras.datasets`: a few toy datasets (already-vectorized, in Numpy format) that can be used for debugging a model or creating simple code examples
- `tf.data.Dataset NOT DATASETS`: Tensorflow input Pipeline
- `pip install tensorflow-datasets`: Download & prepares data from large & public datasets using tf.data.Dataset
`import tensorflow_datasets as tfds`

Dataset
- `tf.data.Dataset`
- `torch.utils.data.Dataset`

torch.nn
vs from tf.keras import datasets, preprocessing, models, layers, losses, metrics, optimizers

Datasets collections. (TF 2.0, Keras 2.0, Keras 3.0 with less TF)
- `tf.keras.datasets.__dir__()`
- `tensorflow_datasets.list_builders()`
- `torchvision.datasets.__dir__()`
- `torchaudio.datasets.__dir__()`
- `torchtext.datasets.__dir__()`
