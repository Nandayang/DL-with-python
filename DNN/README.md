# *Deep neural Networks*
In this part ,we implemented DNN to solve the "is or not cat" problems.
## Environment
python--3.6   
numpy --1.13.3  
h5py  --2.7.0  
### Details
In this part, you can use your own dataset and change the network's architecture through changing the *layer_dim*.The discription of layer_dim is as below:  
If you have a dnn with input images(32,32,3), then the first parameters in layer_dim should be 32\*32\*3, the second parameters in layer_dim is the number of nodes in the first hidden layer, and the depth of model is *len(layer_dim)*.  
(Note that we don't set the input as the first layer)
