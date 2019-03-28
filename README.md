This is a simple replication of [PyTorch](https://pytorch.org/) using [numpy](http://www.numpy.org/) and [autograd](https://github.com/HIPS/autograd). PyTorch has a very nice API especially the dynamic computation graph makes it so flexible to use and easy to debug. However, understanding how dynamic computation graph works can be difficult by reading the source code, as it runs heavy c++ backend code for better performance. This project is to help someone who would like to understand what is behind the scene without reading the complicated c++ code, as well as those who wants to know how deep learning framework works in general.

Branch bintorch-jax uses [Jax](https://github.com/google/jax) which is a better version of Autograd which can be used on GPU, note this branch is work in progress.

To run the example code by 

`python setup.py develop`

`python ./examples/main_feed_forward.py`