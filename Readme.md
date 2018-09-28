# JMusical

JMusical is a Plugin for Imagej which implements [MUSICAL](https://arxiv.org/abs/1611.09086), an algorithm
created by [Krishna Agarwal](https://sites.google.com/site/uthkrishth) for obtaining super-resolution
based on Eigenvalue decomposition.

This plugin works for a single channel image of undefined number of frames,and it is based on author's original Matlab 
version published [here](https://drive.google.com/file/u/1/d/0B03nGjisITftNGxzeE5feFp1OXM/view?usp=sharing).

## About JMusical 0.9

The original Matlab code was analized and optimized to decrease the number of operations. The 
main differences are:

- The matrices are stored as float instead of double to increase computation speed
- The image is divided in blocks to be processed by different threads

Beside that, the process is identical in the sense that each region is processed in a
serial fashion, and the result is generated when all the regions have been analized.

The tests show a difference no greater to 1e-4 with the values computed in Matlab.

The current version allows to autosave the results in addition with a text file
indicating the parameters used.

## How to install

In ImageJ / Fiji:

- Go to Help>Update
- Click on "Manage update site"
- Add 'http://sites.imagej.net/Sebsacuna/' to the list
- Download all the dependencies

All matrices operations are based on Nd4j, and in the sites it is included the MKL dependencies
also. These dependencies are quite heavy. Among the other dependencies it is `javacpp-1.4-2` 
which replaces the `javacpp-0.11` included in Fiji.

## How to use 

The interface allows the user to enter all the required values. The first block
correspond to the optical parameters of the system used for taking the image. The
second block correspond the parameters the user need to play with.

The normal process is the following:

- Enter the optical parameters and plot the singular values
- Analyze the curves and select an horizontal line as threshold
- Enter desired threshold and alpha
- Generate image

Extra options are presented:

- Multithread: allows the program to split the image and process it by parts using
threads. The number of threads recommended is the number of cores availables in the 
machine
- Save: saves the result image automatically after finished, including a text file
with the used parameters. These files are stored in the same folder of the input
image

## Multithreading

This option speed up the computation by an order of 2x or 3x, depending on the machine.
Increments the usage of the CPU so can slow down other task being made on the machine.

Because of the summation of multiple floats in a different order, the values can 
differ from the 1 thread version. However this difference should not be bigger than 
0.01\%.

## Batch mode

To process several files, the best option is to use a Macro. However, the current plugin 
does not work well with the Batch processing included in Fiji, so take care.

### Examples

TODO

## About the author

For any problem or comment, please don't hesitate on contact me at `sebacunam@gmail.com`


