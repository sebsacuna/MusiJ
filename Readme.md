# MusiJ

Note: From November 11 2019, the project has been renamed to MusiJ

MusiJ is a Plugin for Imagej which implements [MUSICAL](https://arxiv.org/abs/1611.09086), an algorithm
created by [Krishna Agarwal](https://sites.google.com/site/uthkrishth) for obtaining super-resolution
based on Eigenvalue decomposition. This is a list of some of the characteristics of the algorithm.

1.	Compatible with any dye or fluorescent protein in theory. Tested on Alexa dyes, GFP, RFP, YFP, CMP, SirTubulin, SirActin, MitoTracker dyes, etc.
2.	Compatible with dense or sparse samples and uses natural fluctuations in fluorescent intensity. Tested for cells and tissues without using any imaging buffer (i.e. redox solutions).
3.	Requires low power in comparison to most techniques, therefore less photo-toxic especially for live cells. 
4.	Requires very few frames (50 – 200 are sufficient in most cases), therefore suitable for dynamic systems such as live cells.
5.	Tested on variety of cameras, objective lenses (0.4 NA 20X to 1.49NA 100X oil immersion), and multi-channel acquisition (4 channels so far).
6.	Works with TIRF and epifluorescence x-y-t image stacks.

This is an example of the algorithm:

![Example](https://i.imgur.com/aoYdQg6.png)

This plugin works for a single channel image of undefined number of frames,and it is based on author's original Matlab 
version published [here](https://drive.google.com/file/u/1/d/0B03nGjisITftNGxzeE5feFp1OXM/view?usp=sharing). 

## About JMusical 0.9x

The original Matlab code was analized and optimized to decrease the number of operations. The 
main differences are:

- The matrices are stored as float instead of double to increase computation speed
- The image is divided into blocks to be processed by different threads

Beside that, the process is identical in the sense that each region is processed in a
serial fashion, and the result is generated once all the regions have been analized.

The tests show a difference no greater to 1e-4 with the values computed in Matlab.

The current version allows us to autosave the results in addition with a text file
indicating the parameters used.

## MusiJ 0.93

- Faster by a factor of 2 compared to previous version
- Video wrapper included for video-generating capabilities


## Instructions

We have released a video tutorial in Youtube with the instruction of how to install and use MUSICAL in FIJI. You can watch
it [here](https://www.youtube.com/watch?v=CsJHqSQb11E) or by clicking on the image:

[![JMUSICAL VIDEO TUTORIAL](https://i.imgur.com/45ExO8b.png)](https://www.youtube.com/watch?v=CsJHqSQb11E "JMUSICAL VIDEO TUTORIAL")

The data used and parameters can be found in the folder `data`.

## How to install

In ImageJ / Fiji:

- Go to Help>Update
- Click on "Manage update site"
- Add 'http://sites.imagej.net/Sebsacuna/' to the list
- Download all the dependencies

All matrices operations are based on Nd4j, which can use MKL if available. These dependencies are quite heavy.
Among the other dependencies it is `javacpp-1.4-2` which replaces the `javacpp-0.11` included in Fiji.

## How to use 

The interface allows the user to enter all the required values. The first block
correspond to the optical parameters of the system used for taking the image. The
second block corresponds to the parameters the user need to play with.

The normal process is the following:

- Enter the optical parameters and plot the singular values
- Analyze the curves and select an horizontal line as threshold
- Enter the desired threshold and alpha
- Generate image

Extra options are presented:

- Multithread: allows the program to split the image and process it by parts using
threads. The number of threads recommended is the number of cores availables in the 
machine
- Save: saves the result image automatically after finished, including a text file
with the used parameters. These files are stored in the same folder of the input
image

## Multithreading

This option speeds up the computation by an order of 2x or 3x, depending on the machine.
Increases the usage of the CPU so it can slow down other tasks running on the machine.

Due to the summation of multiple floats in different orders, the values can 
differ from the 1 thread version. However this difference should not be higher than 
0.01\%.

## Batch mode

To process several files, the best option is to use a Macro. However, the current plugin 
does not work well with the Batch processing included in Fiji, so careful.

### Examples

TODO

## About the author

For any problem or comment, please don't hesitate on contact me at `sebacunam@gmail.com`


