## dfdx_cifar

Classifying the CIFAR-10 dataset using the `dfdx` library. 

```sh
# Build the Docker container for dfdx with CUDA support
$ docker build . -t dfdx:latest
# Mount the working directory into the dfdx container at /home
$ docker run --rm -it -v $(pwd):/home dfdx:latest
```

Tested on Pop!_OS with an NVIDIA 4070Ti