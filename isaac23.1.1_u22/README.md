
Nvidia is not publishing ned images for cudagl, they say someday they will resume.
In the meantime we have to build images ourselves.
This can be done by cloning their repo:

```
git clone https://gitlab.com/nvidia/container-images/cuda.git
```

And then building the image with:

```
./build.sh -d --image-name cudagl --cuda-version 12.3.1 --os ubuntu --os-version 22.04 --arch x86_64 --cudagl
```
