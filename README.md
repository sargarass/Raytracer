# Raytracer
The small raytracer project based on "Ray Tracing in One Weekend" using HIP

# How to run (hip-cuda)
1. Download and install nvidia-cuda-toolkit;
2. Set environment variable $ROCM_PATH to the foulder where ROCm software will be stored;
3. Download and install HIP https://github.com/RadeonOpenCompute/hcc to $ROCM_PATH;
4. Download and install HIP RAND https://github.com/ROCmSoftwarePlatform/rocRAND $ROCM_PATH;
5. git clone https://github.com/sargarass/Raytracer.git;
6. cd Raytracer && mkdir build && cd build;
7. cmake.. && make -j$(nproc);
8. run program: ./raytracer;
