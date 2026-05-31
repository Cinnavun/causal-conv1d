# Causal depthwise conv1d in CUDA with a PyTorch interface

Features:
- Support fp32, fp16, bf16.
- Kernel size 2, 3, 4.

## How to use

```python
from causal_conv1d import causal_conv1d_fn
```

```python
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
```

Equivalent to:
```python
import torch.nn.functional as F

F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]
```

## Additional Prerequisites for AMD cards

### Patching ROCm

If you are on ROCm 6.0, run the following steps to avoid errors during compilation. This is not required for ROCm 6.1 onwards.

1. Locate your ROCm installation directory. This is typically found at `/opt/rocm/`, but may vary depending on your installation.

2. Apply the Patch. Run with `sudo` in case you encounter permission issues.
   ```bash
    patch /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h < rocm_patch/rocm6_0.patch 
   ```
# causal_conv1d is installed on Windows (compiles itself on MSVC)
# Compile yourself on Windows (MSVC)
## 1. Create a compilation environment:
    Download Windows 11/10 SDK using Visual Studio and download the MSVC compiler
    Configure environment variables
    Install ninja
## 2.  Adapted to causal-conv1d source code

    clone This project modifies the setup.py
        
        "``python
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86") #Select your own graphics card architecture for compilation, reducing compilation time and generated file size
        ```

    ### If it is an AMD graphics card, you need to
    1. causal-conv1d\csrc\causal_conv1d.cpp 
    2. causal-conv1d\csrc\causal_conv1d_fwd.cu
    3. causal-conv1d\csrc\causal_conv1d_bwd.cu
    4. causal-conv1d\csrc\causal_conv1d_common.h 
    Middle
        #ifndef USE_ROCM
        æse
        #endif
    Change to keep the code of else
## 3.  Compile code
    #Use x64 Native Tools Command Prompt for VS 2022
    
    # 1.  Tell the compiler that we want to use the SDK
    set DISTUTILS_USE_SDK=1

    # 2.  Limit compilation to only 1 thread, although it is slower but will not burst memory
    set MAX_JOBS=1


    # 3.  Officially start installation
    pip install . --no-build-isolation