# ViT Acceleration Report
**BASELINE:** Open-Vit-Bench
**COMPARISON:** da definire(o l'implementazione di pytorch(?))
**CONTRIBUTION:** ViT in CUDA con librerie

Lavora in fp16, valuta se passare ad un altra precisione(se c'è tempo)

PER LE DIVISIONI E LE MULTIPLICAZIONI, E' NECESSARIO, VISTA LA VERSIONE DI CUDA E LA SCHEDA USATA, PASSARE A FP32 e poi tornare indietro

# General Idea


I need something for the Mat-Mul, but then a flag to decide what version utilize.
So two possible paths:
1. Mat-Mul kernel or full kernel => boolean flag
2. Modularity, for each block decide what piece "accelerate" => more flags but ensure the type consistency across all the branches

## Memory handling
For every kernel we have to say which type of memory handling is done, a guideline can be:

1. How much storage is used on device and on the host
2. How many memory transaction we do

## Dependencies

- Base project
- cuBLAS
- `helpers.h` comes from cudnn FE library

## Structure

Inside `gpu_src` there are the files in cpp that implements the accelerations of the elements


# Implementations
cuBLASLt API calls use a hand set workspace size to perform better, 4MiB is the same as the one of cuBLAS.


## 1. Patch Embedder 

The patch embedder from the slides is wrongly represented, because it first do the computation and then it flatten the tensor

### 1.1 Convolutional Patch embedder (conv2d)
Implementing the convolution in CUDA: **I suppose that we always use 16x16 patches so the kernel is for 16x16 patches**

The standard forward is a sequential computation - *cpp baseline*

1. *Single Channel convolution*: kernel for a single channel single patch line convolution. Time comparison. 
   1. NO DATA PREPROCESSING
   2. NO MAJOR MEMORY USAGE 
   3. B x O_C x P_N x C x P_H kernel launches (a lot)
   4. HOST SYNC, after every kernel launch
2. *Unified Channel Convolution*: better than launching C * P_H separate kernels, less kernels launched and shared resources but **need data sorting**.
   1. YES DATA PREPROCESSING
   2. NO MAJOR MEMORY USAGE
   3. B x O_C x P_N kernel launches
   4. HOST SYNC, after every kernel launch
3. *Parallelized Convolution*: (unified) kernel launched in parallel for each out channel and so patch
   1. YES DATA PREPROCESSING
   2. MAJOR MEMORY USAGE, saving the results in memory and then copying back to host only once
   3. B x O_C x P_N but parallelize on N streams
   4. NO HOST SYNC, on final memcpy at the end
4. *Data transfer reduction*: Save everything on GPU, then do 1 final memcpy to Host. DOUBLES THE MEMORY USAGE.
   
**STREAMING AND CUDA GRAPH** 
Streams check, single, fixed number, and GPU number. 

**Plots**
For the image size all levels lineplot
For the batch size level 1 and 2 and varying the stream n, 0 is too slow

> [!Note]How to Run
> ```bash
> bubu
> ```


## 2. Encoder Block 

Accelerate the Patch Embedder and the Self Attention, including the eventual manipulations like reductions in the blocks.
Parallelize basically the entire block

All the acceleration are made through cuBLAS or cuDNN, in particular:
*Attention*: **cuDNN** (1 Kernel)
*MLP*: **cuBLAS** (2 Kernels)
*Layer Norm*: **CUB**  (**cuDNN** not working) (4/5 kernels)

For the encoder block tests I'll be testing both numerically and performance wise:
- The fused_mlp and a transpose after it
- The hand written epilogue, gpu_mlp which doesn't require the mtx. transpose

*Encoder Block schema img*

### Versions
- 

### Improvements
1. **Buffer reuse**: optimize the buffers used during each transition of the data
2. **Cuda Graph**: instatiate a cudaGraph for reducing kernel launch overhead
3. **Evaluate streaming**: evaluate if streaming kernels might be beneficial or not(optional, report it only)
4. **More fusion**: Between the mlp the transpose and the LayerNorm
5. **Everything in col-major**: due to the cuBLASLt fused GEMM + Epilogue, ops have to be in col-major, but the rest of the block is in row-major format, so a transpose op have to be made in order to get the correct results

### 2.1 Layer Norm

The LayernNorm I'm gonna implement is BlockScoped, this for ensuring no global mem accesses other than the minimal needed and for the problem I'm confronting is more than sufficient(Elements are way less than the shared mem maximum size per block)



cuDNN Frontend NormForwardTraining.
I've tried, first the FE then the BE, but the FE was giving different results and the BE doesn't work in my desired config, so I proceded to implement manually a layer norm, it will not be probably as fast as cuDNN but I will use different techniques for achieving better and better results:
1. Base gpu implementation
   - Shared memory
   - Fused Kernels
   - Half precision
   - two elements per thread for better load balancing
2. CUB reduction
3. Sequence Tiling
   - Using Sh. mem for reusing bias and scale across different tokens
4. More elements per thread
5. Loop unrolling

This kind of operation is memory dominated, once optimized the kernel is important to exploit asyncronous allocation and memory copy for improving performance! **(TO DO!!)**

Be careful some specific configs accelerate the Computation (see DOC)(check if feasible for my application)



### 2.2 Self-Attention

cuDNN has SOTA implementation of self-attention, limited only to the 8.9.2 version (baldo cluster at dev time)

- **Fused Forward MHA**
- **Fused Flash Attention (MHA)**
- **FP8 Fused Flash Attention** (not supported)

1. **Fused Forward MHA**
   - Using half as compute precision for the 768x768 matrices with sufficiently big values returns nan, so also here accumulating in float ensures that the results returned are correct.

### 2.3 MLP

Two layer MLP, is made of GEMM -> ACTIVATION (GELU) -> GEMM

*cuDNN*
Epilogue fusions for bias and gelu not supported (yet, seems in future may be supported)

*cuBLASLt*
I've tried to use the fusion engine but the pointwise add op was not working correctly, on the doc it says that is possible to broadcast two dimensions but in my case only one was actually broadcasted.

The bias epilogue has to be col-major, but to do so my input data should have different shapes, from both row-major to both col-major

Another important feature of newer CUDA VERSION is that they offer a device side library that adds the possibility of using Nvidia kernels fused with hand written CUDA kernels!!


So, think of 2 implementations:
1. No fused CublasLt pipeline of GPU ops (GEMM -> BIAS + GELU -> GEMM)
   1. Using CudaGraph to capture the operations, require to capture one time the operations
   2. Fusing the residual connection epilouge (TO DO)
2. Fused Bias and GELU in the first GEMM (GEMM + BIAS -> GELU)
   1. Two trips to the global mem saved

> [!Warning]
> cuBLAS is doing a Memset probably on the workspace, due to the fact that I'm recycleing it.

**1**

**2** 
The second one should be faster, because in the first I have to access values from the global mem. 3 times(write: GEMM, BIAS + GELU, GEMM), instead the second approach only 2. With Autotuning is slower probably for the host time spend to find a suitable algorithm for the epilouge.

Preallocating the algoritm doesn't seem to produce a significant improvement. I'm observing a predefined time to launch the "globalKernel" (launched from cuBLAS) that is relevant (and slower) with small batch sizes (< 64). This becomes irrelevant when the batch size scale and the second method is faster.
 
There can be different reasons behind this, for example the doc says that the heuristic returns the best algorithm for the descriptors, but is a heuristic is not assure that is actually the best.
Also I was not able to properly put at work the broadcast of cuBLASLt on the C mtx (A*B + C), that should spare even more time due to the less data to move for the bias add.


**GELU**: On the cluster, i don't know why but there isn't the htanh implemented in the cuda_fp16.h and .hpp (the doc said that should be there), so I'm putting it in the cuda_utils


---


# Benchmarking
The benchmarking have been made on the component level and on the overall ViT.

*Components*; 
- Convolutional Layer (cuDNN)
- Layer Norm (CUB)
- Attention Layer (cuDNN)
- MLP (cuBLAS)

For each component, I made a test file for comparing the CPU reference and check the numerical correctness.

```bash
make test_gpu_<component>.exe 
# or
make test_cudnn_<component>.exe
```

Then the benchmark file computes the average time and variance of each application

*ViT*:
- Patch embedder
- Encoder block
- ViT

 

> [!Note]
Each test has been conducted with 5 warm up and 10 effective runs, averaging the results.





---
DEPRECATO
## Mat-Mul

1. *Pure Mat-Mul*: Mat-mul/operations kernels elapsed time and comparison with CPU (the one at the "lowest" level of parallelization). **Do one implementation that compares the smallest part of computation, with inefficient Streaming**
2. *Unified/Fusion Mat-Mul*: Mat-mul scaled on the maximum range possible in one kernel.
3. *Kernels parallelization:* Parallelize bigger routines(made of more independent kernels) for each component. **Schedule many parallel kernels executions and see eventually bottlenecks in computations**
4. *Blocks fusion:* Put together more blocks of ViT in unified kernels. **Streaming & memory reusage**

--- 

## Metrics

The principal metrics are **BANDWITDH** and **FLOPs**
Other possible metrics are:
- Memory usage
- GPU usage

> **STREAMING** 
> Streams benchmarks: single, fixed, and GPU number of streams.

> **KERNEL LAUNCHES**
> The less the better? Maybe having too few kernel launches(where I have for example some kernels with points where single threads are used and a lot of resource wasting).

> **Memory Usage**
> How much memory does every implementation need? does it sped up the performances?

> **Preprocessing**
> Does the preprocessing have a positive impact on the overall routine?

## Graphs

> **BATCH** 
> Change the number of batch keeping fixed the image size

> **IMAGE SIZE** 
> Change the image size keeping fixed the batch n (to 1)

> [!Note]How to Run
> Launch the runs with sbatchman on the cluster
> ```bash
> TO DO
> ```
> Generate the json from sbatchman
> ```bash
> python3 scripts/generate_json.py <block_flag>
> ```


LibDNN 
ReductionShuffle
post training quantization
Tegra 

25 Agosto

# 22/08/25

**Plot di speedup**
Buona alternativa quando il lineplot è incasinato

Lanci lo stesso input, NCU

**ROOFLINE**
Sincronia non si vede dal roofline model

Avrai su un valore di x, lo stesso kernel con modifiche attorno ad esso.
Diversi Kernels (quindi con diverso numero di bytes spostati) staranno su diversi punti dell'asse X

Per vedere la sincronia, hai bisogno di un plot diverso, esempio:
-  Una versione sincrona con il gannt su una linea sola
-  Una versione asincrona con la distribuzione su più streams

**DIFFERENZIA PER KERNELS e PER ROUTINE**

ROUTINE: HtoD, STREAMING, PREPROCESSING ecc ecc
KERNELS: MEMORY ACCESS, CACHING, OPERATIONS TYPE

## 25 Agosto

Presentazione, focus sulla direzione che vuoi prendere, cosa hai fatto cosa vuoi mostrare e su cosa vuoi lavorare.




