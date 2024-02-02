dim3 block(x,y,z);
dim3 grid(u,v,w);
foo<<<grid, block>>>()

globalID = 
```
t.x + t.y * bdim.x + t.z * bdim.x * bdim.y +
bdim.x * bdim.y * bdim.z * b.x + 
bdim.x * bdim.y * bdim.z * gdim.x * b.y +
bdim.x * bdim.y * bdim.z * gdim.x * gdim.y * b.z 
```


- `__device__` : A function that can only be called from within a kernel, i.e. not from the host.  
- `__host__` : A  function that can only run on the host. The `__host__` qualifier is typically omitted, unless used in combination with `__device__` to indicate that the function can run on both the host and the device. Such a scenario implies the generation of two compiled codes for the function. Can you guess why?
- `__host__ __device__`: can be ran on both host and device
- `__global__` : can be called from the host and executed on the device. In CC 3.5 and above, the device can also call `__global__` functions, using a feature called dynamic parallelism.

`__syncthreads();`:  Block-wide barrier
`atomicAdd(int* address, int val) // location to modify, value to add`

Race conditions between warps of the same block exist