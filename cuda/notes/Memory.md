| Variable declaration                     | Memory   | Scope  | Lifetime    |
| ---------------------------------------- | -------- | ------ | ----------- |
| int LocalVar;                            | register | thread | thread      |
| __device__ __shared__   int SharedVar;   | shared   | block  | block       |
| __device__              int GlobalVar;   | global   | grid   | application |
| __device__ __constant__ int ConstantVar; | constant | grid   | application |

![[schemamemoria.png]]
# Registers
- Automatic variables (no decorator)
	- Arrays are stored in local memory
		- Thread specific
		- Off-chip(?)
- Very many
- Assuming that we have a GPU with CC 2.0, hence 32k 32-bit registers.
- If a kernel requires 30 registers/thread, and our execution configuration had blocks of 256 threads, we could have `floor(32768/(256*30))) = 4 resident blocks` resident blocks, and `256*4=1024 resident threads per SM.`
# L1
- `__shared__`
- cudaFuncSetCacheConfig specifies how much to programmer and how much to cache
- Streaming Multiprocessor localized
- Cleared after block terminates
- 128 byte access
Dinamically allocate shared memory for arrays. Extra parameter in <<<b,t,allocated_size>>>.
If more then on array is to be specified, allocate memory for all the arrays and pass the number of elements of each. Then set up pointers to type to the start of that array's partition
![](https://lh7-us.googleusercontent.com/5wRJY5G-97PNutSg3L2DW02gT10_dElBUhuDmMnNkZn8v-IANZ-3MUisRCQu4-zkNsFMU5VZaBxzyrTvBrrEGvlBDrxNYk_pvpi_MAn1EYN40tKAdpbHEuLYD35a5P7YZV9YElGwDzN5AIc3x_FDrg=s2048)
# L2
- `__device__`
- Global (GRID)
- 32 byte access
- Transparent

# Global memory
Big memory, several clocks slow
`__device__`
# Coalescence
Transactions are 32/64/128 bytes at a time. When a warp accesses global a memory block can be read at once. Data should not cross two consecutive memory blocks
- L1
	- 128 bytes
- L2
	- 32 bytes
- Global
	- 32 bytes
## 2D arrays
```C
cudaError_t cudaMallocPitch(
void ∗∗ devPtr, // Address of allocated memory (OUT)
size_t ∗ pitch, // Actual size of row in bytes (OUT)
size_t width(*sizeof(struct)),   // Requested table width in bytes (IN)
size_t height); // Requested table height in rows (IN)

// a matrix element p[i,j] of an int matrix [100][100] allocated 
// with cudaMallocPitch ((void ∗∗)&d_matr, &pitch,400,100);
int *p =(int*)((char∗)d_matr + (i∗pitch + j∗sizeof(int)));
			  ^                *                       *^
					ptr2array
```


 
# Managed memory
```
int *in
cudaMallocManaged ((void **) &in, sizeof(int)*N)
```