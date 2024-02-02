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