# Problem Description

Executing massive computational graphs on hardware with limited on-chip memory is one of the fundamental challenges in modern high-performance computing. When the tensors in a workload are orders of magnitude larger than the accelerator's fast memory, the system cannot simply "load and run." Instead, a complex dance of data movement must be orchestrated, breaking the computation into manageable pieces that fit within the hardware's strict physical constraints.

Your objective is to design a scheduler that analyzes a Directed Acyclic Graph (DAG) of operations and generates an execution strategy that minimizes total latency while respecting all memory capacity limits.

## The Memory Hierarchy

The system simulates a three-tier memory hierarchy common in AI accelerators.

The slow Memory has effectively infinite capacity but limited bandwidth. All graph inputs start here, and all graph outputs must end here. Moving data between the slow memory and the fast memory incurs a time cost.

The fast memory is a high-speed scratchpad with finite capacity (e.g., 50KB). Compute cores can only read/write data that is resident in the fast memory. Accessing this memory has infinite bandwidth (0 time cost), but data stored here consumes capacity.

Ephemeral data exists when operations are grouped into a single subgraph. The intermediate data flowing between them is considered "ephemeral". It passes directly from one operation to the next without ever touching the fast memory. This data consumes zero capacity and incurs zero time cost.

## The Compute Capability

To offset the potentially strict memory limitations, the hardware features a powerful subgraph execution engine. This allows you to execute a sequence of connected operations as a single group, or "subgraph." When operations are grouped into a subgraph, the intermediate data flowing between them becomes ephemeral, passing directly from one operation to the next without ever consuming valuable fast memory.

### Execution Granularity

To maximize performance, you must define exactly how the hardware slices the computation for each subgraph. This is controlled by a 3D configuration tuple `[w, h, k]`. This single configuration creates a unified execution grid that every operation in the subgraph must conform to.

The first two dimensions, `w` and `h`, define the spatial granularity. These values dictate the size of the output slice (`w x h`) produced by the subgraph. Whether the operation is a matrix multiplication or a simple pointwise operation, the hardware will process data in spatial slices of this size to maintain grouping compatibility.

The third dimension, `k`, defines the reduction depth. This is primarily for MatMul operations, where the hardware must compute a dot product over a potentially large inner dimension (`K`). For MatMul, the hardware processes the dot product in steps of size `k`. If `k` is smaller than the full reduction dimension, the system automatically enters an "output stationary" mode. It locks the output slice in the fast memory as an accumulator and iterates through the input matrices in slices per the granule, paying the compute cost for each step. For Pointwise operations, since they have no reduction dimension, `k` is ignored (effectively treating it as 1), and the operation simply executes once per spatial tile.

## The Execution Model

For every computation iteration, the system executes at your specified execution granularity. A slice of input data is loaded from the fast memory, the computation is performed, and the output slice is written back. This creates the primary physical constraint of the problem: The Working Set Limit. For any chosen execution granularity, the sum of the required input slices and the resulting output slices must fit simultaneously within the fast memory capacity. If the working set exceeds this limit, the execution will crash with an Out-Of-Memory (OOM) error.

Note that the hardware has a native execution granularity (e.g., 128x128). While you may choose a finer granularity to fit data into memory, the compute cores cannot physically process chunks smaller than this native size. If you select a granularity smaller than the native size, the hardware 'pads' the execution, meaning you pay the full compute cost of the native size but produce less useful output, thereby increasing the total number of execution steps required.

### Modeling Simplification for Experts

This problem uses an abstract hardware model to focus on scheduling logic rather than low-level cycle counting. First, regarding effective capacity, we treat `fast_memory_capacity` as the effective usable space for a single logical working set. We assume the hardware manages any additional physical overheads (like double-buffering) transparently. Second, regarding strict serialization, while the hardware pipelines operations within a subgraph to hide latency, we enforce strict serialization between subgraphs. This means subgraph N+1 cannot begin its memory operations until subgraph N has fully completed both its computation and memory transfers.

## The Objective

Performance is determined by a throughput-oriented roofline model. For every execution step of a subgraph, the latency is dictated by the bottleneck resource: either the time required to perform the arithmetic (compute bound) or the time required to transfer the boundary data (memory bound). However, execution between distinct subgraphs is strictly serialized; the next subgraph in the schedule begins its memory operations only after the previous subgraph has fully completed both its computation and memory transfers (if applicable).

Your task is to produce a valid execution schedule \- a complete list of subgraphs and their respective execution granularity \- that covers every operation in the graph at least once, while minimizing the sum of these latencies.

## Input / Output Format

### Input

The problem is specified in a JSON file containing the graph topology and hardware specs. Specifically, `tensor[i]` is a graph input if no operation produces it, and `tensor[j]` is a graph output if no operation consumes it. At the beginning of the computation, all graph inputs already reside in the slow memory. At the end of the computation, all graph outputs need to reside in the slow memory.

```json
{
  "widths": [128, 128, ...],       // width of tensor[i]
  "heights": [128, 128, ...],      // height of tensor[i]
  "inputs": [[0, 1], ...],         // inputs[k] is a list of tensor indexes consumed by Operation[k]
                                   // Note: For MatMul, order matters (Left, Right)
  "outputs": [[2], ...],           // outputs[k] is a list of tensor ids produced by Operation[k]
  "base_costs": [1000, 500, ...],  // base_costs[k] is the cost of Operation[k]
  "op_types": ["MatMul", "Pointwise"], // The type of Operation[k]
  "fast_memory_capacity": 25000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128] // Hardware native execution granularity [w, h]
}
```

### Output

You must provide a JSON object containing parallel lists that define the execution schedule.

```json
{ 
  "subgraphs": [
    [0, 1], // Step 1: Group nodes 0 and 1
    [2]     // Step 2: Run node 2 
  ],
  "granularities": [
    [64, 64, 128], // Step 1: MatMul (64x64 output, 128 depth) 
    [128, 128, 1]  // Step 2: Pointwise (128x128 output, k=1)
  ],
  "tensors_to_retain": [ // REQUIRED: Output tensors to keep in Fast Memory
    [1],                 // Step 1: Keep Tensor 1 resident (for Step 2)
    []                   // Step 2: Evict all outputs to Slow Memory
  ],
  "traversal_orders": [ // OPTIONAL: Permutation of slice indices
    [0, 1, 3, 2],       // Step 1: Custom "Snake" order
    null                // Step 2: Default (Raster) [0, 1, 2, 3...] 
  ],
  "subgraph_latencies": [ // REQUIRED: The calculated latency for each step
    2048.0, 
    1024.0
  ]
}
```

Regarding the “granularities” list, the `[w, h, k]` tuple acts as a master key that deterministically sets the shape of all inputs required by the subgraph (recalling that width corresponds to columns and height corresponds to rows). The output and pointwise input will both have width `w` and height `h`. For MatMul inputs, the Left-Hand Side (LHS) input requires width `k` (reduction depth) and height `h`, while the Right-Hand Side (RHS) Input requires width `w` and height `k`.

Regarding the “tensors\_to\_retain” list, a list of lists where tensors\_to\_retain\[k\] specifies which output tensors (or loaded inputs) from Subgraph k should remain resident in the fast memory after the subgraph finishes. Any tensor not in this list is automatically evicted to the slow memory (if it is an output) or discarded (if it was an input). **Note on Data Reuse:** `tensors_to_retain` strictly controls **Inter-Subgraph** persistence (keeping data resident *across* the boundary from one step to the next). For **Intra-Subgraph** reuse (keeping data resident *during* the execution of a single step, e.g., by optimizing `traversal_orders`), the hardware manages residency automatically/implicitly. You do **not** need to list tensors for intra-subgraph reuse in this field.

Regarding the “traversal\_orders” list, when you choose a spatial granularity `(w, h)` smaller than the output tensor, the system implicitly creates a grid of tiles indexed in Row-Major (Raster) Order. For example, a `128x128` tensor with `64x64` granularity creates indices 0 (top-left), 1 (top-right), 2 (bottom-left), and 3 (bottom-right). The “traversal\_orders” field allows you to specify the exact sequence of execution (e.g., `[0, 1, 3, 2]`) to optimize data reuse (like a "Snake" pattern). If omitted, the system defaults to Raster order.

Regarding the “subgraph\_latencies” list, you must provide the total latency for that schedule entry. If a chosen granularity implies multiple tiles (e.g., 4 spatial tiles or 4 split-k steps), the reported latency must be the sum of all those steps.

## Examples

This section provides five examples to demonstrate the core trade-offs of the challenge.

### Example 1: Baseline

````
```mermaid
graph LR
    Tensor0(("Tensor[0]<br>128x128"))
    Tensor1(("Tensor[1]<br>128x128"))
    Tensor2(("Tensor[2]<br>128x128"))
    Op0["Op[0]<br>1000"]
    Op1["Op[1]<br>100"]

    Tensor0 --> Op0
    Op0 --> Tensor1
    Tensor1 --> Op1
    Op1 --> Tensor2
```
````

Input

```json
{
  "widths": [128,128,128],
  "heights": [128,128,128],
  "inputs": [[0], [1]],
  "outputs": [[1], [2]],
  "base_costs": [1000, 100],
  "op_types": ["Pointwise","Pointwise"],
  "fast_memory_capacity": 35000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

#### 

#### Strategy A: Always Spill to Slow Memory

Output

```json
{
  "subgraphs":[[0],[1]],
  "granularities": [[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[]],
  "traversal_orders": [null, null],
  "subgraph_latencies": [3276.8, 3276.8]
}
```

* Compute: efficient. The execution granularity (`128x128`) is the same as the hardware’s native granularity (`128x128`) .  
* Subgraph 0:  
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = Tensor0/B = 128x128/10 = 1638.4`  
  * Run `Op0`. `ComputeTime0 = Op0 = 1,000`  
  * Evict `Tensor1` from the fast memory to the slow memory. `MemoryTime0_out = Tensor1/B = 128x128/10 = 1638.4`  
  * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 3,276.8`  
* Subgraph 1:  
  * Move `Tensor1` from the slow memory to the fast memory. `MemoryTime1_in = Tensor1/B = 128x128/10 = 1638.4`  
  * Run `Op1`. `ComputeTime1 = Op1 = 100`  
  * Evict `Tensor2` from the fast memory to the slow memory. `MemoryTime1_out = Tensor2/B = 128x128/10 = 1638.4`  
  * `TotalLatency1 = max(ComputeTime1, MemoryTime1_in+MemoryTime1_out) = 3,276.8`  
* Graph total:  
  * `TotalLatency = TotalLatency0+TotalLatency1 = 3,276.8 + 3,276.8 = 6,553.6` (Memory Bound).

#### 

#### Strategy B: Mega-Group with Large Granularity (128 x 128\)

Output: 

```json
{
  "subgraphs":[[0,1]],
  "granularities": [[128,128,1]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [3276.8]
}
```

* Compute: efficient. The execution granularity (`128x128`) is the same as the hardware’s native granularity (`128x128`).  
* Subgraph0:  
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = Tensor0/B = 128x128/10 = 1638.4`  
  * Group and run `Op0` and `Op1`.  `ComputeTime0 = Op0+Op1 = 1,000+100 = 1,100`  
  * Evict `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = Tensor2/B = 128x128/10 = 1638.4`  
  * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 3,276.8`  
* Graph total:  
  * `TotalLatency = TotalLatency0 = 3,276.8` (Memory Bound, but 2X faster than Strategy A).

#### 

#### Strategy C: Mega-Group with Small Granularity (64 x 64\)

Output: 

```json
{
  "subgraphs":[[0,1]],
  "granularities": [[64,64,1]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [4400.0]
}
```

* Compute: inefficient. Because the granularity is small, a `128x128` pointwise op now takes 4 passes to fully compute on `64x64` granularity.  
* Subgraph0:  
  * Now has to execute 4 times. Each time:  
    * Move `¼` `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = ¼ Tensor0/B = 64x64/10 = 409.6`  
    * Group and run `Op0` and `Op1`.  `ComputeTime0 = Op0+Op1 = 1,000+100 = 1,100`  
    * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
    * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 1,100`  
* Graph total:  
  * `TotalLatency = 4 x TotalLatency0 = 4,400`  (Compute Bound, but 1.5X faster than Strategy A).

### Example 2: Fast Memory Capacity

The tensors are now `256x256`.   
Input:

```json
{
  "widths": [256,256,256],
  "heights": [256,256,256],
  "inputs": [[0], [1]],
  "outputs": [[1], [2]],
  "base_costs": [1000, 100],
  "op_types": ["Pointwise","Pointwise"],
  "fast_memory_capacity": 25000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

#### 

#### Strategy A: Always Spill to Slow Memory

Output: 

```json
{
  "subgraphs":[[0],[1]],
  "granularities": [[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[]],
  "traversal_orders": [null, null],
  "subgraph_latencies": [13107.2, 13107.2]
}
```

* Compute: efficient. An execution granularity of `256x256` pointwise operations will take 4 passes to fully compute on the `128x128` native granularity.  
* Subgraph0:  
  * Now has to execute 4 times. Each time:  
    * Move `¼` `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = ¼ Tensor0/B = 128x128/10 = 1,638.4`  
    * Run `Op0`.  `ComputeTime0 = Op0 = 1,000`  
    * Evict `¼` `Tensor1` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 128x128/10 = 1,638.4`  
  * `TotalLatency0 = 4 x max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 13,107.2`  
* Subgraph1:  
  * Also has to execute 4 times. Each time:  
    * Move `¼` `Tensor1` from the slow memory to the fast memory. `MemoryTime1_in = ¼ Tensor0/B = 128x128/10 = 1,638.4`  
    * Run `Op1`.  `ComputeTime1 = Op1 = 100`  
    * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime1_out = ¼ Tensor2/B = 128x128/10 = 1,638.4`  
  * `TotalLatency1 = 4 x max(ComputeTime1, MemoryTime0_in+MemoryTime0_out) = 13,107.2`  
* Graph total:  
  * `TotalLatency = TotalLatency0 + TotalLatency1 = 26,214.4` (Memory Bound).

#### 

#### Strategy B: Mega-Group with Small Granularity (128 x 128\)

Output: 

```json
{
  "subgraphs":[[0,1]],
  "granularities": [[128,128,1]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [13107.2]
}
```

* Compute: efficient. The execution granularity (`128x128`) is the same as the hardware’s native granularity (`128x128`).  
* Subgraph0:  
  * Now has to execute 4 times. Each time:  
    * Move `¼` `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = ¼ Tensor0/B = 128x128/10 = 1,638.4`  
    * Group and run `Op0` and `Op1`.  `ComputeTime0 = Op0+Op1 = 1,000+100 = 1,100`  
    * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 128x128/10 = 1,638.4`  
  * `TotalLatency0 = 4 x max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 13,107.2`  
* Graph total:  
  * `TotalLatency = TotalLatency0 = 13,107.2`  (Memory Bound, but 2X faster than Strategy A)

### 

### Example 3: Spilling vs. Recomputation

A "Diamond" graph (skip connection) where an intermediate (`Tensor1`) is needed by two downstream branches. `Op2` needs both `Tensor1` and `Tensor2` to start.

````
```mermaid
graph LR
    Tensor0(("Tensor[0]<br>128x128"))
    Tensor1(("Tensor[1]<br>128x128"))
    Tensor2(("Tensor[2]<br>128x128"))
    Tensor3(("Tensor[3]<br>128x128"))
    Op0["Op[0]<br>1500"]
    Op1["Op[1]<br>1500"]
    Op2["Op[2]<br>1500"]

    Tensor0 --> Op0
    Op0 --> Tensor1
    Tensor1 --> Op1
    Op1 --> Tensor2
    Tensor1 --> Op2
    Tensor2 --> Op2
    Op2 --> Tensor3
```
````

Input:

```json
{
  "widths": [128,128,128,128],
  "heights": [128,128,128,128],
  "inputs": [[0],[1],[1,2]],
  "outputs": [[1],[2],[3]],
  "base_costs": [1500,1500,1500],
  "op_types": ["Pointwise","Pointwise","Pointwise"],
  "fast_memory_capacity": 50000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

#### 

#### Strategy A: Spilling (The "Cache" Approach)

Output:

```json
{
  "subgraphs":[[0],[1],[2]],
  "granularities": [[128,128,1],[128,128,1],[128,128,1]],
  "tensors_to_retain": [[],[],[]],
  "traversal_orders": [null,null,null],
  "subgraph_latencies": [3276.8,1638.4,3276.8]
}
```

* Subgraph0: compute `Tensor1`; evict it to the slow memory.  
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = Tensor0/B = 128x128/10 = 1,638.4`  
  * Run `Op0`. `ComputeTime0 = Op0 = 1,500`  
  * Evict `Tensor1` from the fast memory to the slow memory. `MemoryTime0_out = Tensor1/B = 128x128/10 = 1,638.4`  
  * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 3,276.8`  
* Subgraph 1: compute `Tensor2`. Evict it to the slow memory.  
  * Move `Tensor1` from the slow memory to the fast memory. `MemoryTime1_in = Tensor1/B = 128x128/10 = 1,638.4`  
  * Run `Op1`. `ComputeTime1 = Op1 = 1,500`  
  * Evict `Tensor2` from the fast memory to the slow memory. `MemoryTime1_out = Tensor1/B = 128x128/10 = 1,638.4`  
  * `TotalLatency1 = max(ComputeTime1, MemoryTime1_in+MemoryTime1_out) = 3,276.8`  
* Subgraph 2: compute `Tensor3`. Feed `Tensor1` (evicted) and `Tensor2` (evicted) to make `Tensor3`. `Tensor3` is evicted to the slow memory as the final result of the graph.  
  * Move `Tensor1` from the slow memory to the fast memory. `MemoryTime2_in = Tensor1/B = 128x128/10 = 1,638.4`  
  * Move `Tensor2` from the slow memory to the fast memory. `MemoryTime2_in += Tensor1/B = 128x128/10 = 1,638.4` (`MemoryTime2_in = 3276.8`)  
  * Run `Op2`. `ComputeTime2 = Op2 = 1,500`  
  * Evict `Tensor3` from the fast memory to the slow memory. `MemoryTime2_out = Tensor3/B = 128x128/10 = 1,638.4`  
  * `TotalLatency2 = max(ComputeTime2, MemoryTime2_in+MemoryTime2_out) = 4,915.2`  
* Graph total:  
  * `TotalLatency = TotalLatency0+TotalLatency1+TotalLatency2 = 11,468.8`  (Memory Bound)

#### Strategy B: Recomputation (The "Flash" Approach)

We discard `Tensor1` to save memory, then recompute it when needed.  
Output: 

```json
{
  "subgraphs":[[0,1],[0,2]],
  "granularities": [[128,128,1],[128,128,1]],
  "tensors_to_retain": [[2],[]],
  "traversal_orders": [null, null],
  "subgraph_latencies": [3000,3276.8]
}
```

* Subgraph 0: compute `Tensor2`. Keep it resident.  
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = Tensor0/B = 128x128/10 = 1,638.4`  
  * Run `Op0` and `Op1`. `ComputeTime0 = Op0 + Op1 = 1,500 + 1,500 = 3,000`  
  * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in) = 3,000`  
* Subgraph 1: compute `Tensor3`.   
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime1_in = Tensor0/B = 128x128/10 = 1,638.4`  
  * Run `Op0` and `Op2`. `ComputeTime1 = Op0 + Op2 = 1,500 + 1,500 = 3,000`  
  * Evict `Tensor3` from the fast memory to the slow memory. `MemoryTime1_out = Tensor3/B = 128x128/10 = 1,638.4`  
  * `TotalLatency1 = max(ComputeTime1, MemoryTime1_in + MemoryTime1_out) = 3,276.8`  
* Graph total:  
  * `TotalLatency = TotalLatency0 +TotalLatency1 = 6,276.8` (45% faster than Strategy A).

#### 

#### Strategy C: Selective Residency (The "Hybrid" Approach)

We keep `Tensor1` resident, `Tensor2` ephemeral.  
Output: 

```json
{
  "subgraphs":[[0],[1,2]],
  "granularities": [[128,128,1],[128,128,1]],
  "tensors_to_retain": [[1],[]],
  "traversal_orders": [null, null],
  "subgraph_latencies": [1638.4,3000]
}
```

* Subgraph 0: compute `Tensor1`. Keep it resident.  
  * Move `Tensor0` from the slow memory to the fast memory. `MemoryTime0_in = Tensor0/B = 128x128/10 = 1,638.4`  
  * Run `Op0`. `ComputeTime0 = Op0 = 1,500`  
  * `TotalLatency0 = max(ComputeTime0, MemoryTime0_in) = 1,638.4`  
* Subgraph 1: compute `Tensor3`.   
  * Run `Op1` and `Op2`. `ComputeTime1 = Op1 + Op2 = 1,500 + 1,500 = 3,000`  
  * Evict `Tensor3` from the fast memory to the slow memory. `MemoryTime1_out = Tensor3/B = 128x128/10 = 1,638.4`  
  * `TotalLatency1 = max(ComputeTime1, MemoryTime1_out) = 3,000`  
* Graph total:  
  * `TotalLatency = TotalLatency0 +TotalLatency1 = 4,638.4` (60% faster than Strategy A).

### Example 4: Revisit

This example demonstrates how execution order impacts bandwidth. For MatMul, processing tiles in a "zig-zag" order allows us to keep data resident in the fast   
memory, avoiding expensive re-loads ("Revisits").

````
```mermaid
graph LR
    Tensor0(("Tensor[0]<br>128x128"))
    Tensor1(("Tensor[1]<br>128x128"))
    Tensor2(("Tensor[2]<br>128x128"))
    Op0["Op[0]<br>1500"]

    Tensor0 --> Op0
    Tensor1 --> Op0
    Op0 --> Tensor2
```
````

Input

```json
{
  "widths": [128,128,128],
  "heights": [128,128,128],
  "inputs": [[0,1]],
  "outputs": [[2]],
  "base_costs": [1500],
  "op_types": ["MatMul"],
  "fast_memory_capacity": 25000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

We will divide `tensor0` (`128x128`) into 2 stacked row strips:

* Row strip 0 (`64x128`). `MemoryTime = 64x128/10 = 819.2`  
* Row strip 1 (`64x128`). `MemoryTime = 64x128/10 = 819.2`

Similarly, we divide `tensor1` (`128x128`) into 2 concatenated column strips:

* Column strip 0 (`128x64`). `MemoryTime = 128x64/10 = 819.2`  
* Column strip 1 (`128x64`). `MemoryTime = 128x64/10 = 819.2`

#### Strategy A: Naive Tiling (High Revisit)

We process tiles in standard raster order (top-left \-\> top-right \-\> bottom-left \-\> bottom-right) but flush the fast memory every time.  
Output

```json
{
  "subgraphs": [[0]],
  "granularities": [[64,64,128]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [8192]
}
```

Smaller granularity is required, because all 3 tensors can’t co-exist in the fast memory due to capacity limitation.

The `128x128` output is computed in 4 equal steps.

* Step 1 (top-left):  
  * Move row strip 0 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Move column strip 0 from the slow memory to the fast memory. `MemoryTime0_in += 819.2` (`MemoryTime0_in = 1638.4`).  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_1 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048`  
* Step 2 (top-right):  
  * Move row strip 0 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Move column strip 1 from the slow memory to the fast memory. `MemoryTime0_in += 819.2` (`MemoryTime0_in = 1638.4`).  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_2 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048`  
* Step 3 (lower-left):  
  * Move row strip 1 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Move column strip 0 from the slow memory to the fast memory. `MemoryTime0_in += 819.2` (`MemoryTime0_in = 1638.4`).  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_3 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048`  
* Step 4 (lower-right):  
  * Move row strip 1 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Move column strip 1 from the slow memory to the fast memory. `MemoryTime0_in += 819.2` (`MemoryTime0_in = 1638.4`).  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_4 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048`  
* Graph total:  
  * `TotalLatency = TotalLatency0_1 + TotalLatency0_2 + TotalLatency0_3 + TotalLatency0_4 = 8,192` (Memory bound).

#### 

#### Strategy B: Optimized Traversal (Data Reuse)

We process tiles in a "zig-zag" order (top-left \-\> top-right \-\> bottom-right \-\> bottom-left) to maximize residency.  
Output

```json
{
  "subgraphs": [[0]],
  "granularities": [[64,64,128]],
  "tensors_to_retain": [[]],
  "traversal_orders": [[0, 1, 3, 2]],
  "subgraph_latencies": [6548]
}
```

Smaller granularity is required, because all 3 tensors can’t co-exist in the fast memory due to capacity limitation.

The `128x128` output is divided into 4 chunks.

* Step 1 (top-left):  
  * Move row strip 0 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Move column strip 0 from the slow memory to the fast memory. `MemoryTime0_in += 819.2` (`MemoryTime0_in = 1638.4`).  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_1 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 2,048`  
* Step 2 (top-right):  
  * Reuse resident row strip 0\.  
  * Move column strip 1 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`   
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_2 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 1,500`  
* Step 3 (lower-right):  
  * Move row strip 1 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`  
  * Reuse resident column strip 1\.  
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_3 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 1,500`  
* Step 4 (lower-left):  
  * Reuse resident row strip 1\.  
  * Move column strip 0 from the slow memory to the fast memory. `MemoryTime0_in = 819.2`   
  * Run `Op0`.  `ComputeTime0 = Op0 = 1,500`  
  * Evict `¼` `Tensor2` from the fast memory to the slow memory. `MemoryTime0_out = ¼ Tensor2/B = 64x64/10 = 409.6`  
  * `TotalLatency0_4 = max(ComputeTime0, MemoryTime0_in+MemoryTime0_out) = 1,500`  
* Graph total:  
  * `TotalLatency = TotalLatency0_1 + TotalLatency0_2 + TotalLatency0_3 + TotalLatency0_4 = 6,548` (Largely compute bound, 20% faster than Strategy A).

### 

### Example 5: Chained Matrix Multiplication (Split-K)

This example demonstrates advanced subgraph grouping for chained MatMuls (`(A @ B) @ C`). It shows how manipulating the `k` dimension controls the size of the intermediate tensor in fast memory.

````json
```mermaid
graph LR
    Tensor0(("Tensor[0]<br>128x128"))
    Tensor1(("Tensor[1]<br>128x128"))
    Tensor2(("Tensor[2]<br>128x128"))
    Tensor3(("Tensor[3]<br>128x128"))
    Tensor4(("Tensor[4]<br>128x128"))
    Op0["Op[0]<br>2000"]
    Op1["Op[1]<br>2000"]

    Tensor0 --> Op0
    Tensor1 --> Op0
    Op0 --> Tensor3
    Tensor3 --> Op1
    Tensor2 --> Op1
    Op1 --> Tensor4
```
````

* Scenario**:** We calculate `(Tensor0 @ Tensor1) @ Tensor2`.   
* Constraint**:** The fast memory capacity (45,000) is tight. It cannot hold three full `128x128` tensors (16,384 each) simultaneously.

Input:

```json
{
  "widths": [128,128,128,128,128],
  "heights": [128,128,128,128,128],
  "inputs": [[0,1], [3,2]], // Op0 uses 0&1. Op1 uses 3(result of Op0) & 2.
  "outputs": [[3], [4]],
  "base_costs": [2000, 2000],
  "op_types": ["MatMul", "MatMul"],
  "fast_memory_capacity": 45000,
  "slow_memory_bandwidth": 10,
  "native_granularity": [128, 128]
}
```

#### 

#### Strategy A: Materialization (Large K)

We group the operations but use the full reduction depth (`k=128`). This forces the system to fully compute and store `Tensor3` (the intermediate) before starting `Op1`.  
Output

```json
NA
```

* Memory check (FAIL): to execute `Op0`, we need `Tensor0` (`128x128`), `Tensor1` (`128x128`), and the Output `Tensor3` (`128x128`) resident. The Working Set is `49,152` (`3x128x128`).  
* `49,152 > 45,000`. OOM.

#### 

#### Strategy B: Split-K Pipelining (Small K)

We group the operations with a small reduction depth (`k=32`). This forces the system to accumulate the result in 4 steps, minimizing intermediate memory usage.  
Output

```json
{
  "subgraphs": [[0, 1]],
  "granularities": [[128, 128, 32]],
  "tensors_to_retain": [[]],
  "traversal_orders": [null],
  "subgraph_latencies": [6915.2]
}
```

* Memory check: we keep `Tensor0` (`128x128`) and the accumulator `Tensor4` (`128x128`) resident. We stream `Tensor1` (`128x32` strip) and `Tensor2` (`128x32` strip). Total Working Set is `40,960`.  
* `40,960 < 45,000`. No OOM.

The `128x128` output (`Tensor4`) is computed in 4 accumulation steps.

* Step 1 (`k=0..31`):  
  * Move `Tensor0` (`128x128`) from slow memory. `MemoryTime1_in = (128x128)/10 = 1638.4`  
  * Move `Tensor1` (col strip 0: `128x32`) from slow memory. `MemoryTime1_in += 409.6` (`Total = 2,048`)  
  * Move `Tensor2` (row strip 0: `32x128`) from slow memory. `MemoryTime1_in += 409.6` (`Total = 2,457.6`)  
  * Run `Op0` and `Op1`. `ComputeTime1 = 1,000`  
  * `TotalLatency1 = max(ComputeTime1, MemoryTime1_in) = 2,457.6` (Memory bound)  
* Step 2 (`k=32..63`):  
  * Reuse resident `Tensor0` and Accumulator `Tensor4`.  
  * Move `Tensor1` (col strip 1: `128x32`) from slow memory. `MemoryTime2_in = 409.6`  
  * Move `Tensor2` (row strip 1: `32x128`) from slow memory. `MemoryTime2_in += 409.6` (`Total = 819.2`)  
  * Run `Op0` and `Op1`. `ComputeTime2 = 1,000`  
  * `TotalLatency2 = max(ComputeTime2, MemoryTime2_in) = 1,000` (Compute bound)  
* Step 3 (`k=64..95`):  
  * Reuse resident `Tensor0` and accumulator `Tensor4`.  
  * Move `Tensor1` (col strip 2: `128x32`) and `Tensor2` (row strip 2: `32x128`). `MemoryTime3_in = 819.2`  
  * Run `Op0` and `Op1`. `ComputeTime3 = 1,000`  
  * `TotalLatency3 = max(ComputeTime3, MemoryTime3_in) = 1,000` (Compute bound)  
* Step 4 (`k=96..127`):  
  * Reuse resident `Tensor0` and accumulator `Tensor4`.  
  * Move `Tensor1` (col strip 3: `128x32`) and `Tensor2` (row strip 3: `32x128`). `MemoryTime4_in = 819.2`  
  * Run `Op0` and `Op1`. `ComputeTime4 = 1,000`  
  * Evict `Tensor4` to slow memory. `MemoryTime4_out = 1638.4`  
  * `TotalLatency4 = max(ComputeTime4, MemoryTime4_in + MemoryTime4_out) = 2,457.6` (Memory bound)  
* Graph total:  
  * `TotalLatency = 6,915.2`.
