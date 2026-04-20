# Graph Coloring Register Allocator

Implementation of the Chaitin-Briggs register allocation algorithm — the same class of algorithm used in NVIDIA's PTXAS compiler.

## Why This Matters for NVIDIA PTX
GPU register allocation is uniquely challenging:
- Maxwell/Pascal/Turing GPUs: 32,768 registers per SM
- Each thread uses registers → more registers = fewer active warps
- PTXAS must balance register usage vs thread occupancy
- Spilling to local memory has huge performance cost (DRAM latency)

## Algorithm Pipeline
```
IR Program
    ↓
[Liveness Analysis]     Dataflow equations, fixed-point iteration
    ↓
[Interference Graph]    v1 interferes with v2 if live simultaneously
    ↓
[Simplify]              Remove nodes with degree < K (# registers)
    ↓
[Spill Selection]       If stuck, pick highest-degree node
    ↓
[Color Selection]       Assign physical registers greedily
    ↓
[Spill Code Gen]        Insert loads/stores for spilled vars
```

## Chaitin-Briggs Coloring
```
K = number of physical registers

Simplify:
  while (exists node v with degree < K):
      push v onto stack
      remove v from graph

Spill:
  if graph not empty:
      pick potential spill candidate (highest degree)
      push onto stack, remove from graph
      goto Simplify

Select:
  while stack not empty:
      pop v
      assign lowest available color not used by neighbors
      if no color available: actual spill (insert ld/st)
```

## Features
- Full dataflow liveness analysis
- Interference graph construction
- Chaitin-Briggs simplify/spill/select
- Register coalescing (eliminate mov instructions)
- Spill code generation
- Register pressure analysis

## Build & Run
```bash
g++ -std=c++17 -O2 -o reg_alloc main.cpp
./reg_alloc
```

## Sample Output
```
Virtual → Physical
─────────────────────────
    v1  →  r0
    v2  →  r1
    v3  →  r0  (reuse after v1 dead)
    v4  →  SPILLED TO MEMORY

Register pressure: 4 virtual registers, 3 physical registers
Spill rate: 25%
```

## Concepts Covered
- Register Allocation
- Dataflow Analysis
- Compiler Backend Optimization
- Resource Management

## Future Work
- Coalescing improvements
- SSA-based allocation
- Live range splitting
