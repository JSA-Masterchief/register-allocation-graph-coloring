/**
 * ============================================================
 * Graph Coloring Register Allocator
 * ============================================================
 * Implements the classic Chaitin-Briggs register allocation
 * algorithm used in production compilers including PTXAS.
 *
 * Pipeline:
 *   1. Build Interference Graph from live ranges
 *   2. Compute liveness (dataflow analysis)
 *   3. Simplify: find low-degree nodes to color
 *   4. Spill: select candidates for memory spilling
 *   5. Select: assign physical registers (colors)
 *   6. Rewrite: insert spill/reload code
 *
 * This is directly relevant to PTXAS which must allocate
 * GPU registers (very limited resource) to virtual registers.
 *
 * Author: Jujhaar Singh Aidhen
 * Relevance: PTXAS Register Allocation, Compiler Backend
 * ============================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cassert>

// ============================================================
// SECTION 1: IR REPRESENTATION
// ============================================================

struct IRInstr {
    std::string op;
    std::string dest;       // defined register (empty if none)
    std::vector<std::string> uses;  // used registers
    std::string label;
    bool isLabel = false;
    bool isBranch = false;
    std::string branchTarget;

    static IRInstr makeLabel(const std::string& lbl) {
        IRInstr i; i.isLabel = true; i.label = lbl; return i;
    }

    static IRInstr makeBinop(const std::string& op,
                              const std::string& dest,
                              const std::string& src1,
                              const std::string& src2) {
        IRInstr i; i.op = op; i.dest = dest;
        i.uses = {src1, src2}; return i;
    }

    static IRInstr makeMove(const std::string& dest,
                             const std::string& src) {
        IRInstr i; i.op = "mov"; i.dest = dest;
        i.uses = {src}; return i;
    }

    static IRInstr makeLoad(const std::string& dest,
                             const std::string& addr) {
        IRInstr i; i.op = "load"; i.dest = dest;
        i.uses = {addr}; return i;
    }

    static IRInstr makeStore(const std::string& addr,
                              const std::string& val) {
        IRInstr i; i.op = "store";
        i.uses = {addr, val}; return i;
    }

    static IRInstr makeBranch(const std::string& cond,
                               const std::string& target) {
        IRInstr i; i.op = "bra"; i.uses = {cond};
        i.isBranch = true; i.branchTarget = target; return i;
    }

    static IRInstr makeReturn(const std::string& val = "") {
        IRInstr i; i.op = "ret";
        if (!val.empty()) i.uses = {val};
        return i;
    }

    std::string toString() const {
        if (isLabel) return label + ":";
        std::string s = "  ";
        if (!dest.empty()) s += dest + " = ";
        s += op;
        for (size_t j = 0; j < uses.size(); j++)
            s += (j == 0 ? " " : ", ") + uses[j];
        return s;
    }
};

// ============================================================
// SECTION 2: BASIC BLOCKS AND CFG
// ============================================================

struct BasicBlock {
    std::string name;
    std::vector<IRInstr> instrs;
    std::vector<std::string> successors;
    std::vector<std::string> predecessors;

    // Liveness sets
    std::set<std::string> liveIn;
    std::set<std::string> liveOut;
    std::set<std::string> def;  // variables defined in block
    std::set<std::string> use;  // variables used before defined

    void computeDefUse() {
        def.clear(); use.clear();
        for (auto& instr : instrs) {
            if (instr.isLabel) continue;
            // Uses come before defs
            for (auto& u : instr.uses) {
                if (!def.count(u) && u[0] != '%' || u.size() > 1)
                    if (isVirtualReg(u) && !def.count(u))
                        use.insert(u);
            }
            if (!instr.dest.empty() && isVirtualReg(instr.dest))
                def.insert(instr.dest);
        }
    }

    bool isVirtualReg(const std::string& s) {
        return !s.empty() && s[0] == 'v';
    }
};

class CFG {
public:
    std::map<std::string, BasicBlock> blocks;
    std::string entryBlock;

    void addBlock(const std::string& name) {
        blocks[name].name = name;
    }

    void addEdge(const std::string& from, const std::string& to) {
        blocks[from].successors.push_back(to);
        blocks[to].predecessors.push_back(from);
    }

    void addInstr(const std::string& block, IRInstr instr) {
        blocks[block].instrs.push_back(instr);
    }

    void print() const {
        std::cout << "Control Flow Graph:\n";
        for (auto& [name, bb] : blocks) {
            std::cout << "  Block " << name << ":\n";
            for (auto& instr : bb.instrs)
                std::cout << "  " << instr.toString() << "\n";
            std::cout << "    Successors: ";
            for (auto& s : bb.successors) std::cout << s << " ";
            std::cout << "\n";
        }
    }
};

// ============================================================
// SECTION 3: LIVENESS ANALYSIS (Dataflow)
// ============================================================

class LivenessAnalyzer {
private:
    CFG& cfg;

    bool isVirtualReg(const std::string& s) {
        return !s.empty() && s[0] == 'v';
    }

public:
    LivenessAnalyzer(CFG& g) : cfg(g) {}

    void computeLiveness() {
        // Step 1: Compute def and use sets for each block
        for (auto& [name, bb] : cfg.blocks) {
            cfg.blocks[name].computeDefUse();
        }

        // Step 2: Iterative dataflow (backward analysis)
        bool changed = true;
        int iteration = 0;

        while (changed) {
            changed = false;
            iteration++;

            // Process blocks in reverse postorder
            for (auto& [name, bb] : cfg.blocks) {
                BasicBlock& block = cfg.blocks[name];

                // liveOut[B] = union of liveIn[S] for all successors S
                std::set<std::string> newLiveOut;
                for (auto& succName : block.successors) {
                    auto& succ = cfg.blocks[succName];
                    for (auto& v : succ.liveIn)
                        newLiveOut.insert(v);
                }

                // liveIn[B] = use[B] ∪ (liveOut[B] - def[B])
                std::set<std::string> newLiveIn = block.use;
                for (auto& v : newLiveOut) {
                    if (!block.def.count(v))
                        newLiveIn.insert(v);
                }

                if (newLiveIn != block.liveIn || newLiveOut != block.liveOut) {
                    block.liveIn = newLiveIn;
                    block.liveOut = newLiveOut;
                    changed = true;
                }
            }
        }

        std::cout << "[Liveness] Converged in " << iteration << " iterations\n";
    }

    void printLiveness() const {
        std::cout << "\nLiveness Analysis Results:\n";
        for (auto& [name, bb] : cfg.blocks) {
            std::cout << "  Block " << name << ":\n";
            std::cout << "    USE:     { ";
            for (auto& v : bb.use) std::cout << v << " ";
            std::cout << "}\n    DEF:     { ";
            for (auto& v : bb.def) std::cout << v << " ";
            std::cout << "}\n    LiveIN:  { ";
            for (auto& v : bb.liveIn) std::cout << v << " ";
            std::cout << "}\n    LiveOUT: { ";
            for (auto& v : bb.liveOut) std::cout << v << " ";
            std::cout << "}\n";
        }
    }
};

// ============================================================
// SECTION 4: INTERFERENCE GRAPH
// ============================================================

class InterferenceGraph {
public:
    std::set<std::string> nodes;
    std::map<std::string, std::set<std::string>> adjList;
    std::map<std::pair<std::string,std::string>, bool> edges;

    void addNode(const std::string& v) {
        nodes.insert(v);
        if (!adjList.count(v)) adjList[v] = {};
    }

    void addEdge(const std::string& u, const std::string& v) {
        if (u == v) return;
        nodes.insert(u); nodes.insert(v);
        adjList[u].insert(v);
        adjList[v].insert(u);
        edges[{std::min(u,v), std::max(u,v)}] = true;
    }

    bool hasEdge(const std::string& u, const std::string& v) const {
        auto key = std::make_pair(std::min(u,v), std::max(u,v));
        return edges.count(key) > 0;
    }

    int degree(const std::string& v) const {
        auto it = adjList.find(v);
        return it != adjList.end() ? (int)it->second.size() : 0;
    }

    void print() const {
        std::cout << "\nInterference Graph:\n";
        std::cout << "  Nodes: " << nodes.size() << "\n";
        for (auto& v : nodes) {
            std::cout << "  " << std::setw(6) << v
                      << " (deg=" << degree(v) << "): { ";
            auto it = adjList.find(v);
            if (it != adjList.end())
                for (auto& n : it->second) std::cout << n << " ";
            std::cout << "}\n";
        }
    }

    // Build interference graph from liveness info
    void buildFromCFG(const CFG& cfg) {
        for (auto& [bname, bb] : cfg.blocks) {
            // Compute live set at each instruction (backward scan)
            std::set<std::string> live = bb.liveOut;

            for (int i = (int)bb.instrs.size()-1; i >= 0; i--) {
                const auto& instr = bb.instrs[i];
                if (instr.isLabel) continue;

                // If this instruction defines a register,
                // it interferes with everything in live set
                if (!instr.dest.empty() && instr.dest[0] == 'v') {
                    for (auto& v : live) {
                        if (v != instr.dest)
                            addEdge(instr.dest, v);
                    }
                    addNode(instr.dest);
                    live.erase(instr.dest);
                }

                // Add uses to live set
                for (auto& u : instr.uses) {
                    if (!u.empty() && u[0] == 'v')
                        live.insert(u);
                }
            }
        }
    }
};

// ============================================================
// SECTION 5: CHAITIN-BRIGGS GRAPH COLORING REGISTER ALLOCATOR
// ============================================================

class RegisterAllocator {
private:
    InterferenceGraph ig;
    int numRegisters;  // Available physical registers (colors)
    std::vector<std::string> regNames;

    // Maps virtual register → physical register
    std::map<std::string, std::string> allocation;
    // Registers that were spilled to memory
    std::set<std::string> spilled;

    // Stack for coloring order (Chaitin simplification)
    std::stack<std::string> coloringStack;

    // Degree of each node during simplification
    std::map<std::string, int> currentDegree;

    void initDegrees() {
        for (auto& v : ig.nodes) {
            currentDegree[v] = ig.degree(v);
        }
    }

    // Simplify: repeatedly remove nodes with degree < K
    void simplify(std::set<std::string>& remaining) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = remaining.begin(); it != remaining.end(); ) {
                const std::string& v = *it;
                if (currentDegree[v] < numRegisters) {
                    coloringStack.push(v);
                    // Reduce degree of neighbors
                    auto& adj = ig.adjList[v];
                    for (auto& n : adj) {
                        if (remaining.count(n))
                            currentDegree[n]--;
                    }
                    it = remaining.erase(it);
                    changed = true;
                } else {
                    ++it;
                }
            }
        }
    }

    // Spill: pick a candidate to spill (highest degree)
    std::string selectSpillCandidate(const std::set<std::string>& remaining) {
        std::string best;
        int maxDeg = -1;
        for (auto& v : remaining) {
            if (currentDegree[v] > maxDeg) {
                maxDeg = currentDegree[v];
                best = v;
            }
        }
        return best;
    }

    // Select: assign colors greedily
    bool select(const std::string& v,
                std::map<std::string, std::string>& coloring) {
        std::set<std::string> usedColors;
        auto& adj = ig.adjList[v];
        for (auto& n : adj) {
            if (coloring.count(n))
                usedColors.insert(coloring[n]);
        }

        // Find first available color (register)
        for (auto& reg : regNames) {
            if (!usedColors.count(reg)) {
                coloring[v] = reg;
                return true;
            }
        }
        return false; // Actual spill
    }

public:
    RegisterAllocator(InterferenceGraph graph, int numRegs)
        : ig(std::move(graph)), numRegisters(numRegs) {
        // Create register names (r0, r1, ..., r(K-1))
        for (int i = 0; i < numRegs; i++)
            regNames.push_back("r" + std::to_string(i));
    }

    void allocate() {
        std::cout << "\n[RegAlloc] Starting allocation with "
                  << numRegisters << " registers\n";
        std::cout << "[RegAlloc] Virtual registers: " << ig.nodes.size() << "\n";

        std::map<std::string, std::string> coloring;
        std::set<std::string> remaining(ig.nodes.begin(), ig.nodes.end());

        initDegrees();

        // Phase 1: Simplify + Spill
        while (!remaining.empty()) {
            // Try to simplify (remove low-degree nodes)
            simplify(remaining);

            // If nodes still remain, must spill one
            if (!remaining.empty()) {
                std::string spillCand = selectSpillCandidate(remaining);
                std::cout << "[RegAlloc] Potential spill: " << spillCand
                          << " (degree=" << currentDegree[spillCand] << ")\n";
                coloringStack.push(spillCand);

                // Reduce neighbor degrees
                for (auto& n : ig.adjList[spillCand]) {
                    if (remaining.count(n))
                        currentDegree[n]--;
                }
                remaining.erase(spillCand);
            }
        }

        // Phase 2: Select (pop stack and assign colors)
        std::cout << "\n[RegAlloc] Assigning registers:\n";
        while (!coloringStack.empty()) {
            std::string v = coloringStack.top();
            coloringStack.pop();

            if (select(v, coloring)) {
                std::cout << "  " << std::setw(6) << v
                          << " → " << coloring[v] << "\n";
            } else {
                // Must spill to memory
                spilled.insert(v);
                std::cout << "  " << std::setw(6) << v
                          << " → SPILLED TO MEMORY\n";
            }
        }

        allocation = coloring;
    }

    void printAllocation() const {
        std::cout << "\n[RegAlloc] Final Register Allocation:\n";
        std::cout << "  Virtual → Physical\n";
        std::cout << "  " << std::string(25, '-') << "\n";
        for (auto& [virt, phys] : allocation) {
            std::cout << "  " << std::setw(8) << virt
                      << "  →  " << phys << "\n";
        }
        if (!spilled.empty()) {
            std::cout << "\n  Spilled to memory: { ";
            for (auto& s : spilled) std::cout << s << " ";
            std::cout << "}\n";
        }
        std::cout << "\n  Register pressure: "
                  << allocation.size() << " virtual registers mapped to "
                  << numRegisters << " physical registers\n";
    }

    void analyzeSpillCost() const {
        std::cout << "\n[RegAlloc] Analysis:\n";
        std::cout << "  Total virtual registers: " << ig.nodes.size() << "\n";
        std::cout << "  Successfully allocated:  " << allocation.size() << "\n";
        std::cout << "  Spilled:                 " << spilled.size() << "\n";

        double spillRate = ig.nodes.empty() ? 0 :
            100.0 * spilled.size() / ig.nodes.size();
        std::cout << "  Spill rate:              "
                  << std::fixed << std::setprecision(1)
                  << spillRate << "%\n";

        // Count coalescing opportunities (mov instructions)
        std::cout << "\n  Note: GPU register pressure is critical.\n";
        std::cout << "  NVIDIA GPUs have 32K-64K 32-bit registers per SM.\n";
        std::cout << "  More registers per thread = fewer active warps.\n";
        std::cout << "  PTXAS must balance register usage vs occupancy.\n";
    }

    const std::map<std::string, std::string>& getAllocation() const {
        return allocation;
    }
};

// ============================================================
// SECTION 6: REGISTER COALESCING
// ============================================================

class RegisterCoalescer {
public:
    // Try to eliminate move instructions by coalescing registers
    // v1 and v2 can be coalesced if they don't interfere
    std::map<std::string, std::string> coalesce(
        const InterferenceGraph& ig,
        const std::vector<std::pair<std::string,std::string>>& moves)
    {
        std::map<std::string, std::string> coalesceMap;

        for (auto& [v1, v2] : moves) {
            if (!ig.hasEdge(v1, v2)) {
                // They don't interfere → can coalesce!
                coalesceMap[v2] = v1;
                std::cout << "  [Coalesce] " << v2 << " → " << v1
                          << " (mov eliminated!)\n";
            } else {
                std::cout << "  [Coalesce] " << v1 << " and " << v2
                          << " interfere, cannot coalesce\n";
            }
        }
        return coalesceMap;
    }
};

// ============================================================
// SECTION 7: SPILL CODE GENERATOR
// ============================================================

class SpillCodeGenerator {
private:
    int spillSlot = 0;
    std::map<std::string, int> spillLocations;

public:
    void generateSpillCode(const std::set<std::string>& spilled,
                            std::vector<IRInstr>& instrs) {
        // Assign memory slots to spilled variables
        for (auto& v : spilled) {
            spillLocations[v] = spillSlot++;
            std::cout << "  [Spill] " << v
                      << " → memory[sp+" << spillLocations[v] << "]\n";
        }

        // Insert load/store around uses/defs
        std::vector<IRInstr> newInstrs;
        for (auto& instr : instrs) {
            if (instr.isLabel) { newInstrs.push_back(instr); continue; }

            // Before instruction: load spilled uses
            for (auto& u : instr.uses) {
                if (spilled.count(u)) {
                    std::string tmp = u + "_reload";
                    IRInstr load = IRInstr::makeLoad(tmp,
                        "sp+" + std::to_string(spillLocations[u]));
                    load.op = "// RELOAD " + u;
                    newInstrs.push_back(load);
                }
            }

            newInstrs.push_back(instr);

            // After instruction: store spilled defs
            if (!instr.dest.empty() && spilled.count(instr.dest)) {
                IRInstr store = IRInstr::makeStore(
                    "sp+" + std::to_string(spillLocations[instr.dest]),
                    instr.dest);
                store.op = "// SPILL " + instr.dest;
                newInstrs.push_back(store);
            }
        }
        instrs = newInstrs;
    }
};

// ============================================================
// SECTION 8: TEST CASES
// ============================================================

CFG buildTestCFG1() {
    // Simple linear program:
    // v1 = a + b
    // v2 = v1 * c
    // v3 = v2 - d
    // return v3
    CFG cfg;
    cfg.entryBlock = "entry";
    cfg.addBlock("entry");

    cfg.addInstr("entry", IRInstr::makeBinop("+", "v1", "v0", "va"));
    cfg.addInstr("entry", IRInstr::makeBinop("*", "v2", "v1", "vb"));
    cfg.addInstr("entry", IRInstr::makeBinop("-", "v3", "v2", "vc"));
    cfg.addInstr("entry", IRInstr::makeReturn("v3"));

    return cfg;
}

CFG buildTestCFG2() {
    // Program with loop:
    // v_sum = 0
    // loop: v_sum = v_sum + v_i
    //       v_i = v_i + 1
    //       if v_i < v_n: goto loop
    CFG cfg;
    cfg.entryBlock = "init";
    cfg.addBlock("init");
    cfg.addBlock("loop");
    cfg.addBlock("exit");

    cfg.addInstr("init", IRInstr::makeMove("v_sum", "v_zero"));
    cfg.addInstr("init", IRInstr::makeMove("v_i", "v_zero"));

    cfg.addInstr("loop", IRInstr::makeBinop("+", "v_sum", "v_sum", "v_i"));
    cfg.addInstr("loop", IRInstr::makeBinop("+", "v_i", "v_i", "v_one"));
    cfg.addInstr("loop", IRInstr::makeBinop("<", "v_cond", "v_i", "v_n"));
    cfg.addInstr("loop", IRInstr::makeBranch("v_cond", "loop"));

    cfg.addInstr("exit", IRInstr::makeReturn("v_sum"));

    cfg.addEdge("init", "loop");
    cfg.addEdge("loop", "loop");
    cfg.addEdge("loop", "exit");

    return cfg;
}

CFG buildTestCFG3() {
    // Heavy register pressure program
    // Forces spilling with limited registers
    CFG cfg;
    cfg.entryBlock = "entry";
    cfg.addBlock("entry");

    // Many live variables simultaneously
    cfg.addInstr("entry", IRInstr::makeBinop("+", "va", "v0", "v1"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vb", "v2", "v3"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vc", "v4", "v5"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vd", "v6", "v7"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "ve", "v8", "v9"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vf", "v10", "v11"));

    // All of va-vf live here simultaneously!
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vr1", "va", "vb"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vr2", "vc", "vd"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vr3", "ve", "vf"));
    cfg.addInstr("entry", IRInstr::makeBinop("+", "vfinal", "vr1", "vr2"));
    cfg.addInstr("entry", IRInstr::makeReturn("vfinal"));

    return cfg;
}

void runRegisterAllocation(const std::string& name, CFG& cfg,
                            int numRegs) {
    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "REGISTER ALLOCATION: " << name << "\n";
    std::cout << "Available physical registers: " << numRegs << "\n";
    std::cout << std::string(65, '-') << "\n";

    cfg.print();

    // Step 1: Liveness Analysis
    LivenessAnalyzer liveness(cfg);
    liveness.computeLiveness();
    liveness.printLiveness();

    // Step 2: Build Interference Graph
    InterferenceGraph ig;
    ig.buildFromCFG(cfg);
    ig.print();

    // Step 3: Register Allocation
    RegisterAllocator allocator(ig, numRegs);
    allocator.allocate();
    allocator.printAllocation();
    allocator.analyzeSpillCost();
}

int main() {
    std::cout << "GRAPH COLORING REGISTER ALLOCATOR\n";
    std::cout << "Chaitin-Briggs Algorithm Implementation\n\n";

    // Test 1: Simple program with enough registers
    auto cfg1 = buildTestCFG1();
    runRegisterAllocation("Simple Linear (4 registers)", cfg1, 4);

    // Test 2: Loop program
    auto cfg2 = buildTestCFG2();
    runRegisterAllocation("Loop Program (3 registers)", cfg2, 3);

    // Test 3: High register pressure (forces spilling)
    auto cfg3 = buildTestCFG3();
    runRegisterAllocation("High Pressure (4 registers, expect spills)", cfg3, 4);

    // Test 4: Same high pressure with more registers
    auto cfg4 = buildTestCFG3();
    runRegisterAllocation("High Pressure (8 registers, no spills)", cfg4, 8);

    // Test 5: Register coalescing
    std::cout << "\n" << std::string(65, '=') << "\n";
    std::cout << "REGISTER COALESCING\n";
    std::cout << std::string(65, '-') << "\n";

    InterferenceGraph ig;
    ig.addEdge("v1", "v3");
    ig.addEdge("v2", "v3");
    // v1 and v2 don't interfere, v3 interferes with both

    RegisterCoalescer coalescer;
    std::vector<std::pair<std::string,std::string>> moves = {
        {"v1", "v4"},  // v4 = mov v1 (can coalesce if no interference)
        {"v2", "v5"},  // v5 = mov v2
        {"v1", "v2"}   // Cannot coalesce if they interfere
    };

    std::cout << "Coalescing opportunities:\n";
    auto coalesceMap = coalescer.coalesce(ig, moves);

    std::cout << "\n[Done] Register allocation complete!\n";
    std::cout << "This algorithm is used in PTXAS to allocate\n";
    std::cout << "GPU registers to virtual PTX registers.\n";

    return 0;
}
