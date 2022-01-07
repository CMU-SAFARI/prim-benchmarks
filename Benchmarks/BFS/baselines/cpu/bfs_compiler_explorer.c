// TODO Mark regions

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUND_UP_TO_MULTIPLE_OF_2(x) ((((x) + 1) / 2) * 2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x) ((((x) + 7) / 8) * 8)
#define ROUND_UP_TO_MULTIPLE_OF_64(x) ((((x) + 63) / 64) * 64)

struct CSRGraph {
    uint32_t numNodes;
    uint32_t numEdges;
    uint32_t *nodePtrs;
    uint32_t *neighborIdxs;
};

static void freeCSRGraph(struct CSRGraph csrGraph) {
    free(csrGraph.nodePtrs);
    free(csrGraph.neighborIdxs);
}

int main() {
    // Initialize BFS data structures
    volatile struct CSRGraph csrGraph;

    csrGraph.numNodes = 1024;
    csrGraph.numEdges = 1024;
    csrGraph.nodePtrs = (uint32_t *)malloc(sizeof(uint32_t) * (csrGraph.numNodes + 1));
    csrGraph.neighborIdxs = (uint32_t *)malloc(sizeof(uint32_t) * csrGraph.numEdges);

    volatile uint32_t *nodeLevel = (uint32_t *)malloc(csrGraph.numNodes * sizeof(uint32_t));
    volatile uint32_t *nodeLevelRef = (uint32_t *)malloc(csrGraph.numNodes * sizeof(uint32_t));
    for (uint32_t i = 0; i < csrGraph.numNodes; ++i) {
        nodeLevel[i] = UINT32_MAX;    // Unreachable
        nodeLevelRef[i] = UINT32_MAX; // Unreachable
    }
    volatile uint32_t srcNode = 0;

    // Initialize frontier double buffers
    volatile uint32_t *buffer1 = (uint32_t *)malloc(csrGraph.numNodes * sizeof(uint32_t));
    volatile uint32_t *buffer2 = (uint32_t *)malloc(csrGraph.numNodes * sizeof(uint32_t));
    volatile uint32_t *prevFrontier = buffer1;
    volatile uint32_t *currFrontier = buffer2;

    // start_region();
    nodeLevel[srcNode] = 0;
    prevFrontier[0] = srcNode;
    uint32_t numPrevFrontier = 1;
    for (uint32_t level = 1; numPrevFrontier > 0; ++level) {

        uint32_t numCurrFrontier = 0;

        for (uint32_t i = 0; i < numPrevFrontier; ++i) {
            uint32_t node = prevFrontier[i];
            for (uint32_t edge = csrGraph.nodePtrs[node]; edge < csrGraph.nodePtrs[node + 1]; ++edge) {
                uint32_t neighbor = csrGraph.neighborIdxs[edge];
                uint32_t justVisited = 0;

                if (nodeLevel[neighbor] == UINT32_MAX) { // Node not previously visited
                    nodeLevel[neighbor] = level;
                    justVisited = 1;
                }

                if (justVisited) {
                    uint32_t currFrontierIdx;

                    currFrontierIdx = numCurrFrontier++;

                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }

        // Swap buffers
        uint32_t *tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
        numPrevFrontier = numCurrFrontier;
    }
    // end_region();

    // Deallocate data structures
    freeCSRGraph(csrGraph);
    free(nodeLevel);
    free(buffer1);
    free(buffer2);

    return 0;
}
