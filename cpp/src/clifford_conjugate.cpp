#include "qeccore/clifford_conjugate.hpp"
#include <vector>
namespace qeccore {
bool conjugate_by_gate(DiagPauliClifford& D, GateKind U, int a, int b, int c) {
    // A finalized cache describes the operator at finalize() time; any mutation makes
    // it stale, and a copy sharing the shared_ptr would silently serve pre-mutation
    // pairs. No production path finalizes before conjugating today — this reset makes
    // that ordering structurally impossible rather than an unguarded convention. The
    // null-check branch is predicted-true throughout the propagation walk (atoms are
    // never finalized), so the hot path pays one predictable compare.
    if (D.cache) D.cache.reset();
    return conjugate_layer(D.a, D.B, D.v, D.gamma, U, a, b, c);   // lean L6: the rule lives in the header (templated on the layer)
}
}
