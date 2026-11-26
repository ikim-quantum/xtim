// test_xdem_file — the xdem reader, emitter and composite (qeccore/xdem_file.hpp) on the SPEC §3.7 worked file:
//  * parse: D O r m tc A L = 3 2 3 2 3 8 3;
//  * compose({0, 1}): eta empty (the two rotations cancel by the cocycle) and the flip of line 2 (the Y2 line);
//  * emit(parse(text)) == text (the canonical text is a fixed point);
//  * `tc 4` is refused by name, naming the header line.
// Must end main() with REPORT() — see check.hpp.
#include "check.hpp"

#include <string>
#include <vector>

#include "qeccore/xdem_file.hpp"

using namespace qeccore::xdem;

static const std::string kEx422 =
    "xdem 1 detectors 3 observables 2 gauge 3 hidden 2 tc 3\n"
    "axis Q0 img D0 D1 D2 gauge g0\n"
    "axis Q1 img D1 D2 gauge g1\n"
    "axis Q2 img D0 D1 D2 L0 L1 gauge g2\n"
    "axis Q3 img D1 D2 L0 L1 gauge g0 g1 g2\n"
    "axis Q4 sign - img D0 D1 D2 L0 L1 gauge g2 hidden h0\n"
    "axis Q5 sign - img D1 D2 L0 L1 gauge g0 g1 g2 hidden h0\n"
    "axis Q6 sign - img D0 D1 D2 gauge g0 hidden h1\n"
    "axis Q7 sign - img D1 D2 gauge g1 hidden h1\n"
    "error(0.001) D0 D1 L1 x Q2 Q3 rot -Q2 +Q3\n"
    "error(0.001) D2 L0 x Q2 Q3 rot -Q2 +Q3\n"
    "error(0.001) D0 D1 D2 L0 L1\n";

int main() {
    const File F = parse(kEx422);
    CHECK(F.extended);
    CHECK_EQ(F.D, 3);
    CHECK_EQ(F.O, 2);
    CHECK_EQ(F.r, 3);
    CHECK_EQ(F.m, 2);
    CHECK_EQ(F.tc, (int64_t)3);
    CHECK_EQ(F.axes.size(), (size_t)8);
    CHECK_EQ(F.num_lines(), (size_t)3);

    const Composed c = compose(F, {0, 1});
    CHECK(c.eta.empty());
    const Composed y = compose(F, {2});
    CHECK(c.flip.w == y.flip.w);
    CHECK_EQ(c.flip.w.size(), (size_t)1);
    CHECK_EQ(c.flip.w[0], (uint64_t)0x1f);   // D0 D1 D2 L0 L1

    CHECK(emit(F) == kEx422);

    bool refused = false;
    std::string msg;
    try {
        std::string t = kEx422;
        t.replace(t.find("tc 3"), 4, "tc 4");
        parse(t);
    } catch (const XdemFileError& e) {
        refused = true;
        msg = e.what();
    }
    CHECK(refused);
    CHECK(msg == "xdem: line 1: tc 4 out of range (the file has 3 error lines)");

    REPORT();
}
