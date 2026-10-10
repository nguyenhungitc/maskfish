#!/bin/bash
# Verify perft numbers of the Jieqi move generator.
#
# Run from the src directory, after building: ../tests/perft.sh
# Like 'go', 'go perft' needs a compatible network (EvalFile).
#
# A move of a dark piece is expanded into one child per kind of piece left in
# the mover's pool (see src/perft.h). There is no independent reference for
# Jieqi perft, so these numbers are regression values recorded with this
# engine: any change must be explained by an intended change of the rules or
# of the move generation.

TESTS_FAILED=0

echo "perft testing started"

run_test() {
  local pos="$1"
  local depth="$2"
  local expected="$3"

  echo -n "Testing depth $depth: ${pos:0:50}... "

  local nodes
  nodes=$(printf "position %s\ngo perft %s\nquit\n" "$pos" "$depth" | ./pikafish 2>&1 \
          | awk '/^Nodes searched:/ {print $3}')

  if [ "$nodes" = "$expected" ]; then
    echo "OK"
  else
    echo "FAILED (expected $expected, got ${nodes:-nothing})"
    TESTS_FAILED=1
  fi
}

# Start position: every move reveals a piece
run_test "startpos" 3 2593799
run_test "startpos" 4 538328131

# Middle game and endgame positions with dark pieces (from the benchmark)
run_test "fen x1xxkxxxx/9/c6x1/2x6/a3r1n1p/R1P3A2/4X3X/1X2B1AX1/4P4/1C2KXX1X w R1A0C1P3N1B1r0a1c1p4n1b2 1 8" 3 963202
run_test "fen 3xkx1xx/9/2a1p2ap/9/9/2r1P4/1R4p2/1CP1n3r/4N4/4K1CB1 w b2p2 0 25" 4 1354415
run_test "fen 9/5k1c1/2R1p4/7C1/p7p/4P4/X8/5r3/P3K4/2X4c1 w B1P1 3 51" 4 1512080

# All pieces revealed: Jieqi rules (advisors may leave the palace, bishops
# may cross the river), so the numbers differ from xiangqi
run_test "fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - 0 1" 4 3958056

echo "perft testing completed"

if [ $TESTS_FAILED -ne 0 ]; then
  echo "Some tests failed"
  exit 1
fi
