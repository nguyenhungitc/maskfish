#!/bin/bash
# Invalid positions and moves must be reported and never searched.
#
# Run from the src directory, after building: ../tests/invalid_input.sh
# No network is needed: an invalid position is rejected before any evaluation.
# Best run on a build with sanitizers (make build debug=yes sanitize="address
# undefined"): these inputs used to overflow fixed-size tables (NNUE feature
# lists, layer stack buckets, move lists) or return a move for the wrong side.

TESTS_FAILED=0

echo "invalid input testing started"

# run_test <name> <position command> <expected text in the output> [perft]
# With "perft", perft must refuse the position too (an invalid FEN; after an
# invalid move, perft still runs on the position before that move).
run_test() {
  local name="$1"
  local position="$2"
  local expected="$3"
  local perft="$4"

  echo -n "Testing $name... "

  local output
  output=$(printf "%s\ngo perft 1\nevalraw\ngo depth 1\nisready\nquit\n" "$position" \
           | timeout 60 ./maskfish 2>&1)
  local rc=$?

  if [ $rc -ne 0 ]; then
    echo "FAILED (exit code $rc)"
    echo "$output" | tail -20
    TESTS_FAILED=1
  elif ! echo "$output" | grep -q "$expected"; then
    echo "FAILED (no '$expected' in the output)"
    echo "$output" | tail -20
    TESTS_FAILED=1
  elif ! echo "$output" | grep -q "^bestmove (none)"; then
    echo "FAILED (the position was searched)"
    echo "$output" | tail -20
    TESTS_FAILED=1
  elif [ "$perft" = "perft" ] && ! echo "$output" | grep -q "^Nodes searched: 0"; then
    echo "FAILED (perft ran on the position)"
    echo "$output" | tail -20
    TESTS_FAILED=1
  else
    echo "OK"
  fi
}

# More than 32 pieces on the board (layer stack bucket out of range)
run_test "too many pieces" \
  "position fen 4k4/9/ppppppppp/ppppppppp/ppppppppp/PPPPPPPPP/PPPPPPPPP/PPPPPPPPP/9/4K4 w - 0 1" \
  "Invalid position: too many pieces" perft

# Pool beyond the initial numbers (NNUE feature list overflow)
run_test "pool too large" \
  "position fen xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R5A5C5P5N5B5r5a5c5p5n5b5 0 1" \
  "Invalid position: too many pieces" perft

# Revealed pieces plus pool beyond the initial number of rooks
run_test "pool and board" \
  "position fen 4k4/9/9/9/9/9/9/9/R7R/4K4 w R1 0 1" \
  "Invalid position: too many pieces" perft

# Nine rooks: more legal moves than MAX_MOVES used to allow
run_test "too many rooks" \
  "position fen 4k4/8R/7R1/3R5/6R2/2R1p4/5R3/1R7/8R/R3K4 w - 0 1" \
  "Invalid position: too many pieces" perft

# A rank with too many squares must not wrap around the board
run_test "malformed rank" \
  "position fen 4k4/99999999999999999999999999999R/9/9/9/9/9/9/9/4K4 w - 0 1" \
  "Invalid position" perft

# King outside its palace
run_test "king outside palace" \
  "position fen 4k4/9/9/9/9/9/9/9/9/K8 w - 0 1" \
  "Invalid position: white king is outside its palace" perft

# More dark pieces than pieces in the pool
run_test "dark pieces without pool" \
  "position fen xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w - 0 1" \
  "Invalid position: white has more dark pieces" perft

# A move of a dark piece without the revealed piece: searching the position
# before it would return a move for the wrong side
run_test "missing revealed piece" \
  "position startpos moves a3a4P a6a5" \
  "Invalid move a6a5: missing the revealed piece"

run_test "illegal move" \
  "position startpos moves a3a4P e9e7" \
  "Invalid move e9e7: illegal move"

echo "invalid input testing completed"

if [ $TESTS_FAILED -ne 0 ]; then
  echo "Some tests failed"
  exit 1
fi
