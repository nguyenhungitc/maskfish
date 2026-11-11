/*
  Maskfish, a UCI jieqi engine
  Copyright (C) 2026 The Maskfish developers (see AUTHORS file)

  Maskfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Maskfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Board state, move generation and make/unmake for Jieqi.
//
// Information model (from the point of view of the player the engine plays for):
//   urn[c][slot]  candidate identities still unaccounted for among c's face-down pieces.
//   A face-down piece reveals as slot t with probability urn[c][t] / urnTotal[c].
// Captured face-down pieces whose identity the observer did not see stay in the urn;
// by exchangeability the live pieces are then still a uniform draw from the urn.
// Inside search, capturing a face-down piece never branches on its identity for the
// same reason (the urn is left unchanged), while moving one is a chance node.
#pragma once
#include "tables.h"
#ifdef MASKFISH_BITBOARD
#include "bitboard.h"
#endif
#include "types.h"

#include <bitset>
#include <string>
#include <vector>

namespace jq {

struct StateInfo {
    uint64_t key;        // key before the move
    Piece captured;
    int8_t urnSlot;      // slot drawn by the mover's reveal, -1 if none
    int16_t noCapture;   // counter before the move
    int16_t lastIrrev;   // irreversible boundary before the move
    uint64_t skey;       // structure key before the move
};

constexpr int NO_CAPTURE_LIMIT = 80;
// 1: a repetition without checks is adjudicated by the perpetual-chase rule of the match referee
// (tools/native/chase.cpp); 0: every such repetition is a draw.
extern int g_chaseRule;
constexpr int MAX_MOVES = 160;

class Position {
public:
#ifdef MASKFISH_BITBOARD
    // The sets of pieces, kept beside board[] by set_fen and do / undo_move: occBB all pieces; byColorBB[c];
    // byRoleBB[t] the pieces that move as t (a face-down piece under the role of its start point, T.darkRole);
    // darkBB the face-down pieces of both colours.
    Bitboard occBB = 0;
    Bitboard byColorBB[2] = {0, 0};
    Bitboard byRoleBB[9] = {};
    Bitboard darkBB = 0;
#endif
    Piece board[NUM_SQ];
    Color stm = RED;
    int8_t urn[2][NUM_SLOTS];
    int8_t urnTotal[2];
    int8_t darkCount[2];
    int8_t kingSq[2];
    int16_t noCapture = 0;
    uint8_t plist[2][16];
    int8_t pcount[2];
    int8_t pidx[NUM_SQ];
    uint64_t key = 0;
    // Structure key: the xor of the piece keys of the soldiers and the face-down pieces, kept by do_move. The
    // search indexes its correction history by it; it is not part of key.
    uint64_t skey = 0;
    // The side not to move is not in check: true of every position reached by legal moves, tested by set_fen for
    // the position it is given. do_move relies on it to find the new check flag from the move alone.
    bool checkFree = false;
    // Game + search history: key and "side to move is in check" per position.
    std::vector<uint64_t> keyHist;
    std::vector<uint8_t> checkHist;
    std::vector<Move> moveHist;  // moveHist[j]: base move played from position j (MOVE_NULL for a null move)
    int lastIrrev = 0;  // index into keyHist of the latest position after an irreversible move

    Position() { keyHist.reserve(1024); checkHist.reserve(1024); moveHist.reserve(1024); }

    // FEN: board (ranks 9..0, X/x = face down), side, pools, half-move clock, move number.
    bool set_fen(const std::string& fen, std::string* error = nullptr);
    // History token from-to + optional reveal letter (mover colour) + optional captured identity (victim colour) or 'x'.
    bool apply_token(const std::string& token, std::string* error = nullptr);
    // Same as apply_token with parsed parts (reveal / captured identity NO_TYPE when not observed).
    bool apply_observed(Move base, PieceType reveal, PieceType capturedId, std::string* error = nullptr);
    std::string fen() const;
    std::string pretty() const;

    PieceType role(int s) const {
        const Piece p = board[s];
        return type_of(p) == DARK ? PieceType(T.darkRole[color_of(p)][s]) : type_of(p);
    }
    bool is_dark_move(Move m) const { return type_of(board[from_sq(m)]) == DARK; }
    bool is_capture(Move m) const { return board[to_sq(m)] != NO_PIECE; }

#ifdef MASKFISH_BITBOARD
    // Is square k attacked by colour e on a board that differs from this one: the occupancy is occ, e's pieces
    // are E (a subset of occ), and square x (if >= 0) holds a face-up piece of e of type xt? The roles and the
    // face-down set are this position's. The answer of king_attacked on that board.
    bool attacked_bb(Color e, int k, Bitboard occ, Bitboard E, int x = -1, PieceType xt = NO_TYPE) const;
    bool ray_attacked_bb(Color e, int k, int d, Bitboard occ, Bitboard E, int x = -1, PieceType xt = NO_TYPE) const;
    void compute_bitboards();     // the sets from board[]
    bool bitboards_ok() const;    // the sets are those of board[] (tests)
#endif
    bool king_attacked(Color c, int k) const;
    bool ray_attacked(Color c, int k, int d) const;  // rook / cannon / facing-king attack along one ray
    bool in_check() const { return king_attacked(stm, kingSq[stm]); }
    // Pseudo-legal generation of base moves (no reveal). Captures-only variant for quiescence.
    int gen_moves(Move* list) const;
    template <bool CAP> int gen(Move* list) const;
    int gen_captures(Move* list) const;
    // Own-king safety for a pseudo-legal base move; identity of a dark mover is irrelevant.
    bool legal(Move m) const;
    // Whether a legal move by a face-up piece checks the opponent (false for face-down movers).
    bool gives_check(Move m) const;
    // A face-down mover from -> to revealed as t: does the revealed piece itself attack the enemy king?
    // reveal_may_check is a cheap necessary condition for any identity to differ.
    bool reveal_gives_check(int from, int to, PieceType t) const;
    bool reveal_may_check(int to) const;
    // Square attacked by `by` (king-centric test; adequate for capture safety heuristics).
    bool attacked(Color by, int s) const { return king_attacked(~by, s); }
    int gen_legal(Move* list) const;
    bool has_legal_move() const;

    // The search may hand do_move the gives_check(m) it has already worked out for a face-up mover m of this
    // position; do_move then takes it as the new side's check flag and saves its own test. The hint is matched by
    // the move, the key and the length of the history, and is exact (gives_check is, for a legal move).
    void check_hint(Move m, bool givesCheck) {
        hintMove = m;
        hintKey = key;
        hintLen = uint32_t(keyHist.size());
        hintVal = givesCheck;
    }
    // m must carry a reveal type when the mover is face down.
    void do_move(Move m, StateInfo& st);
    uint64_t key_after(Move m) const;  // the key do_move(m) will produce (m carries the reveal for a dark mover)
    void undo_move(Move m, const StateInfo& st);
    void do_null(StateInfo& st);
    void undo_null(const StateInfo& st);

    // 0: none, 1: draw, 2: side to move wins (opponent perpetually checked or chased), 3: side to move loses.
    int repetition_status() const;
    std::bitset<NUM_SQ> chase_threats(Color actor) const;  // enemy squares actor threatens under the chase rule
    bool has_major(Color c) const;
    uint64_t compute_key() const;
    uint64_t compute_skey() const;
    bool valid(std::string* error = nullptr) const;

private:
    Move hintMove = MOVE_NONE;
    uint64_t hintKey = 0;
    uint32_t hintLen = 0;
    bool hintVal = false;
    static uint64_t skey_of(Piece p, int s) {
        return type_of(p) == SOLDIER || type_of(p) == DARK ? T.zPiece[p][s] : 0;
    }
    int chase_verdict(int first) const;
#ifdef MASKFISH_BITBOARD
    void bb_put(int s, Piece p) {
        const Color c = color_of(p);
        const PieceType t = type_of(p);
        const Bitboard b = sq_bb(s);
        occBB |= b;
        byColorBB[c] |= b;
        if (t == DARK) { darkBB |= b; byRoleBB[T.darkRole[c][s]] |= b; }
        else byRoleBB[t] |= b;
    }
    void bb_clear(int s, Piece p) {
        const Color c = color_of(p);
        const PieceType t = type_of(p);
        const Bitboard b = ~sq_bb(s);
        occBB &= b;
        byColorBB[c] &= b;
        if (t == DARK) { darkBB &= b; byRoleBB[T.darkRole[c][s]] &= b; }
        else byRoleBB[t] &= b;
    }
    // Bitboard delta of a move from -> to by pc (revealed as rev if face down) capturing cap; applied by
    // do_move and again (XOR is its own inverse) by undo_move.
    void bb_move(Color us, int from, int to, Piece pc, PieceType rev, Piece cap) {
        const Bitboard bf = sq_bb(from), bt = sq_bb(to);
        if (cap) {
            const Color th = ~us;
            byColorBB[th] ^= bt;
            if (type_of(cap) == DARK) { darkBB ^= bt; byRoleBB[T.darkRole[th][to]] ^= bt; }
            else byRoleBB[type_of(cap)] ^= bt;
            occBB ^= bf;
        } else occBB ^= bf | bt;
        byColorBB[us] ^= bf | bt;
        if (type_of(pc) == DARK) { darkBB ^= bf; byRoleBB[T.darkRole[us][from]] ^= bf; byRoleBB[rev] ^= bt; }
        else byRoleBB[type_of(pc)] ^= bf | bt;
    }
#endif
    void add_piece(int s, Piece p);
    void remove_piece(int s);
    void move_piece(int from, int to);
    void push_history() { keyHist.push_back(key); checkHist.push_back(uint8_t(in_check())); }
};

std::string move_str(Move m);
Move parse_move_coords(const std::string& s);
char piece_char(Piece p);

extern const char* START_FEN;

}  // namespace jq
