/*
  Maskfish, a UCI jieqi engine derived from Pikafish and Stockfish
  Copyright (C) 2026 The Maskfish developers (see AUTHORS file)
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

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

#ifndef POSITION_H_INCLUDED
#define POSITION_H_INCLUDED

#include <array>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <deque>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "attacks.h"
#include "bitboard.h"
#include "nnue/features/half_ka_v2_hm.h"
#include "types.h"

namespace Maskfish {

// StateInfo struct stores information needed to restore a Position object to
// its previous state when we retract a move. Whenever a move is made on the
// board (by calling Position::do_move), a StateInfo object must be passed.

struct StateInfo {

    // Copied when making a move
    Key   pawnKey;
    Key   minorPieceKey;
    Key   nonPawnKey[COLOR_NB];
    Value majorMaterial[COLOR_NB];
    int   rule40;
    int   pliesFromNull;

    // Not copied when making a move (will be recomputed anyhow)
    Key        key;
    Bitboard   checkersBB;
    StateInfo* previous;
    Bitboard   blockersForKing[COLOR_NB];
    Bitboard   pinners[COLOR_NB];
    Bitboard   checkSquares[PIECE_TYPE_NB];
    bool       needFullCheck;
    Piece      capturedPiece;
    Move       move;
    bool       captureDark;
    // Squares attacked by the side not to move, by piece type (ALL_PIECES:
    // all of them), x-raying the king of the side to move: computed on demand
    // by threats_by() (UCI options ThreatCache, ThreatCacheMovePick)
    mutable Bitboard threats[PIECE_TYPE_NB];
    mutable bool     threatsComputed;
#ifndef NDEBUG
    // A dark piece has moved and do_flip() has not revealed it yet: the check
    // info (checkersBB, blockersForKing, pinners, checkSquares, needFullCheck)
    // is not valid. Debug builds only, see Position::check_info_valid().
    bool revealPending;
#endif
};


// A list to keep track of the position states along the setup moves (from the
// start position to the position just before the search starts). Needed by
// 'draw by repetition' detection. Use a std::deque because pointers to
// elements are not invalidated upon list resizing.
using StateListPtr = std::unique_ptr<std::deque<StateInfo>>;

// Position class stores information regarding the board representation as
// pieces, side to move, hash keys, etc. Important methods are
// do_move() and undo_move(), used by the search to update node info when
// traversing the search tree.
class Position {
   public:
    static void init();

    Position()                           = default;
    Position(const Position&)            = delete;
    Position& operator=(const Position&) = delete;

    // FEN string input/output
    Position&   set(const std::string& fenStr, StateInfo* si);
    Position&   set(const Position& pos, StateInfo* si);
    std::string fen() const;

    // Position representation
    Bitboard pieces() const;  // All pieces
    template<typename... PieceTypes>
    Bitboard pieces(PieceTypes... pts) const;
    Bitboard pieces(Color c) const;
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;
    Piece    piece_on(Square s) const;
    bool     empty(Square s) const;
    template<PieceType Pt>
    int count(Color c) const;
    template<PieceType Pt>
    int    count() const;
    Square king_square(Color c) const;
    bool   is_dark(Square s) const;

    // Checking
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    Bitboard pinners(Color c) const;

    // Attacks to/from a given square
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    template<Color c>
    void     update_blockers() const;
    Bitboard checkers_to(Color c, Square s) const;
    Bitboard checkers_to(Color c, Square s, Bitboard occupied) const;
    template<PieceType Pt>
    Bitboard attacks_by(Color c) const;
    // Squares attacked by the pieces of type pt of the side not to move (as
    // Pikafish), with the king of the side to move removed from the occupancy
    // (x-ray), cached in the StateInfo. threats_by(ALL_PIECES) & s is the same
    // as checkers_to(~side_to_move(), s, pieces() ^ king square | s) for a
    // square s next to the king.
    Bitboard threats_by(PieceType pt) const;
    // Pikafish's legality test and generation of king moves with
    // threats_by(ALL_PIECES) (UCI option ThreatCache): no functional change
    static inline bool useThreatCache = false;
    // The threats by lesser pieces of the move ordering from threats_by()
    // (UCI option ThreatCacheMovePick): x-rays the king, changes the ordering
    static inline bool useThreatCacheMovePick = false;

    // Properties of moves
    bool  legal(Move m) const;
    bool  capture(Move m) const;
    bool  gives_check(Move m) const;
    Piece moved_piece(Move m) const;
    Piece captured_piece() const;
    bool  move_dark(Move m) const;

    // Doing and undoing moves
    void       do_move(Move m, StateInfo& newSt);
    // With dts, the threats the move adds and removes (see DirtyThreats) are
    // appended to it (make threats=yes, UCI option ThreatsIncremental)
    DirtyPiece do_move(Move m, StateInfo& newSt, bool givesCheck, DirtyThreats* dts = nullptr);
    void       undo_move(Move m);
    void       do_null_move(StateInfo& newSt);
    void       undo_null_move();
    Piece do_flip(Square s, Piece pc, DirtyPiece* dp = nullptr, DirtyThreats* dts = nullptr);
    void undo_flip(Square s, Piece fromPc);

    // Take one piece out of / put it back into the pool of unrevealed pieces,
    // when the identity of a captured dark piece becomes known (setting up a
    // position, or trying each possible identity in the search). In the search
    // the caller must also refresh the NNUE accumulator (pool features).
    void remove_rest_piece(Piece pc);
    void restore_rest_piece(Piece pc);

    // Static Exchange Evaluation
    bool see_ge(Move m, int threshold = 0) const;

    // Accessing hash keys
    Key key() const;
    Key prefetch_key(Move m) const;
    Key flip_key(Square s, Piece pc) const;
    Key pawn_key() const;
    Key minor_piece_key() const;
    Key non_pawn_key(Color c) const;

    // Other properties of the position
    Color    side_to_move() const;
    int      game_ply() const;
    bool     rule_judge(Value& result, int ply = 0);
    int      rule40_count() const;
    uint16_t chased(Color c);
    Value    major_material(Color c) const;
    Value    major_material() const;
    // major_material() with each dark piece counted at the expected value of the
    // rooks, cannons and knights of its owner's pool
    Value    jieqi_major_material() const;
    // Expected value of a dark piece of the given color: the average value
    // of the pieces left in its pool
    Value dark_value(Color c) const;
    // Value of the piece on a square, a dark piece counting at dark_value()
    Value piece_value(Square s) const;
    // Index of a captured piece in the capture history: its type, or
    // NO_PIECE_TYPE (unused otherwise) for a dark piece, whose type is unknown
    PieceType capture_type(Square s) const;
    // Value and capture history index of the piece captured by the last move
    Value     captured_value() const;
    PieceType captured_type() const;
#ifdef JIEQI_NNUE_ATT
    uint64_t mid_encoding(Color c) const { return midEncoding[c]; }
#endif

    const int&                         rest_piece(Piece pc) const;
    // Allocation-free variant used in the search hot path (chance nodes).
    static constexpr int MaxRestPieceTypes = 6;
    using RestPieceList                    = std::array<std::pair<Piece, int>, MaxRestPieceTypes>;
    int                                rest_pieces(Color c, RestPieceList& pieces) const;
    std::vector<std::pair<Piece, int>> rest_pieces(Color c) const;

    // Position consistency check, for debugging
    bool pos_is_ok() const;
    // Whether the position can be searched (kings, piece counts, pool, side not
    // to move not in check), with the reason when it cannot
    bool pool_is_consistent(std::string* reason = nullptr) const;
    bool is_playable() const;
    void flip();

    StateInfo* state() const;

    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);

   private:
    // Initialization helpers (used while setting up a position)
    void set_state() const;
    void set_check_info() const;
    void set_threats() const;

    // Threat bookkeeping (see DirtyThreats): class of the piece on a square,
    // and the threats added (add) or removed by putting or removing, on s, a
    // piece of class cls that attacks like a piece of type pt. The piece must be
    // on the board when it is called, the rest of the board as it is before
    // (for a removal) or after (for an addition). Without computeRay, only the
    // threats from and to s (the class of the piece on s changes, not the
    // occupancy).
    int  threat_class(Square s) const;
    void update_piece_threats(
      int cls, PieceType pt, bool add, Square s, DirtyThreats& dts, bool computeRay) const;

    // Other helpers
    void                  move_piece(Square from, Square to);
    std::pair<Piece, int> do_move(Move m);
    void                  undo_move(Move m, Piece captured, int id = 0);
    Value                 detect_chases(int d, int ply = 0);
    bool                  chase_legal(Move m) const;

    template<bool AfterMove = false>
    Key adjust_key40(Key k) const;

    // False between the move of a dark piece and its reveal (debug builds)
    bool check_info_valid() const;
    void set_reveal_pending(bool pending);

    // Data members
    Piece      board[SQUARE_NB];
    Bitboard   byTypeBB[PIECE_TYPE_NB + 1];
    Bitboard   byColorBB[COLOR_NB];
    Square     kingSquare[COLOR_NB];
    int        pieceCount[PIECE_NB];
    StateInfo* st;
    int        gamePly;
    Color      sideToMove;
    int        restPieces[PIECE_NB];
#ifdef JIEQI_NNUE_ATT
    uint64_t   midEncoding[COLOR_NB];  // see HalfKAv2_hm::MidMirrorEncoding
#endif

    // Bloom filter for fast repetition filtering
    BloomFilter filter;

    // Board for chasing detection
    int idBoard[SQUARE_NB];
};

std::ostream& operator<<(std::ostream& os, const Position& pos);

inline Color Position::side_to_move() const { return sideToMove; }

inline Piece Position::piece_on(Square s) const {
    assert(is_ok(s));
    return board[s];
}

inline bool Position::empty(Square s) const { return piece_on(s) == NO_PIECE; }

inline Piece Position::moved_piece(Move m) const { return piece_on(m.from_sq()); }

inline Bitboard Position::pieces() const { return byTypeBB[ALL_PIECES]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(PieceTypes... pts) const {
    return (byTypeBB[pts] | ...);
}

inline Bitboard Position::pieces(Color c) const { return byColorBB[c]; }

template<typename... PieceTypes>
inline Bitboard Position::pieces(Color c, PieceTypes... pts) const {
    return pieces(c) & pieces(pts...);
}

template<PieceType Pt>
inline int Position::count(Color c) const {
    return pieceCount[make_piece(c, Pt)];
}

template<PieceType Pt>
inline int Position::count() const {
    return count<Pt>(WHITE) + count<Pt>(BLACK);
}

inline Square Position::king_square(Color c) const { return kingSquare[c]; }

inline bool Position::is_dark(Square s) const { return pieces(DARK) & s; }

inline Bitboard Position::attackers_to(Square s) const { return attackers_to(s, pieces()); }

inline Bitboard Position::checkers_to(Color c, Square s) const {
    return checkers_to(c, s, pieces());
}

template<PieceType Pt>
inline Bitboard Position::attacks_by(Color c) const {

    Bitboard threats   = 0;
    Bitboard attackers = pieces(c, Pt);
    while (attackers)
    {
        const Square s = pop_lsb(attackers);
        if constexpr (Pt == PAWN)
            threats |= Attacks::attacks_bb(PAWN, s, c);
        else if constexpr (Pt == ADVISOR)
            // A dark advisor must stay in the palace
            threats |= Attacks::attacks_bb(ADVISOR, s) & (is_dark(s) ? Palace : ~Bitboard(0));
        else
            threats |= Attacks::attacks_bb(Pt, s, pieces());
    }
    return threats;
}

inline Bitboard Position::threats_by(PieceType pt) const {
    assert(check_info_valid());
    if (!st->threatsComputed)
        set_threats();
    return st->threats[pt];
}

inline bool Position::check_info_valid() const {
#ifndef NDEBUG
    return !st->revealPending;
#else
    return true;
#endif
}

inline void Position::set_reveal_pending([[maybe_unused]] bool pending) {
#ifndef NDEBUG
    st->revealPending = pending;
#endif
}

inline Bitboard Position::checkers() const {
    assert(check_info_valid());
    return st->checkersBB;
}

inline Bitboard Position::blockers_for_king(Color c) const {
    assert(check_info_valid());
    return st->blockersForKing[c];
}

inline Bitboard Position::pinners(Color c) const {
    assert(check_info_valid());
    return st->pinners[c];
}

inline Bitboard Position::check_squares(PieceType pt) const {
    assert(check_info_valid());
    return st->checkSquares[pt];
}

inline Key Position::key() const { return adjust_key40(st->key); }

template<bool AfterMove>
inline Key Position::adjust_key40(Key k) const {
    return (st->rule40 < 14 - AfterMove ? k : k ^ make_key((st->rule40 - (14 - AfterMove)) / 8))
         ^ (filter[k] ? make_key(14) : 0);
}

inline Key Position::pawn_key() const { return st->pawnKey; }

inline Key Position::minor_piece_key() const { return st->minorPieceKey; }

inline Key Position::non_pawn_key(Color c) const { return st->nonPawnKey[c]; }

inline Value Position::major_material(Color c) const { return st->majorMaterial[c]; }

inline Value Position::major_material() const {
    return major_material(WHITE) + major_material(BLACK);
}

inline Value Position::jieqi_major_material() const {
    Value total = major_material();
    for (Color c : {WHITE, BLACK})
    {
        const Bitboard darkBB = pieces(c, DARK);
        const int      dark   = popcount(darkBB);
        if (!dark)
            continue;

        // major_material() counts a dark piece at the value of the piece that
        // starts on its square: replace that value by the expected one below
        for (Bitboard b = darkBB & pieces(ROOK, CANNON, KNIGHT); b;)
            total -= PieceValue[piece_on(pop_lsb(b))];

        int poolCount = 0, poolMajor = 0;
        for (PieceType pt = ROOK; pt < KING; ++pt)
        {
            const Piece pc = make_piece(c, pt);
            poolCount += restPieces[pc];
            if (pt & 1)  // rook, cannon, knight
                poolMajor += restPieces[pc] * PieceValue[pc];
        }
        if (poolCount)
            total += dark * poolMajor / poolCount;
    }
    return total;
}


inline Value Position::dark_value(Color c) const {
    int count = 0, value = 0;
    for (PieceType pt = ROOK; pt < KING; ++pt)
    {
        const Piece pc = make_piece(c, pt);
        count += restPieces[pc];
        value += restPieces[pc] * PieceValue[pc];
    }
    return count ? Value(value / count) : VALUE_ZERO;
}

inline Value Position::piece_value(Square s) const {
    return is_dark(s) ? dark_value(color_of(piece_on(s))) : PieceValue[piece_on(s)];
}

inline PieceType Position::capture_type(Square s) const {
    return is_dark(s) ? NO_PIECE_TYPE : type_of(piece_on(s));
}

inline Value Position::captured_value() const {
    return st->captureDark ? dark_value(color_of(st->capturedPiece)) : PieceValue[st->capturedPiece];
}

inline PieceType Position::captured_type() const {
    return st->captureDark ? NO_PIECE_TYPE : type_of(st->capturedPiece);
}

inline const int& Position::rest_piece(Piece pc) const { return restPieces[pc]; }

inline int Position::rest_pieces(Color c, RestPieceList& pieces) const {
    int             count        = 0;
    const PieceType pieceOrder[] = {ROOK, CANNON, KNIGHT, ADVISOR, BISHOP, PAWN};
    for (PieceType pt : pieceOrder)
    {
        Piece pc = make_piece(c, pt);
        if (restPieces[pc])
            pieces[count++] = {pc, restPieces[pc]};
    }
    return count;
}

inline std::vector<std::pair<Piece, int>> Position::rest_pieces(Color c) const {
    RestPieceList list;
    const int     count = rest_pieces(c, list);
    return {list.begin(), list.begin() + count};
}

inline int Position::game_ply() const { return gamePly; }

inline int Position::rule40_count() const { return st->rule40; }

inline bool Position::capture(Move m) const {
    assert(m.is_ok());
    return !empty(m.to_sq());
}

inline Piece Position::captured_piece() const { return st->capturedPiece; }

inline bool Position::move_dark(Move m) const { return pieces(DARK) & m.from_sq(); }

inline int Position::threat_class(Square s) const {
    return is_dark(s) ? int(color_of(piece_on(s))) << 3 : int(piece_on(s));
}

inline void Position::put_piece(Piece pc, Square s) {

    board[s] = pc;
    byTypeBB[ALL_PIECES] |= byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]++;
#ifdef JIEQI_NNUE_ATT
    midEncoding[color_of(pc)] += Eval::NNUE::Features::HalfKAv2_hm::MidMirrorEncoding[pc][s];
#endif
}

inline void Position::remove_piece(Square s) {

    Piece pc = board[s];
    byTypeBB[ALL_PIECES] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), ALL_PIECES)]--;
#ifdef JIEQI_NNUE_ATT
    midEncoding[color_of(pc)] -= Eval::NNUE::Features::HalfKAv2_hm::MidMirrorEncoding[pc][s];
#endif
}

inline void Position::move_piece(Square from, Square to) {

    Piece    pc     = board[from];
    Bitboard fromTo = from | to;
    byTypeBB[ALL_PIECES] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to]   = pc;
    if (type_of(pc) == KING)
        kingSquare[color_of(pc)] = to;
#ifdef JIEQI_NNUE_ATT
    midEncoding[color_of(pc)] += Eval::NNUE::Features::HalfKAv2_hm::MidMirrorEncoding[pc][to]
                               - Eval::NNUE::Features::HalfKAv2_hm::MidMirrorEncoding[pc][from];
#endif
}

inline void Position::do_move(Move m, StateInfo& newSt) {
    do_move(m, newSt, gives_check(m));
}

inline StateInfo* Position::state() const { return st; }

inline Position& Position::set(const Position& pos, StateInfo* si) {

    set(pos.fen(), si);

    // Special cares for bloom filter
    std::memcpy(&filter, &pos.filter, sizeof(BloomFilter));

    return *this;
}

}  // namespace Maskfish

#endif  // #ifndef POSITION_H_INCLUDED
