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

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

#include "attacks.h"
#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "uci.h"

using std::string;

namespace Maskfish {

using namespace Attacks;

namespace Zobrist {

Key psq[PIECE_NB + 1][SQUARE_NB];
Key rest[PIECE_NB][MAX_REST_COUNT + 1];
Key side, noPawns;
}

namespace {

constexpr std::string_view PieceToChar(" RACPNBK racpnbkXx");

constexpr std::string_view DarkPieces("RNBAKABNR"
                                      "........."
                                      ".C.....C."
                                      "P.P.P.P.P"
                                      "........."
                                      "........."
                                      "p.p.p.p.p"
                                      ".c.....c."
                                      "........."
                                      "rnbakabnr");

static constexpr Piece Pieces[] = {W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING,
                                   B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING};
}  // namespace

// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_9;; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
        {
            Square sq = make_square(f, r);
            Piece  pc = pos.piece_on(sq);
            os << " | " << (pos.is_dark(sq) ? "Xx"[color_of(pc)] : PieceToChar[pc]);
        }

        os << " | " << int(r) << "\n +---+---+---+---+---+---+---+---+---+\n";

        if (r == RANK_0)
            break;
    }

    os << "   a   b   c   d   e   f   g   h   i\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    return os;
}


// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A0; s <= SQ_I9; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();

    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        Zobrist::psq[DARK_PIECE][s] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();

    // Keys for the pool of unrevealed pieces. They must not reuse psq[][]:
    // e.g. psq[W_ROOK][SQ_A0] is the key of a rook on a0, and reusing it for
    // "one rook in the pool" makes both XOR to zero. Drawn after all other keys
    // so that the existing keys are unchanged.
    for (Piece pc : Pieces)
        for (int i = 0; i <= MAX_REST_COUNT; ++i)
            Zobrist::rest[pc][i] = rng.rand<Key>();
}


// Initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.
Position& Position::set(const string& fenStr, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A Jieqi FEN string contains five fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 9 and ending with rank 0. Within each rank, the contents of each
      square are described from file A through file I. Each piece is identified
      by a single letter: white pieces use upper-case letters ("RACPNBK"), black
      pieces lower-case ones ("racpnbk"), and "X" / "x" is a dark (face down)
      piece of white / black, accepted only on a starting square of its own side
      other than the king's. Blank squares are noted using digits 1 through 9
      (the number of blank squares), and "/" separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Pool of unrevealed pieces: <piece><count> pairs, e.g.
      R2A2C2P5N2B2r2a2c2p5n2b2 (kinds that are not listed count as 0).

   4) Number of plies since the last capture, for the 40-move rule (80 plies).

   5) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

    unsigned char      token;
    size_t             idx;
    std::istringstream ss(fenStr);

    // Current rank and file of the board field, tracked as plain ints so that
    // a malformed field (too many squares in a rank, too many ranks) can never
    // wrap around the 8-bit Square type or place a piece on an occupied square
    int rank = RANK_9, file = FILE_A;

    std::memset(this, 0, sizeof(Position));
#ifdef JIEQI_NNUE_ATT
    midEncoding[WHITE] = midEncoding[BLACK] = Eval::NNUE::FeatureSet::BalanceEncoding;
#endif

    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // 1. Piece placement
    while ((ss >> token) && !isspace(token))
    {
        if (isdigit(token))
            file = std::min(file + (token - '0'), int(FILE_NB));  // Advance the given number of files

        else if (token == '/')
        {
            --rank;
            file = FILE_A;
        }

        else if ((idx = PieceToChar.find(token)) != string::npos && idx != 0)
        {
            // Ignore anything that would be placed outside the board
            if (rank < RANK_0 || rank > RANK_9 || file < FILE_A || file > FILE_I)
            {
                ++file;
                continue;
            }

            const Square sq = make_square(File(file), Rank(rank));
            ++file;

            if (tolower(token) == 'x')
            {
                // A dark piece moves like the piece that starts on its square,
                // so it can only stand on a starting square of its own color,
                // and the king is never hidden.
                const char start = DarkPieces[sq];
                if (start == '.' || tolower(start) == 'k'
                    || bool(isupper(start)) != (token == 'X'))
                    continue;  // Treat the invalid dark piece as an empty square
                idx = PieceToChar.find(start);
                byTypeBB[DARK] |= sq;
            }
            put_piece(Piece(idx), sq);
            if (type_of(Piece(idx)) == KING)
                kingSquare[color_of(Piece(idx))] = sq;
        }
    }

    // 2. Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 3. Rest pieces: a list of <piece><count>, e.g. R2A2C2P5N2B2r2a2c2p5n2b2.
    // Only non-king pieces are accepted, counts are single digits.
    while ((ss >> token) && !isspace(token))
    {
        idx = PieceToChar.find(token);
        if (idx == string::npos || idx == 0 || idx >= size_t(PIECE_NB)
            || type_of(Piece(idx)) == KING || type_of(Piece(idx)) == NO_PIECE_TYPE)
            continue;

        if (!(ss >> token) || isspace(token))
            break;

        if (isdigit(token))
            restPieces[idx] = std::min(int(token - '0'), MAX_REST_COUNT);
    }

    // 4-5. Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule40 >> gamePly;
    st->rule40 = std::max(st->rule40, 0);

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    set_state();

    // A board that cannot be played (see pool_is_consistent()) is reported as
    // invalid by Engine::set_position() instead of asserting.
    assert(!is_playable() || pos_is_ok());

    return *this;
}


// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {

    update_blockers<WHITE>();
    update_blockers<BLACK>();

    Square ksq = king_square(~sideToMove);

    // We have to take special cares about the cannon and checks
    st->needFullCheck =
      checkers() || (attacks_bb(ROOK, king_square(sideToMove)) & pieces(~sideToMove, CANNON));

    st->checkSquares[PAWN]    = attacks_bb(PAWN_TO, ksq, sideToMove);
    st->checkSquares[KNIGHT]  = attacks_bb(KNIGHT_TO, ksq, pieces());
    st->checkSquares[CANNON]  = attacks_bb(CANNON, ksq, pieces());
    st->checkSquares[ROOK]    = attacks_bb(ROOK, ksq, pieces());
    st->checkSquares[ADVISOR] = attacks_bb(ADVISOR, ksq);
    st->checkSquares[BISHOP]  = attacks_bb(BISHOP, ksq, pieces());
    st->checkSquares[KING]    = 0;

    Bitboard hollowCannons = st->checkSquares[ROOK] & pieces(sideToMove, CANNON);
    if (hollowCannons)
    {
        Bitboard hollowCannonDiscover = Bitboard(0);
        while (hollowCannons)
            hollowCannonDiscover |= between_bb(ksq, pop_lsb(hollowCannons));
        for (PieceType pt = ROOK; pt < KING; ++pt)
            st->checkSquares[pt] |= hollowCannonDiscover;
    }
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key               = 0;
    st->minorPieceKey     = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->majorMaterial[WHITE] = st->majorMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = checkers_to(~sideToMove, king_square(sideToMove));
    st->move       = Move::none();

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square    s  = pop_lsb(b);
        Piece     pc = piece_on(s);
        PieceType pt = type_of(pc);

        st->key ^= Zobrist::psq[is_dark(s) ? DARK_PIECE : pc][s];

        // A dark piece is of unknown kind: it is not part of the pawn or minor
        // piece structure, only of the non-pawn pieces of its side, with its
        // own key. (Its material counts at the value of the piece that starts
        // on its square, see also jieqi_major_material().)
        if (is_dark(s))
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[DARK_PIECE][s];
            if (pt & 1)
                st->majorMaterial[color_of(pc)] += PieceValue[pc];
        }

        else if (pt == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];

        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

            if (pt != KING && (pt & 1))
            {
                st->majorMaterial[color_of(pc)] += PieceValue[pc];
                if (pt != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }

    for (const auto& color : {WHITE, BLACK})
        for (const auto& [piece, count] : rest_pieces(color))
            for (int i = 0; i < count; ++i)
                st->key ^= Zobrist::rest[piece][i];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;
}


// Returns a FEN representation of the position.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_9;; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
        {
            for (emptyCnt = 0; f <= FILE_I && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_I)
            {
                Piece pc = piece_on(make_square(f, r));
                ss << (is_dark(make_square(f, r)) ? "Xx"[color_of(pc)] : PieceToChar[pc]);
            }
        }

        if (r == RANK_0)
            break;
        ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    for (Color c : {WHITE, BLACK})
        for (PieceType pt = ROOK; pt < KING; ++pt)
        {
            Piece pc = make_piece(c, pt);
            ss << PieceToChar[pc] << restPieces[pc];
        }

    ss << " " << st->rule40 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}


// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
template<Color c>
void Position::update_blockers() const {

    Square ksq             = king_square(c);
    st->blockersForKing[c] = 0;
    st->pinners[~c]        = 0;

    // Snipers are pieces that attack 's' when a piece and other pieces are removed
    Bitboard snipers =
      ((attacks_bb(ROOK, ksq) & (pieces(ROOK) | pieces(CANNON) | pieces(KING)))
       | (attacks_bb(KNIGHT, ksq) & pieces(KNIGHT)) | (attacks_bb(BISHOP, ksq) & pieces(BISHOP)))
      & pieces(~c);
    Bitboard occupancy = pieces() ^ (snipers & ~pieces(CANNON));

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        bool     isCannon = type_of(piece_on(sniperSq)) == CANNON;
        Bitboard b = between_bb(ksq, sniperSq) & (isCannon ? pieces() ^ sniperSq : occupancy);

        if (b && ((!isCannon && !more_than_one(b)) || (isCannon && popcount(b) == 2)))
        {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

    // A dark advisor must stay in the palace: it attacks no square outside
    const Bitboard advisors = Palace & s ? pieces(ADVISOR) : pieces(ADVISOR) & ~pieces(DARK);

    return (attacks_bb(PAWN_TO, s, WHITE) & pieces(WHITE, PAWN))
         | (attacks_bb(PAWN_TO, s, BLACK) & pieces(BLACK, PAWN))
         | (attacks_bb(KNIGHT_TO, s, occupied) & pieces(KNIGHT))
         | (attacks_bb(ROOK, s, occupied) & pieces(ROOK))
         | (attacks_bb(CANNON, s, occupied) & pieces(CANNON))
         | (attacks_bb(BISHOP, s, occupied) & pieces(BISHOP))
         | (attacks_bb(ADVISOR, s) & advisors) | (attacks_bb(KING, s) & pieces(KING));
}


// Computes a bitboard of all pieces of a given color
// which gives check to a given square. Slider attacks use the occupied bitboard
// to indicate occupancy. The square is one of a king, in the palace: dark
// advisors need no special care (see attackers_to()).
Bitboard Position::checkers_to(Color c, Square s, Bitboard occupied) const {

    return ((attacks_bb(PAWN_TO, s, c) & pieces(PAWN))
            | (attacks_bb(KNIGHT_TO, s, occupied) & pieces(KNIGHT))
            | (attacks_bb(ROOK, s, occupied) & pieces(KING, ROOK))
            | (attacks_bb(CANNON, s, occupied) & pieces(CANNON))
            | (attacks_bb(BISHOP, s, occupied) & pieces(BISHOP))
            | (attacks_bb(ADVISOR, s, occupied) & pieces(ADVISOR)))
         & pieces(c);
}


// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {

    assert(m.is_ok());

    Color    us       = sideToMove;
    Square   from     = m.from_sq();
    Square   to       = m.to_sq();
    Bitboard occupied = (pieces() ^ from) | to;

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(king_square(us)) == make_piece(us, KING));
    assert(check_info_valid());

    // If the moving piece is a dark advisor, make sure the destination square
    // is within the palace
    if (type_of(piece_on(from)) == ADVISOR && move_dark(m) && !(Palace & to))
        return false;

    // If the moving piece is a king, check whether the destination square is
    // attacked by the opponent.
    if (type_of(piece_on(from)) == KING)
        return !(checkers_to(~us, to, occupied));

    // If we don't need slow check. A non-king move is always legal when either:
    // 1. Not moving a pinned piece.
    // 2. Moving a pinned non-cannon piece and aligned with king.
    // 3. Moving a pinned cannon and aligned with king but it's not a capture move.
    if (!st->needFullCheck
        && (!(blockers_for_king(us) & from)
            || (((type_of(piece_on(from)) != CANNON) || !capture(m))
                && aligned(from, to, king_square(us)))))
        return true;

    // A non-king move is legal if the king is not under attack after the move.
    return !(checkers_to(~us, king_square(us), occupied) & ~square_bb(to));
}


// Takes a random move and tests whether the move is
// pseudo-legal. It is used to validate moves from TT that can be corrupted
// due to SMP concurrent access or hash position key aliasing.
bool Position::pseudo_legal(const Move m) const {

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = moved_piece(m);

    // If the 'from' square is not occupied by a piece belonging to the side to
    // move, the move is obviously not legal.
    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    // The destination square cannot be occupied by a friendly piece
    if (pieces(us) & to)
        return false;

    // Handle the special cases
    if (type_of(pc) == PAWN)
    {
        if (!(attacks_bb(PAWN, from, us) & to))
            return false;
    }
    else if (type_of(pc) == CANNON && !capture(m))
    {
        if (!(attacks_bb(ROOK, from, pieces()) & to))
            return false;
    }
    else if (!(attacks_bb(type_of(pc), from, pieces()) & to))
        return false;

    // In check, only the evasions are pseudo-legal
    if (checkers())
        return MoveList<EVASIONS>(*this).contains(m);

    return true;
}


// Tests whether a pseudo-legal move gives a check
bool Position::gives_check(Move m) const {

    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to   = m.to_sq();
    Square ksq  = king_square(~sideToMove);

    PieceType pt = type_of(moved_piece(m));

    // A dark piece moves like the piece of its starting square, but gives a
    // direct check as the piece it turns out to be: the move may give check
    // if any kind of piece left in the mover's pool would check from 'to'.
    if (move_dark(m))
    {
        const Bitboard occupied = (pieces() ^ from) | to;
        auto           inPool   = [&](PieceType t) { return restPieces[make_piece(sideToMove, t)] > 0; };

        if ((inPool(ROOK) && (attacks_bb(ROOK, ksq, occupied) & to))
            || (inPool(CANNON) && (attacks_bb(CANNON, ksq, occupied) & to))
            || (inPool(KNIGHT) && (attacks_bb(KNIGHT_TO, ksq, occupied) & to))
            || (inPool(PAWN) && (attacks_bb(PAWN_TO, ksq, sideToMove) & to))
            || (inPool(ADVISOR) && (attacks_bb(ADVISOR, ksq) & to))
            || (inPool(BISHOP) && (attacks_bb(BISHOP, ksq, occupied) & to)))
            return true;

        // Discovered checks, including the dark piece becoming the screen of
        // a cannon, do not depend on the revealed piece
        if (attacks_bb(ROOK, ksq) & pieces(sideToMove, CANNON))
            return bool(checkers_to(sideToMove, ksq, occupied) & ~square_bb(from));

        return (blockers_for_king(~sideToMove) & from) && !aligned(from, to, ksq);
    }

    // Is there a direct check?
    if (pt == CANNON && (check_squares(ROOK) & from) && aligned(from, to, ksq))
    {
        if (capture(m) && (ray_pass_bb(ksq, from) & to))
            return true;
    }
    else if (check_squares(pt) & to)
        return true;

    // Is there a discovered check?
    if ((blockers_for_king(~sideToMove) & from) && (!aligned(from, to, ksq) || capture(m)))
        return true;

    return false;
}


Piece Position::do_flip(Square s, Piece pc, DirtyPiece* dp) {

    assert(is_ok(s) && is_dark(s));
    assert(!check_info_valid());  // Called right after the move of the dark piece

    Color us     = ~sideToMove;
    Color them   = ~us;
    Piece fromPc = piece_on(s);

    if (dp)
    {
        dp->add_pc = pc;
        dp->add_sq = s;
    }

#ifdef JIEQI_NNUE_ATT
    // Revealing a piece can change the attack bucket and the mid mirror
    const std::pair<int, bool> bucketBefore[COLOR_NB] = {
      Eval::NNUE::FeatureSet::make_feature_bucket(*this, WHITE),
      Eval::NNUE::FeatureSet::make_feature_bucket(*this, BLACK)};
#endif

    assert(color_of(pc) == us);

    // Flip the piece
    remove_piece(s);
    put_piece(pc, s);
    byTypeBB[DARK] ^= s;
    restPieces[pc]--;

    // Recalculate checkersBB, then the rest of the check info (below): from
    // now on it is valid again
    st->checkersBB = checkers_to(us, king_square(them));
    set_reveal_pending(false);

    // Update hash key
    st->key ^= Zobrist::psq[pc][s] ^ Zobrist::rest[pc][restPieces[pc]];
    // If the moving piece is a pawn, update pawn hash key.
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][s];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][s];

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][s];

        if (type_of(pc) != KING && (type_of(pc) & 1))
            st->majorMaterial[us] += PieceValue[pc];
    }

    // Update king attacks used for fast check detection
    set_check_info();

#ifdef JIEQI_NNUE_ATT
    if (dp)
        for (Color c : {WHITE, BLACK})
            dp->requires_refresh[c] |=
              Eval::NNUE::FeatureSet::make_feature_bucket(*this, c) != bucketBefore[c];
#endif

    assert(pos_is_ok());

    return fromPc;
}


void Position::undo_flip(Square s, Piece fromPc) {

    assert(is_ok(s) && !is_dark(s));

    Color us = ~sideToMove;
    Piece pc = piece_on(s);

    // Update hash key. This must mirror do_flip(), which toggled the pool slot
    // of index restPieces[pc] *after* decrementing it, so toggle it *before*
    // incrementing it back.
    st->key ^= Zobrist::psq[pc][s] ^ Zobrist::rest[pc][restPieces[pc]];

    restPieces[pc]++;
    remove_piece(s);
    put_piece(fromPc, s);
    byTypeBB[DARK] ^= s;
    // If the moving piece is a pawn, update pawn hash key.
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][s];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][s];

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][s];

        if (type_of(pc) != KING && (type_of(pc) & 1))
            st->majorMaterial[us] -= PieceValue[pc];
    }

    // Back to the state left by do_move() for a move of a dark piece: no
    // checkers known, and the check info, computed for the revealed piece,
    // is not valid any more. The caller must call do_flip() (next outcome)
    // or undo_move() before anything reads it (asserted in debug builds).
    st->checkersBB = Bitboard(0);
    set_reveal_pending(true);

    assert(pos_is_ok());
}


// Removes one piece from the pool of unrevealed pieces. Keeps the hash key
// consistent with set_state().
void Position::remove_rest_piece(Piece pc) {

    assert(pc != NO_PIECE && type_of(pc) != KING && restPieces[pc] > 0);

    restPieces[pc]--;
    st->key ^= Zobrist::rest[pc][restPieces[pc]];
}

void Position::restore_rest_piece(Piece pc) {

    assert(pc != NO_PIECE && type_of(pc) != KING);

    // Mirror of remove_rest_piece(): toggle the same pool slot, then increment.
    st->key ^= Zobrist::rest[pc][restPieces[pc]];
    restPieces[pc]++;
}


// Makes a move, and saves all information necessary
// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
// moves should be filtered out before this function is called.
// The search prefetches the TT entry of the new position itself (the TT key
// depends on the observer, see Search::observer_tt_key()).
DirtyPiece Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {

    assert(m.is_ok());
    assert(&newSt != st);
    assert(check_info_valid());  // A pending reveal must be done first

    // Update the bloom filter
    ++filter[st->key];

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;
    st->move       = m;

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = piece_on(to);
    bool   moveDark = move_dark(m);

    DirtyPiece dp{};  // add_pc/remove_pc are only set when used: no indeterminate copies
    dp.pc     = moveDark ? DARK_PIECE : pc;
    dp.from   = from;
    dp.to     = moveDark ? SQ_NONE : to;
    dp.add_sq = SQ_NONE;
    dp.rest_remove_pc = NO_PIECE;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == them);
    assert(type_of(captured) != KING);

#ifdef JIEQI_NNUE_ATT
    // A move can change the feature bucket (attack bucket, mid mirror) of
    // either side: compare before and after (see the end of this function)
    const std::pair<int, bool> bucketBefore[COLOR_NB] = {
      Eval::NNUE::FeatureSet::make_feature_bucket(*this, WHITE),
      Eval::NNUE::FeatureSet::make_feature_bucket(*this, BLACK)};
#endif

    // Increment ply counters.
    // In particular, rule40 will be reset to zero later on in case of a capture.
    ++gamePly;
    ++st->rule40;
    ++st->pliesFromNull;

    if (pc == make_piece(us, KING))
    {
        dp.requires_refresh[us] = true;
        bool mirror_before = Eval::NNUE::FeatureSet::KingBuckets[king_square(them)][from].second;
        bool mirror_after  = Eval::NNUE::FeatureSet::KingBuckets[king_square(them)][to].second;
        dp.requires_refresh[them] = (mirror_before != mirror_after);
    }
    else
        dp.requires_refresh[us] = dp.requires_refresh[them] = false;

    if (captured)
    {
        Square capsq = to;

        st->captureDark = is_dark(to);

        // If the captured piece is a pawn, update pawn hash key, otherwise
        // update major material. A dark piece only has a non-pawn key.
        if (st->captureDark)
        {
            st->nonPawnKey[them] ^= Zobrist::psq[DARK_PIECE][capsq];
            if (type_of(captured) & 1)
                st->majorMaterial[them] -= PieceValue[captured];
        }

        else if (type_of(captured) == PAWN)
            st->pawnKey ^= Zobrist::psq[captured][capsq];

        else
        {
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];

            if (type_of(captured) & 1)
            {
                st->majorMaterial[them] -= PieceValue[captured];
                if (type_of(captured) != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[captured][capsq];
            }
        }

        dp.remove_pc = st->captureDark ? DARK_PIECE : captured;
        dp.remove_sq = capsq;

        if (st->captureDark)
            byTypeBB[DARK] ^= to;

        // Update board and piece lists
        remove_piece(capsq);

        // Update hash key
        k ^= Zobrist::psq[st->captureDark ? DARK_PIECE : captured][capsq];

        // Reset rule 40 counter
        st->rule40 = 0;
    }
    else
    {
        st->captureDark = false;
        dp.remove_sq    = SQ_NONE;
    }

    // Update hash key
    k ^= moveDark ? Zobrist::psq[DARK_PIECE][from] : Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // Update the key with the final value (the key of a dark move is
    // completed by the reveal)
    st->key = k;

    // If the moving piece is a pawn, update pawn hash key. A dark piece
    // leaves its square: do_flip() adds the revealed piece on 'to'.
    if (moveDark)
    {
        st->nonPawnKey[us] ^= Zobrist::psq[DARK_PIECE][from];
        if (type_of(pc) & 1)
            st->majorMaterial[us] -= PieceValue[pc];
    }
    else if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }

    // Move the piece.
    move_piece(from, to);

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers bitboard (if move gives check)
    st->checkersBB = !moveDark && givesCheck ? checkers_to(us, king_square(them)) : Bitboard(0);
    assert(moveDark || givesCheck == bool(st->checkersBB));

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection. After a move of a
    // dark piece, the check info is only computed by do_flip().
    set_reveal_pending(moveDark);
    if (!moveDark)
        set_check_info();
    else
        byTypeBB[DARK] ^= from | to;

#ifdef JIEQI_NNUE_ATT
    for (Color c : {WHITE, BLACK})
        dp.requires_refresh[c] |=
          Eval::NNUE::FeatureSet::make_feature_bucket(*this, c) != bucketBefore[c];
#endif

    assert(pos_is_ok());

    assert(dp.pc != NO_PIECE);
    assert(!bool(captured) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE && (moveDark || dp.to != SQ_NONE));
    assert(dp.add_sq == SQ_NONE);
    return dp;
}


// Key of the position after a move of a revealed piece, as used by the TT,
// to prefetch its entry before the move is made. Approximate in rare cases
// (repetition filter); the prefetch then lands on an unused line.
Key Position::prefetch_key(Move m) const {
    assert(!move_dark(m));

    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = piece_on(to);
    Key    k        = st->key ^ Zobrist::side ^ Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    if (captured)
        return k ^ Zobrist::psq[is_dark(to) ? DARK_PIECE : captured][to];

    return adjust_key40<true>(k);
}


// Key of the position after do_flip(s, pc), as returned by key(), to
// prefetch its TT entry before the reveal is made. Exact: a reveal changes
// neither the rule 40 counter nor the repetition filter.
Key Position::flip_key(Square s, Piece pc) const {
    assert(is_dark(s) && color_of(pc) == ~sideToMove && restPieces[pc] > 0);

    return adjust_key40(st->key ^ Zobrist::psq[pc][s] ^ Zobrist::rest[pc][restPieces[pc] - 1]);
}


// Unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.
void Position::undo_move(Move m) {

    assert(m.is_ok());

    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to   = m.to_sq();

    assert(empty(from));
    assert(type_of(st->capturedPiece) != KING);

    if (is_dark(to))
        byTypeBB[DARK] ^= from | to;  // Update dark pieces bitboard

    move_piece(to, from);  // Put the piece back at the source square

    if (st->capturedPiece)
    {
        Square capsq = to;

        put_piece(st->capturedPiece, capsq);  // Restore the captured piece

        if (st->captureDark)
            byTypeBB[DARK] ^= capsq;
    }

    // Finally point our state pointer back to the previous state
    st = st->previous;
    --gamePly;

    // Update the bloom filter
    --filter[st->key];

    assert(pos_is_ok());
}


// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt) {

    assert(!checkers());
    assert(&newSt != st);

    // Update the bloom filter
    ++filter[st->key];

    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st             = &newSt;

    st->key ^= Zobrist::side;

    st->pliesFromNull = 0;

    // A null move captures nothing
    st->capturedPiece = NO_PIECE;
    st->captureDark   = false;

    sideToMove = ~sideToMove;

    set_check_info();

    assert(pos_is_ok());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;

    // Update the bloom filter
    --filter[st->key];
}


// Tests if the SEE (Static Exchange Evaluation)
// value of move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    assert(m.is_ok());

    Square from = m.from_sq(), to = m.to_sq();

    assert(piece_on(from) != NO_PIECE);

    // A dark piece counts at the average value of the pieces left in its
    // owner's pool: that is what it is worth, as a captured piece and as a
    // piece that will be revealed on 'to' and may be captured in turn.
    const Value darkValue[COLOR_NB] = {pieces(WHITE, DARK) ? dark_value(WHITE) : VALUE_ZERO,
                                       pieces(BLACK, DARK) ? dark_value(BLACK) : VALUE_ZERO};
    auto value_on = [&](Square s, Value typeValue) {
        return is_dark(s) ? darkValue[color_of(piece_on(s))] : typeValue;
    };

    int swap = value_on(to, PieceValue[piece_on(to)]) - threshold;
    if (swap < 0)
        return false;

    swap = value_on(from, PieceValue[piece_on(from)]) - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied  = pieces() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);

    // Flying general
    bool kingAttacks = attackers & pieces(KING);
    if (kingAttacks)
        attackers |= attacks_bb(ROOK, to, occupied) & pieces(KING);

    Bitboard nonCannons = attackers & ~pieces(CANNON);
    Bitboard cannons    = attackers & pieces(CANNON);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to the
        // bitboard 'attackers' any protential attackers when it is removed.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = value_on(lsb(bb), PawnValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |=
              attacks_bb(ROOK, to, occupied) & (kingAttacks ? pieces(KING, ROOK) : pieces(ROOK));
            cannons   = attacks_bb(CANNON, to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = value_on(lsb(bb), BishopValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(ADVISOR)))
        {
            if ((swap = value_on(lsb(bb), AdvisorValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |= attacks_bb(KNIGHT_TO, to, occupied) & pieces(KNIGHT);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(CANNON)))
        {
            if ((swap = value_on(lsb(bb), CannonValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            cannons   = attacks_bb(CANNON, to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = value_on(lsb(bb), KnightValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            // (A revealed rook is the most valuable attacker, a dark piece
            // on a rook square may be worth less than the previous one)
            if ((swap = value_on(lsb(bb), RookValue) - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            nonCannons |=
              attacks_bb(ROOK, to, occupied) & (kingAttacks ? pieces(KING, ROOK) : pieces(ROOK));
            cannons   = attacks_bb(CANNON, to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}


// A lighter version of do_move(), used in chasing detection
std::pair<Piece, int> Position::do_move(Move m) {

    assert(capture(m));

    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  captured = piece_on(to);
    int    id       = idBoard[to];

    // Update id board
    idBoard[to]   = idBoard[from];
    idBoard[from] = 0;

    // Update board and piece lists
    remove_piece(to);
    move_piece(from, to);

    sideToMove = ~sideToMove;

    return {captured, id};
}


// A lighter version of undo_move(), used in chasing detection
void Position::undo_move(Move m, Piece captured, int id) {

    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Put back id board
    idBoard[from] = idBoard[to];
    idBoard[to]   = id;

    move_piece(to, from);  // Put the piece back at the source square

    if (captured)
        put_piece(captured, to);  // Restore the captured piece
}


// Tests whether a pseudo-legal move is chase legal
bool Position::chase_legal(Move m) const {

    assert(m.is_ok());

    Color    us       = sideToMove;
    Square   from     = m.from_sq();
    Square   to       = m.to_sq();
    Bitboard occupied = (pieces() ^ from) | to;

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(king_square(us)) == make_piece(us, KING));

    // If the moving piece is a king, check whether the destination
    // square is not under new attack after the move.
    if (type_of(piece_on(from)) == KING)
        return !(checkers_to(~us, to, occupied));

    // A non-king move is chase legal if the king is not under new attack after the move.
    return !(checkers_to(~us, king_square(us), occupied) & ~square_bb(to));
}


// Calculates the chase information for a given color.
uint16_t Position::chased(Color c) {

    uint16_t chase = 0;

    std::swap(c, sideToMove);

    // King and pawn can legally perpetual chase
    Bitboard attackers = pieces(sideToMove) ^ pieces(sideToMove, KING, PAWN);
    while (attackers)
    {
        Square    from         = pop_lsb(attackers);
        PieceType attackerType = type_of(piece_on(from));
        Bitboard  attacks      = attacks_bb(attackerType, from, pieces());

        // A dark advisor must stay in the palace
        if (attackerType == ADVISOR && is_dark(from))
            attacks &= Palace;

        // Restrict to pinners if pinned, otherwise exclude attacks on unpromoted pawns and checks
        if (blockers_for_king(sideToMove) & from)
            attacks &= pinners(~sideToMove) & ~pieces(KING);
        else
            attacks &= (pieces(~sideToMove) ^ pieces(~sideToMove, KING, PAWN))
                     | (pieces(~sideToMove, PAWN) & HalfBB[sideToMove]);

        while (attacks)
        {
            Square to = pop_lsb(attacks);
            Move   m  = Move(from, to);

            if (chase_legal(m))
            {
                // Attacks against stronger pieces
                if ((attackerType == KNIGHT || attackerType == CANNON)
                    && type_of(piece_on(to)) == ROOK)
                    chase |= (1 << idBoard[to]);
                else if ((attackerType == ADVISOR || attackerType == BISHOP)
                         && type_of(piece_on(to)) & 1)
                    chase |= (1 << idBoard[to]);
                // Attacks against potentially unprotected pieces
                else
                {
                    bool trueChase             = true;
                    const auto& [captured, id] = do_move(m);
                    Bitboard recaptures        = attackers_to(to) & pieces(sideToMove);
                    while (recaptures)
                    {
                        Square s = pop_lsb(recaptures);
                        if (chase_legal(Move(s, to)))
                        {
                            trueChase = false;
                            break;
                        }
                    }
                    undo_move(m, captured, id);

                    if (trueChase)
                    {
                        // Exclude mutual/symmetric attacks except pins
                        if (attackerType == type_of(piece_on(to)))
                        {
                            sideToMove = ~sideToMove;
                            if ((attackerType == KNIGHT && ((between_bb(from, to) ^ to) & pieces()))
                                || !chase_legal(Move(to, from)))
                                chase |= (1 << idBoard[to]);
                            sideToMove = ~sideToMove;
                        }
                        else
                            chase |= (1 << idBoard[to]);
                    }
                }
            }
        }
    }

    std::swap(c, sideToMove);

    return chase;
}


// Detects chases from state st - d to state st
Value Position::detect_chases(int d, int ply) {

    // Grant each piece on board a unique id for each side
    int whiteId = 0;
    int blackId = 0;
    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        if (board[s] != NO_PIECE)
            idBoard[s] = color_of(board[s]) == WHITE ? whiteId++ : blackId++;

    Color us = sideToMove, them = ~us;

    // Rollback until we reached st - d
    uint16_t chase[COLOR_NB] = {0xFFFF, 0xFFFF};
    for (int i = 0; i < d; ++i)
    {
        if (st->checkersBB)
            return VALUE_DRAW;
        else if (!chase[~sideToMove])
        {
            if (!chase[sideToMove])
                break;
            undo_move(st->move, st->capturedPiece);
            st = st->previous;
        }
        else
        {
            uint16_t after = chased(~sideToMove);
            undo_move(st->move, st->capturedPiece);
            st = st->previous;
            // Take the exact diff to detect the chase
            chase[sideToMove] &= after & ~chased(sideToMove);
        }
    }

    return bool(chase[us]) ^ bool(chase[them]) ? chase[us] ? mated_in(ply) : mate_in(ply)
                                               : VALUE_DRAW;
}


// Tests whether the position may end the game by rule 60, insufficient material, draw repetition,
// perpetual check repetition or perpetual chase repetition that allows a player to claim a game result.
bool Position::rule_judge(Value& result, int ply) {

    int end = std::min(st->rule40, st->pliesFromNull);

    if (end >= 4 && filter[st->key] >= 1)
    {
        int        cnt       = 0;
        StateInfo* stp       = st->previous->previous;
        bool       checkThem = st->checkersBB && stp->checkersBB;
        bool       checkUs   = st->previous->checkersBB && stp->previous->checkersBB;

        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            checkThem &= bool(stp->checkersBB);

            // Return a score if a position repeats once earlier but strictly
            // after the root, or repeats twice before or at the root.
            if (stp->key == st->key && (++cnt == 2 || ply > i))
            {
                if (!checkThem && !checkUs)
                {
                    // Copy the current position to a rollback struct, so we don't need to do those moves again
                    Position rollback;
                    memcpy((void*) &rollback, (const void*) this, offsetof(Position, filter));

                    // Chasing detection
                    result = rollback.detect_chases(i, ply);
                }
                else
                    // Checking detection
                    result = !checkUs ? mate_in(ply) : !checkThem ? mated_in(ply) : VALUE_DRAW;

                // 3 folds and 2 fold draws can be judged immediately
                if (result == VALUE_DRAW || cnt == 2)
                    return true;

                // 2 fold mates need further investigations
                if (filter[st->key] <= 1)
                {
                    // Not exceeding rule 40 and have the same previous step
                    if (st->rule40 < 80 && st->previous->key == stp->previous->key)
                    {
                        // Even if we entering this loop again, it will not lead to a 3 fold repetition
                        StateInfo* prev = st->previous;
                        while ((prev = prev->previous) != stp)
                            if (filter[prev->key] > 1)
                                break;
                        if (prev == stp)
                            return true;
                    }
                    // We know there can't be another fold
                    break;
                }
            }

            if (i + 1 <= end)
                checkUs &= bool(stp->previous->checkersBB);
        }
    }

    // 40 move rule
    if (st->rule40 >= 80)
    {
        result = MoveList<LEGAL>(*this).size() ? VALUE_DRAW : mated_in(ply);
        return true;
    }

    return false;
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
void Position::flip() {

    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_9;; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_0 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));

        if (r == RANK_0)
            break;
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](unsigned char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;
    f += token;

    std::getline(ss, token);  // Half and full moves
    f += token;

    set(f, st);

    assert(pos_is_ok());
}


// Checks that the position is one that move generation, the search and the
// network can handle, and reports what is wrong otherwise:
//  - exactly one king per side, in its own palace;
//  - for every kind of piece, the revealed pieces on the board plus the pieces
//    of that kind in the pool do not exceed the initial number (2 rooks, 5
//    pawns, ...). Every piece is either revealed and on the board, captured, or
//    still in the pool, so this holds in any real game. It also bounds the
//    number of pieces of a side by 16, which the fixed-size tables rely on
//    (NNUE feature lists, layer stack buckets, move lists);
//  - every dark piece has at least one candidate in its owner's pool;
//  - the side that has just moved has not left its king in check (the side to
//    move could capture it, which do_move() does not support).
// Positions that fail this check are not searched (see Engine::set_position()).
bool Position::pool_is_consistent(std::string* reason) const {

    constexpr int MaxCount[PIECE_TYPE_NB] = {0, 2, 2, 2, 5, 2, 2, 0};

    auto fail = [&](std::string why) {
        if (reason)
            *reason = std::move(why);
        return false;
    };
    auto side = [](Color c) { return std::string(c == WHITE ? "white" : "black"); };

    for (Color c : {WHITE, BLACK})
    {
        if (count<KING>(c) != 1)
            return fail(side(c) + " does not have exactly one king");

        if (!(Palace & HalfBB[c] & king_square(c)))
            return fail(side(c) + " king is outside its palace");
    }

    for (Color c : {WHITE, BLACK})
    {
        int pool = 0;
        for (PieceType pt = ROOK; pt < KING; ++pt)
        {
            const Piece pc       = make_piece(c, pt);
            const int   n        = restPieces[pc];
            const int   revealed = popcount(pieces(c, pt) & ~pieces(DARK));

            if (n < 0 || n > MaxCount[pt])
                return fail(std::string("too many pieces of kind '") + PieceToChar[pc]
                            + "' in the pool");

            if (revealed + n > MaxCount[pt])
                return fail(std::string("too many pieces of kind '") + PieceToChar[pc]
                            + "' on the board and in the pool");

            pool += n;
        }

        if (pool < popcount(pieces(c) & pieces(DARK)))
            return fail(side(c) + " has more dark pieces on the board than pieces in its pool");
    }

    if (checkers_to(sideToMove, king_square(~sideToMove)))
        return fail(side(~sideToMove) + " (not to move) is in check");

    return true;
}


// Whether move generation and search are defined (see pool_is_consistent())
bool Position::is_playable() const { return pool_is_consistent(); }


// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {

    constexpr bool Fast = true;  // Quick (default) or full check?

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(king_square(WHITE)) != W_KING
        || piece_on(king_square(BLACK)) != B_KING)
        assert(0 && "pos_is_ok: Default");

    if (Fast)
        return true;

    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1
        || checkers_to(sideToMove, king_square(~sideToMove)))
        assert(0 && "pos_is_ok: Kings");

    if ((pieces(WHITE, PAWN) & ~PawnBB[WHITE]) || (pieces(BLACK, PAWN) & ~PawnBB[BLACK])
        || pieceCount[W_PAWN] > 5 || pieceCount[B_PAWN] > 5)
        assert(0 && "pos_is_ok: Pawns");

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            assert(0 && "pos_is_ok: Pieces");

    return true;
}

}  // namespace Maskfish
