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

#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "history.h"
#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "numa.h"
#include "position.h"
#include "reveal.h"
#include "score.h"
#include "timeman.h"
#include "types.h"

namespace Maskfish {

// Different node types, used as a template parameter
enum NodeType {
    NonPV,
    PV,
    Root
};

class TranspositionTable;
class ThreadPool;
class OptionsMap;

namespace Search {

// Stack struct keeps track of the information we need to remember from nodes
// shallower and deeper in the tree during the search. Each search thread has
// its own array of Stack objects, indexed by the current ply.
struct Stack {
    Move*                       pv;
    PieceToHistory*             continuationHistory;
    CorrectionHistory<PieceTo>* continuationCorrectionHistory;
    int                         ply;
    Move                        currentMove;
    // Piece of currentMove as the histories index it: the piece that moved,
    // for a dark piece the kind of its starting square (what the move picker
    // sees before the reveal), not the piece revealed on the target square
    Piece                       movedPiece;
    Move                        excludedMove;
    Value                       staticEval;
    int                         statScore;
    int                         moveCount;
    bool                        inCheck;
    bool                        ttPv;
    bool                        ttHit;
    int                         cutoffCnt;
    int                         priorNMPFailHigh;  // null move fail highs of previous siblings
    int                         reduction;
    bool                        isPvNode;
    int                         quietMoveStreak;
    bool                        capturedDark;  // currentMove captured a dark piece
    bool                        followPV;      // on the PV of the previous iteration
    // The value last returned by search()/qsearch() at this ply derives from
    // the estimate of a chance node (see Worker::chance_node()). If it lies
    // within the caller's window, it is not exact and must not be stored as
    // BOUND_EXACT.
    bool estimated;
};


// A chance node (flip_search) searches the same child ply once per possible
// identity of the revealed piece. search() reads (ss - 1)->reduction as its
// "prior reduction" and then clears it, so without this guard only the first
// outcome would see the parent's LMR reduction and the others would see 0.
// restore() puts the parent's value back before every outcome; the destructor
// leaves the field cleared, as a single call to search() would.
// (Idea from AB-JChess.)
class ChanceBranchReduction {
   public:
    explicit ChanceBranchReduction(int& r) :
        reduction(r),
        parentReduction(r) {}

    ~ChanceBranchReduction() { reduction = 0; }

    void restore() const { reduction = parentReduction; }

    ChanceBranchReduction(const ChanceBranchReduction&)            = delete;
    ChanceBranchReduction& operator=(const ChanceBranchReduction&) = delete;

   private:
    int&      reduction;
    const int parentReduction;
};


// RootMove struct is used for moves at the root of the tree. For each root move
// we store a score and a PV (really a refutation in the case of moves which
// fail low). Score is normally set at -VALUE_INFINITE for all non-pv moves.
struct RootMove {

    explicit RootMove(Move m) :
        pv(1, m) {}
    bool extract_ponder_from_tt(const TranspositionTable& tt, Position& pos, Color observer);
    bool score_is_bound() const { return scoreLowerbound || scoreUpperbound; }
    bool score_is_exact_loss() const {
        return score != -VALUE_INFINITE && is_loss(score) && !score_is_bound();
    }
    void unset_bound_flags() { scoreLowerbound = scoreUpperbound = false; }
    bool operator==(const Move& m) const { return pv[0] == m; }
    // Sort in descending order
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score : m.previousScore < previousScore;
    }

    uint64_t          effort           = 0;
    Value             score            = -VALUE_INFINITE;
    Value             previousScore    = -VALUE_INFINITE;
    Value             averageScore     = -VALUE_INFINITE;
    Value             meanSquaredScore = -VALUE_INFINITE * VALUE_INFINITE;
    Value             uciScore         = -VALUE_INFINITE;
    bool              scoreLowerbound  = false;
    bool              scoreUpperbound  = false;
    // previousScore / previousPV come from a complete search of the move in the
    // previous iteration (it was one of the MultiPV lines)
    bool              previousScoreExact = false;
    int               selDepth           = 0;
    // Safe reveal bonus of the move itself (see Reveal::move_bonus()): part of
    // score, which the search compares, but not of the score sent to the GUI
    int               revealBonus        = 0;
    std::vector<Move> pv, previousPV;
};

using RootMoves = std::vector<RootMove>;


// LimitsType struct stores information sent by the caller about the analysis required.
struct LimitsType {

    // Init explicitly due to broken value-initialization of non POD in MSVC
    LimitsType() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
        movestogo = depth = mate = perft = infinite = 0;
        nodes                                       = 0;
        ponderMode                                  = false;
    }

    bool use_time_management() const { return time[WHITE] || time[BLACK]; }

    std::vector<std::string> searchmoves;
    TimePoint                time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
    int                      movestogo, depth, mate, perft, infinite;
    uint64_t                 nodes;
    bool                     ponderMode;
};


// The UCI stores the uci options, thread pool, and transposition table.
// This struct is used to easily forward data to the Search::Worker class.
struct SharedState {
    SharedState(const OptionsMap&                                         optionsMap,
                ThreadPool&                                               threadPool,
                TranspositionTable&                                       transpositionTable,
                const LazyNumaReplicatedSystemWide<Eval::NNUE::Networks>& nets) :
        options(optionsMap),
        threads(threadPool),
        tt(transpositionTable),
        networks(nets) {}

    const OptionsMap&                                         options;
    ThreadPool&                                               threads;
    TranspositionTable&                                       tt;
    const LazyNumaReplicatedSystemWide<Eval::NNUE::Networks>& networks;
};

class Worker;

// TT key for a position searched for the given observer (the side whose
// knowledge of the pool the search models, see HiddenCaptureModel). All 64
// bits change, so that the entries of the two observers fall in different
// clusters instead of competing for the slots of the same one. The search
// prefetches TT entries with this key (Position does not know the observer).
inline Key observer_tt_key(Key positionKey, Color observer) {
    return observer == WHITE ? positionKey : positionKey ^ Key(0xA3B195354A39B70DULL);
}

// Null Object Pattern, implement a common interface for the SearchManagers.
// A Null Object will be given to non-mainthread workers.
class ISearchManager {
   public:
    virtual ~ISearchManager() {}
    virtual void check_time(Search::Worker&) = 0;
};

struct InfoShort {
    int   depth;
    Score score;
};

struct InfoFull: InfoShort {
    int              selDepth;
    size_t           multiPV;
    std::string_view wdl;
    std::string_view bound;
    size_t           timeMs;
    size_t           nodes;
    size_t           nps;
    size_t           tbHits;
    std::string_view pv;
    int              hashfull;
};

struct InfoIteration {
    int              depth;
    std::string_view currmove;
    size_t           currmovenumber;
};

// SearchManager manages the search from the main thread. It is responsible for
// keeping track of the time, and storing data strictly related to the main thread.
class SearchManager: public ISearchManager {
   public:
    using UpdateShort    = std::function<void(const InfoShort&)>;
    using UpdateFull     = std::function<void(const InfoFull&)>;
    using UpdateIter     = std::function<void(const InfoIteration&)>;
    using UpdateBestmove = std::function<void(std::string_view, std::string_view)>;
    using UpdateStart    = std::function<void()>;

    struct UpdateContext {
        UpdateShort    onUpdateNoMoves;
        UpdateFull     onUpdateFull;
        UpdateIter     onIter;
        UpdateBestmove onBestmove;
        UpdateStart    onStart;
    };


    SearchManager(const UpdateContext& updateContext) :
        updates(updateContext) {}

    void check_time(Search::Worker& worker) override;

    void pv(const Search::Worker&     worker,
            const ThreadPool&         threads,
            const TranspositionTable& tt,
            Depth                     depth) const;

    Maskfish::TimeManagement  tm;
    double                    originalTimeAdjust;
    int                       callsCnt;
    std::atomic_bool          ponder;

    std::array<Value, 4> iterValue;
    double               previousTimeReduction;
    Value                bestPreviousScore;
    Value                bestPreviousAverageScore;
    bool                 stopOnPonderhit;

    size_t id;

    const UpdateContext& updates;
};

class NullSearchManager: public ISearchManager {
   public:
    void check_time(Search::Worker&) override {}
};


// Search::Worker is the class that does the actual search.
// It is instantiated once per thread, and it is responsible for keeping track
// of the search history, and storing data required for the search.
// The histories that can be shared between the threads of a NUMA node
// (UCI option SharedHistory). Without sharing, each Worker uses its own copy.
struct HistoryTables {
    ContinuationHistory continuationHistory[2][2];
    PawnHistory         pawnHistory;

    CorrectionHistory<Pawn>         pawnCorrectionHistory;
    CorrectionHistory<Minor>        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation> continuationCorrectionHistory;
};

class Worker {
   public:
    Worker(SharedState&, std::unique_ptr<ISearchManager>, size_t, NumaReplicatedAccessToken);

    // Called at instantiation to initialize reductions tables.
    // Reset histories, usually before a new game.
    void clear();

    // Called when the program receives the UCI 'go' command.
    // It searches from the root position and outputs the "bestmove".
    void start_searching();

    bool is_mainthread() const { return threadIdx == 0; }

    void ensure_network_replicated();

    // Public because they need to be updatable by the stats
    ButterflyHistory mainHistory;
    LowPlyHistory    lowPlyHistory;

    CapturePieceToHistory captureHistory;

    // Shared tables of this thread's NUMA node (nullptr when not shared), and
    // whether this Worker is the one that clears them.
    bool           ownsSharedHistories = false;  // declared first: set by the claim below
    HistoryTables* sharedHistories     = nullptr;
    HistoryTables  ownHistories;

    // The tables actually used: references into sharedHistories or ownHistories
    ContinuationHistory (&continuationHistory)[2][2];
    PawnHistory& pawnHistory;

    CorrectionHistory<Pawn>&         pawnCorrectionHistory;
    CorrectionHistory<Minor>&        minorPieceCorrectionHistory;
    CorrectionHistory<NonPawn>&      nonPawnCorrectionHistory;
    CorrectionHistory<Continuation>& continuationCorrectionHistory;

    TTMoveHistory ttMoveHistory;

    // Experimental search options ported from Pikafish (see engine.cpp), off by
    // default. Public because the free history update functions read them.
    bool contCorr4Ply    = false;
    bool multiCutHistory = false;
    bool cmhc            = false;
    bool malusDecay      = false;
    bool bestMoveBonusScale = false;
    bool contHistOffset = false;
    bool contCorr6Ply = false;
    bool bigPawnHistory = false;

    // Index of the pawn history for the position (UCI option BigPawnHistory)
    int pawn_history_index(const Position& pos) const {
        return pawn_structure_index(pos, bigPawnHistory);
    }

   private:
    // Returns whether the last PV sent to the GUI is up to date
    bool iterative_deepening();

    void reset_correction_histories();

    void do_move(Position& pos, const Move move, StateInfo& st);
    void prefetch_correction_histories(const Stack* ss, const Position& pos, Move move) const;
    void do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck);
    void do_null_move(Position& pos, StateInfo& st);
    void undo_move(Position& pos, const Move move);
    void undo_null_move(Position& pos);

    // This is the main search function, for both PV and non-PV nodes
    template<NodeType nodeType>
    Value search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    // Quiescence search function, which is called by the main search
    template<NodeType nodeType>
    Value qsearch(Position& pos, Stack* ss, Value alpha, Value beta);

    // Searches a child and returns its value from the parent's point of view,
    // adding the safe reveal bonus of the move that led to it
    template<NodeType nodeType>
    Value search_child(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth,
                       bool cutNode, int revealBonus);

    // Chance node over the identity of a dark piece just captured by the root
    // side (see Worker::hiddenCaptureModel)
    template<NodeType nodeType>
    Value hidden_capture_search(Position& pos,
                                Stack*    ss,
                                Value     alpha,
                                Value     beta,
                                bool      isQsearch,
                                Depth     depth,
                                bool      cutNode);

    // Whether move m, made in pos, is a capture of a dark piece by the
    // observer whose identity is uncertain (its owner's pool holds more than
    // one kind of piece): with hiddenCaptureModel the search goes on with
    // every possible identity, so the moves after m in a PV come from one of
    // them only. Such a PV is cut after m, and gives no ponder move.
    bool uncertain_hidden_capture(const Position& pos, Move m) const;

    // This is the flip search function for any nodes
    template<NodeType nodeType>
    Value flip_search(Position&              pos,
                      Stack*                 ss,
                      Value                  alpha,
                      Value                  beta,
                      bool                   isQsearch = true,
                      [[maybe_unused]] Depth depth     = -1,
                      [[maybe_unused]] bool  cutNode   = false);

    Depth reduction(bool i, Depth d, int mn, int delta) const;

    // Pointer to the search manager, only allowed to be called by the main thread
    SearchManager* main_manager() const {
        assert(threadIdx == 0);
        return static_cast<SearchManager*>(manager.get());
    }

    TimePoint elapsed() const;
    TimePoint elapsed_time() const;

    Value evaluate(const Position&);

    LimitsType limits;

    size_t                pvIdx, pvLast;
    std::atomic<uint64_t> nodes, bestMoveChanges;
    int                   selDepth, nmpMinPly;

    Value optimism[COLOR_NB];

    Position  rootPos;
    StateInfo rootState;
    // False for a position that is not searched (invalid, see
    // ThreadPool::start_thinking()): it has no root moves, but is not a mate
    bool rootSearchable = true;

    // Jieqi information model. The side the engine is searching for (the
    // side to move at the root; also while pondering, since the root is then
    // the position after the expected reply, see iterative_deepening()) learns
    // the identity of the dark pieces it captures, but not of its own dark pieces
    // captured by the opponent. With hiddenCaptureModel, a capture of a dark
    // piece by the observer is a chance node over the captured identity.
    // TT keys are salted with the observer because the same position is then
    // searched differently depending on who is observing. (From AB-JChess.)
    Color                   rootObserver = WHITE;
    bool                    hiddenCaptureModel = false;

    // Experimental search options (see engine.cpp), all off by default
    bool   probCutDarkCapture = false;
    bool   ttPenalize         = false;
    bool   chanceBranchDepth  = false;
    bool   seekMateOption     = false;
    bool   pruningTweaks      = false;
    bool   lmrTweaks          = false;
    bool   failHighRecoveryOption = false;
    bool   aspirationTweaks = false;
    bool   mainHistoryDecay = false;
    bool   lmrHistoryDivisor = false;
    bool   corrHistAsym = false;
    bool   hindsightFirst = false;
    bool   followPVOption = false;
    // PV of the previous iteration for the current PV line (UCI option FollowPV)
    std::vector<Move> followPVLine;
    bool   cutoffCntTweak = false;
    bool   nmpFailHighCount = false;
    bool   mateTTExtension = false;
    bool   stopEarly = false;
    bool   frequentTimeCheck = false;
    bool   revealQsearchEvasions = false;
    bool   rootScoreEMA = false;
    bool   standPatNoTT = false;
    bool   jieqiMaterial      = false;
    double chanceScaling      = 360.83524;
    double chanceRisk         = 0.0;

    // Aggregates the outcomes of a chance node (flip_search, hidden_capture_search),
    // from their exact values or from bounds of them
    enum class ChanceBound {
        Exact,
        Lower,
        Upper
    };
    Value aggregate_chance(
      const Value* values, const int* counts, int n, int total, ChanceBound kind) const;

    // Searches the outcomes of a chance node, see search.cpp
    template<typename SearchOutcome>
    Value chance_node(Stack*          ss,
                      const int*      counts,
                      int             n,
                      int             total,
                      Value           alpha,
                      Value           beta,
                      SearchOutcome&& searchOutcome);
    Reveal::QuietParameters revealQuiet;
    RootMoves rootMoves;
    Depth     rootDepth, completedDepth;
    Value     rootDelta;
    // At least one root move has been fully searched since the search started,
    // so that rootMoves[0] holds a searched move: a hard time or node limit may
    // then stop the search before the first iteration is completed (in Jieqi,
    // depth 1 of an opening position can take hundreds of milliseconds)
    bool rootMoveSearched = false;

    size_t                    threadIdx;
    NumaReplicatedAccessToken numaAccessToken;

    // Reductions lookup table initialized at startup
    std::array<int, std::max(MAX_PLY, MAX_MOVES) + 10> reductions;  // [depth or moveNumber]

    // The main thread has a SearchManager, the others have a NullSearchManager
    std::unique_ptr<ISearchManager> manager;

    const OptionsMap&                                         options;
    ThreadPool&                                               threads;
    TranspositionTable&                                       tt;
    const LazyNumaReplicatedSystemWide<Eval::NNUE::Networks>& networks;

    // Used by NNUE
    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;

    friend class Maskfish::ThreadPool;
    friend class SearchManager;
};

struct ConthistBonus {
    int index;
    int weight;
};


}  // namespace Search

}  // namespace Maskfish

#endif  // #ifndef SEARCH_H_INCLUDED
