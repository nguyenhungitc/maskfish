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

#include "jieqi/c_api.h"
#include "jieqi/rules.hpp"
#include "jieqi/repetition.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <exception>
#include <stdexcept>
#include <string_view>

struct jq_game {
    jieqi::RepetitionHistory history;
    std::uint32_t limit;
    jq_game(jieqi::Position initial, std::uint32_t cap) : history(std::move(initial)), limit(cap) {}
};

namespace {
void message(char* buffer, std::size_t capacity, std::string_view text) {
    if (buffer == nullptr || capacity == 0) return;
    const auto count = std::min(text.size(), capacity - 1);
    std::copy_n(text.begin(), count, buffer);
    buffer[count] = '\0';
}
template<typename Function>
int guarded(char* error, std::size_t capacity, Function operation) noexcept {
    message(error, capacity, "");
    try { return operation(); }
    catch (const std::invalid_argument& failure) {
        message(error, capacity, failure.what()); return JQ_INVALID;
    } catch (const std::exception& failure) {
        message(error, capacity, failure.what()); return JQ_INTERNAL;
    } catch (...) {
        message(error, capacity, "unknown native failure"); return JQ_INTERNAL;
    }
}
jieqi::Color color(int side) {
    if (side == 0) return jieqi::Color::Red;
    if (side == 1) return jieqi::Color::Black;
    throw std::invalid_argument("side must be 0 or 1");
}
jieqi::Position position(const char* board, std::size_t length, int side, std::uint32_t idle = 0) {
    if (board == nullptr) throw std::invalid_argument("null board");
    return jieqi::Position(std::string_view(board, length), color(side), idle);
}
jieqi::Kind reveal_kind(char value) {
    switch (value) {
    case '\0': return jieqi::Kind::Empty;
    case 'R': return jieqi::Kind::Rook;
    case 'N': return jieqi::Kind::Horse;
    case 'B': return jieqi::Kind::Elephant;
    case 'A': return jieqi::Kind::Advisor;
    case 'C': return jieqi::Kind::Cannon;
    case 'P': return jieqi::Kind::Soldier;
    default: throw std::invalid_argument("invalid reveal character");
    }
}
}  // namespace

extern "C" {
int jq_abi_version(void) { return 5; }
int jq_legal_moves(const char* board, size_t length, int side, uint16_t* output,
                   size_t capacity, size_t* count, char* error, size_t error_capacity) {
    return guarded(error, error_capacity, [&]() -> int {
        if (count == nullptr) throw std::invalid_argument("null move count output");
        const auto moves = position(board, length, side).legal_moves();
        *count = moves.size();
        if (capacity < moves.size()) {
            message(error, error_capacity, "move output buffer too small"); return JQ_BUFFER_TOO_SMALL;
        }
        if (!moves.empty() && output == nullptr) throw std::invalid_argument("null move output");
        for (std::size_t i = 0; i < moves.size(); ++i)
            output[i] = static_cast<std::uint16_t>(moves[i].from * 90 + moves[i].to);
        return JQ_OK;
    });
}
int jq_in_check(const char* board, size_t length, int side, int* checked, char* error, size_t error_capacity) {
    return guarded(error, error_capacity, [&]() -> int {
        if (checked == nullptr) throw std::invalid_argument("null check output");
        *checked = position(board, length, side).in_check(color(side)) ? 1 : 0;
        return JQ_OK;
    });
}
int jq_game_create(const char* board, size_t length, int side, uint32_t idle,
                     uint32_t limit, jq_game** output, char* error, size_t error_capacity) {
    return guarded(error, error_capacity, [&]() -> int {
        if (output == nullptr || limit == 0) throw std::invalid_argument("null game output or zero limit");
        auto game = std::make_unique<jq_game>(position(board, length, side, idle), limit);
        *output = game.release();
        return JQ_OK;
    });
}
void jq_game_destroy(jq_game* game) { delete game; }
int jq_game_advance(jq_game* game, int from, int to, char reveal, char* error, size_t error_capacity) {
    return guarded(error, error_capacity, [&]() -> int {
        if (game == nullptr) throw std::invalid_argument("null game handle");
        game->history.append_game({from, to}, reveal_kind(reveal), game->limit);
        return JQ_OK;
    });
}
int jq_game_snapshot(const jq_game* game, char* board, size_t capacity, int* side,
                       uint32_t* idle, size_t* plies, int* result, char* error, size_t error_capacity) {
    return guarded(error, error_capacity, [&]() -> int {
        if (game == nullptr || board == nullptr || side == nullptr || idle == nullptr || plies == nullptr || result == nullptr)
            throw std::invalid_argument("null snapshot input/output");
        if (capacity < 90) return JQ_BUFFER_TOO_SMALL;
        const auto& current = game->history.current();
        const auto text = current.public_board();
        const auto status = game->history.terminal_status(game->limit);
        std::copy(text.begin(), text.end(), board);
        *side = current.turn() == jieqi::Color::Red ? 0 : 1;
        *idle = current.no_capture_plies();
        *plies = game->history.size() - 1;
        *result = static_cast<int>(status);
        return JQ_OK;
    });
}
}  // extern C
