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

#ifndef JIEQI_C_API_H
#define JIEQI_C_API_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
# if defined(JIEQI_BUILD_DLL)
#  define JIEQI_API __declspec(dllexport)
# else
#  define JIEQI_API __declspec(dllimport)
# endif
#else
# define JIEQI_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ABI v5: board = exactly 90 public ASCII cells, rank 0 first; side=0 red/1 black.
 * Errors are caller-owned buffers; game handles are created/freed by this library.
 * No hidden identity is required for move generation or in-check queries.
 * A reveal is uppercase R/N/B/A/C/P for a dark mover, or zero for a bright mover.
 * (v5: the queries that tools/jieqi/native.py did not use were removed: jq_apply_move, jq_basic_status,
 * jq_perft_revealed, jq_repetition_replay, jq_chase_targets, jq_game_replay.) */
enum { JQ_OK = 0, JQ_INVALID = 1, JQ_BUFFER_TOO_SMALL = 2, JQ_INTERNAL = 3 };
JIEQI_API int jq_abi_version(void);
JIEQI_API int jq_legal_moves(const char* board, size_t length, int side,
                             uint16_t* moves, size_t capacity, size_t* count,
                             char* error, size_t error_capacity);
JIEQI_API int jq_in_check(const char* board, size_t length, int side, int* checked,
                          char* error, size_t error_capacity);
/* Opaque public-history handles. One thread owns a handle; do not copy/free it
 * yourself. Destroy exactly once (null is permitted). Invalid/dangling non-null
 * handles are outside the C ABI contract. Limits stay fixed for the game. */
typedef struct jq_game jq_game;
JIEQI_API int jq_game_create(const char* board, size_t length, int side, uint32_t idle,
                             uint32_t limit, jq_game** output, char* error, size_t error_capacity);
JIEQI_API void jq_game_destroy(jq_game* game);
JIEQI_API int jq_game_advance(jq_game* game, int from, int to, char reveal,
                              char* error, size_t error_capacity);
/* Snapshot writes no outputs on failure. Board capacity must be at least 90. */
JIEQI_API int jq_game_snapshot(const jq_game* game, char* board, size_t capacity,
                               int* side, uint32_t* idle, size_t* plies, int* result,
                               char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
