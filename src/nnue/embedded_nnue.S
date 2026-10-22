/*
  Maskfish, a UCI jieqi engine derived from Pikafish and Stockfish
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

/*
  The default network (maskfish.nnue, zstd-compressed as distributed) embedded
  in the binary, as Stockfish does with incbin. The Makefile defines
  NNUE_EMBEDDING when src/maskfish.nnue (the copy of the network of the hidden
  ref refs/assets/net that `make net` fetches) exists and embed=yes (the
  default); otherwise the embedded network is empty (gEmbeddedNNUESize == 0)
  and the engine needs the EvalFile option.

  This is an assembly file on purpose: it is not part of the LTO, so a universal
  binary (Makefile, Section 6) links it once, from its baseline build, instead
  of once per instruction set. The symbols have C linkage, so the namespace
  renaming of the universal builds does not touch them.

    extern "C" const unsigned char gEmbeddedNNUEData[];
    extern "C" const unsigned int  gEmbeddedNNUESize;
*/

#if defined(__APPLE__) || (defined(_WIN32) && !defined(_WIN64))
    #define NNUE_SYM(x) _##x
#else
    #define NNUE_SYM(x) x
#endif

#if defined(__APPLE__)
    .section __TEXT,__const
#elif defined(_WIN32)
    .section .rdata,"dr"
#else
    .section .rodata
#endif

    .globl NNUE_SYM(gEmbeddedNNUESize)
    .p2align 2
NNUE_SYM(gEmbeddedNNUESize):
    .long Lnnue_end - Lnnue_begin

    .globl NNUE_SYM(gEmbeddedNNUEData)
    .p2align 6
NNUE_SYM(gEmbeddedNNUEData):
Lnnue_begin:
#if defined(NNUE_EMBEDDING)
    .incbin "maskfish.nnue"
#endif
Lnnue_end:
    /* So that gEmbeddedNNUEData always has a byte (an empty net) */
    .byte 0

#if defined(__ELF__)
    .section .note.GNU-stack,"",%progbits
#endif
