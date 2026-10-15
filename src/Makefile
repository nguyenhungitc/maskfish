# Maskfish, a UCI jieqi engine derived from Pikafish and Stockfish
# Copyright (C) 2026 The Maskfish developers (see AUTHORS file)
# Copyright (C) 2004-2023 The Pikafish developers (see AUTHORS file)
#
# Maskfish is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Maskfish is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


### ==========================================================================
### Section 1. General Configuration
### ==========================================================================

### Establish the operating system name
KERNEL := $(shell uname -s)
ifeq ($(KERNEL),Linux)
	OS := $(shell uname -o)
endif


### Command prefix used to run the freshly built binary (e.g. wine, qemu, sde).
### WINE_PATH is still accepted for backwards compatibility.
ifneq ($(strip $(WINE_PATH)),)
	ifeq ($(strip $(RUN_PREFIX)),)
		RUN_PREFIX := $(WINE_PATH)
	else ifneq ($(strip $(RUN_PREFIX)),$(strip $(WINE_PATH)))
$(warning *** Both RUN_PREFIX and WINE_PATH are set; ignoring WINE_PATH. ***)
	endif
$(warning *** WINE_PATH is deprecated; use RUN_PREFIX instead. ***)
endif

### Target Windows OS
ifeq ($(OS),Windows_NT)
	ifneq ($(COMP),ndk)
		target_windows = yes
	endif
else ifeq ($(COMP),mingw)
	target_windows = yes
	ifeq ($(RUN_PREFIX),)
		RUN_PREFIX := $(shell which wine)
	endif
endif

### Target operating system (may differ from KERNEL when cross-compiling)
TARGET_KERNEL := $(KERNEL)
ifeq ($(COMP),ndk)
	TARGET_KERNEL := Linux
else ifeq ($(target_windows),yes)
	TARGET_KERNEL := Windows
endif

### Executable name
ifeq ($(target_windows),yes)
	EXE = maskfish.exe
else
	EXE = maskfish
endif

### Installation dir definitions
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin

### Built-in benchmark for pgo-builds
### (a binary built with embed=no has no embedded network: it runs with the file)
PGOBENCH = printf "$(if $(filter yes,$(embed)),,setoption name EvalFile value ./maskfish.nnue\n)speedtest\nbench\n" | $(RUN_PREFIX) ./$(EXE)

### Source and object files
### (universal/ holds the entry points of the universal binaries, temp_builds/
### their builds: neither is part of a normal build)
SRCS = $(shell find -L . \( -path './universal' -o -path './temp_builds' \) -prune -o \( -name '*.cpp' -o -name '*.S' \) -print)

HEADERS = $(shell find -L . \( -path './universal' -o -path './temp_builds' \) -prune -o -name '*.h' -print)

OBJS = $(notdir $(patsubst %.cpp,%.o,$(patsubst %.S,%.o,$(SRCS))))

VPATH = $(shell find -L . \( -path './universal' -o -path './temp_builds' \) -prune -o -type d -print | tr '\n' ':')

### ==========================================================================
### Section 2. High-level Configuration
### ==========================================================================
#
# flag                --- Comp switch        --- Description
# ----------------------------------------------------------------------------
#
# debug = yes/no      --- -DNDEBUG           --- Enable/Disable debug mode
# sanitize = none/<sanitizer> ... (-fsanitize )
#                     --- ( undefined )      --- enable undefined behavior checks
#                     --- ( thread    )      --- enable threading error checks
#                     --- ( address   )      --- enable memory access checks
#                     --- ...etc...          --- see compiler documentation for supported sanitizers
# optimize = yes/no   --- (-O3/-fast etc.)   --- Enable/Disable optimizations
# nnue = default/att  --- -DJIEQI_NNUE_ATT   --- Experimental network with attack buckets
#                                                and mid mirror (needs a matching net)
# threats = yes/no    --- -DJIEQI_NNUE_THREATS --- Experimental network with threat features
#                                                (L1 1024, needs a matching net)
# layers = 1/2        --- -DJIEQI_NNUE_LAYERS2 --- Experimental layer stacks (fc0 32 outputs,
#                                                squared fc1 activations, fc2 fed by all
#                                                activations; needs a matching net)
# psqt = yes/no       --- -DJIEQI_NNUE_NO_PSQT --- psqt=no: network without the PSQT part of
#                                                the feature transformer (SFNNv17 of
#                                                Pikafish; needs a matching net)
# ctx = yes/no        --- -DJIEQI_NNUE_CTX   --- ctx=yes: inventory context inputs of the
#                                                layer stacks (AB-JChess; needs a matching net)
# lsblend = yes/no    --- -DJIEQI_NNUE_BLEND --- lsblend=yes: layer stacks selected by the
#                                                uncertainty phase and interpolated (AB-JChess;
#                                                needs a matching net)
# meta = yes/no       --- -DJIEQI_NNUE_META  --- meta=yes: meta input features (pool summaries,
#                                                unidentified losses, visible targets;
#                                                AB-JChess; needs a matching net)
# quant = 1/2         --- -DJIEQI_NNUE_QUANT2 --- quant=2: quantization of current Pikafish
#                                                (feature transformer at scale 255, fc_0 at
#                                                scale 128 and >> 7, fc_2 in fc_0 units;
#                                                needs a matching net)
# stacks = 16/32      --- -DJIEQI_NNUE_STACKS32 --- stacks=32: 32 layer stacks, each piece count
#                                                bucket split by the share of dark pieces
#                                                (Pikafish 32 LayerStack buckets; not with
#                                                lsblend=yes; needs a matching net)
# embed = yes/no      --- -DNNUE_EMBEDDING   --- embed=yes: embed the network of the ref
#                                                refs/assets/net (src/maskfish.nnue, fetched
#                                                by `make net`) in the binary, as Stockfish
# arch = (name)       --- (-arch)            --- Target architecture
# bits = 64/32        --- -DIS_64BIT         --- 64-/32-bit operating system
# prefetch = yes/no   --- -DUSE_PREFETCH     --- Use prefetch asm-instruction
# popcnt = yes/no     --- -DUSE_POPCNT       --- Use popcnt asm-instruction
# pext = yes/no       --- -DUSE_PEXT         --- Use pext x86_64 asm-instruction
# sse = yes/no        --- -msse              --- Use Intel Streaming SIMD Extensions
# mmx = yes/no        --- -mmmx              --- Use Intel MMX instructions
# sse2 = yes/no       --- -msse2             --- Use Intel Streaming SIMD Extensions 2
# ssse3 = yes/no      --- -mssse3            --- Use Intel Supplemental Streaming SIMD Extensions 3
# sse41 = yes/no      --- -msse4.1           --- Use Intel Streaming SIMD Extensions 4.1
# avx2 = yes/no       --- -mavx2             --- Use Intel Advanced Vector Extensions 2
# avxvnni = yes/no    --- -mavxvnni          --- Use Intel Vector Neural Network Instructions AVX
# avx512 = yes/no     --- -mavx512f          --- Use Intel Advanced Vector Extensions 512
# avx512bw = yes/no   --- -mavx512bw         --- Use Intel Byte and Word Instructions 512
# vnni256 = yes/no    --- -mavx256vnni       --- Use Intel Vector Neural Network Instructions 512 with 256bit operands
# vnni512 = yes/no    --- -mavx512vnni       --- Use Intel Vector Neural Network Instructions 512
# avx512icl = yes/no  --- ... multiple ...   --- Use All AVX-512 features available on both Intel Ice Lake and AMD Zen 4
# altivec = yes/no    --- -maltivec          --- Use PowerPC Altivec SIMD extension
# vsx = yes/no        --- -mvsx              --- Use POWER VSX SIMD extension
# neon = yes/no       --- -DUSE_NEON         --- Use ARM SIMD architecture
# dotprod = yes/no    --- -DUSE_NEON_DOTPROD --- Use ARM advanced SIMD Int8 dot product instructions
# lsx = yes/no        --- -mlsx              --- Use Loongson SIMD eXtension
# lasx = yes/no       --- -mlasx             --- use Loongson Advanced SIMD eXtension
#
# Note that Makefile is space sensitive, so when adding new architectures
# or modifying existing flags, you have to make sure there are no extra spaces
# at the end of the line for flag values.
#
# Example of use for these flags:
# make build ARCH=x86-64-avx512 debug=yes sanitize="address undefined"


### 2.1. General and architecture defaults

ifeq ($(ARCH),)
   ARCH = native
endif

ifeq ($(ARCH), native)
   override ARCH := $(shell $(SHELL) ../scripts/get_native_properties.sh | cut -d " " -f 1)
endif

# explicitly check for the list of supported architectures (as listed with make help),
# the user can override with `make ARCH=x86-32-vnni256 SUPPORTED_ARCH=true`
ifeq ($(ARCH), $(filter $(ARCH), \
                 x86-64-universal x86-64-avx512icl x86-64-vnni512 x86-64-vnni256 x86-64-avx512bw x86-64-avx512 x86-64-avxvnni \
                 x86-64-bmi2 x86-64-avx2 x86-64-sse41-popcnt x86-64-modern x86-64-ssse3 x86-64-sse3-popcnt \
                 x86-64 x86-32-sse41-popcnt x86-32-sse2 x86-32 ppc-64 ppc-64-altivec ppc-64-vsx ppc-32 e2k \
                 armv7 armv7-neon armv8 armv8-dotprod arm64-universal apple-silicon general-64 general-32 riscv64 \
                 riscv64-rva23 loongarch64 loongarch64-lsx loongarch64-lasx))
   SUPPORTED_ARCH=true
else
   SUPPORTED_ARCH=false
endif

ifneq (,$(filter $(ARCH),x86-64-universal arm64-universal))
   UNIVERSAL_BUILD=true
else
   UNIVERSAL_BUILD=false
endif

optimize = yes
debug = no
sanitize = none
nnue = default
threats = no
layers = 1
psqt = yes
ctx = no
lsblend = no
meta = no
quant = 1
stacks = 16
embed = yes
bits = 64
prefetch = no
popcnt = no
pext = no
sse = no
mmx = no
sse2 = no
ssse3 = no
sse41 = no
avx2 = no
avxvnni = no
avx512 = no
avx512bw = no
vnni256 = no
vnni512 = no
avx512icl = no
altivec = no
vsx = no
neon = no
dotprod = no
arm_version = 0
lsx = no
lasx = no
rva23 = no
STRIP = strip

ifneq ($(shell which clang-format-20 2> /dev/null),)
	CLANG-FORMAT = clang-format-20
else
	CLANG-FORMAT = clang-format
endif

### 2.2 Architecture specific

ifeq ($(findstring x86,$(ARCH)),x86)

# x86-32/64

ifeq ($(findstring x86-32,$(ARCH)),x86-32)
	arch = i386
	bits = 32
	sse = no
	mmx = yes
else
	arch = x86_64
	sse = yes
	sse2 = yes
endif

ifeq ($(findstring -sse,$(ARCH)),-sse)
	sse = yes
endif

ifeq ($(findstring -popcnt,$(ARCH)),-popcnt)
	popcnt = yes
endif

ifeq ($(findstring -mmx,$(ARCH)),-mmx)
	mmx = yes
endif

ifeq ($(findstring -sse2,$(ARCH)),-sse2)
	sse = yes
	sse2 = yes
endif

ifeq ($(findstring -ssse3,$(ARCH)),-ssse3)
	sse = yes
	sse2 = yes
	ssse3 = yes
endif

ifeq ($(findstring -sse41,$(ARCH)),-sse41)
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
endif

ifeq ($(findstring -modern,$(ARCH)),-modern)
        $(warning *** ARCH=$(ARCH) is deprecated, defaulting to ARCH=x86-64-sse41-popcnt. Execute `make help` for a list of available architectures. ***)
        $(shell sleep 5)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
endif

ifeq ($(findstring -avx2,$(ARCH)),-avx2)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
endif

ifeq ($(findstring -avxvnni,$(ARCH)),-avxvnni)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	avxvnni = yes
	pext = yes
endif

ifeq ($(findstring -bmi2,$(ARCH)),-bmi2)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
endif

ifeq ($(findstring -avx512,$(ARCH)),-avx512)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
endif

ifeq ($(findstring -avx512bw,$(ARCH)),-avx512bw)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
	avx512bw = yes
endif

ifeq ($(findstring -vnni256,$(ARCH)),-vnni256)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
	avx512bw = yes
	vnni256 = yes
endif

ifeq ($(findstring -vnni512,$(ARCH)),-vnni512)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
	avx512bw = yes
	vnni512 = yes
endif

ifeq ($(findstring -avx512icl,$(ARCH)),-avx512icl)
	popcnt = yes
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	avx2 = yes
	pext = yes
	avx512 = yes
	avx512bw = yes
	vnni512 = yes
	avx512icl = yes
endif

ifeq ($(sse),yes)
	prefetch = yes
endif

# 64-bit pext is not available on x86-32
ifeq ($(bits),32)
	pext = no
endif

else

# all other architectures

ifeq ($(ARCH),general-32)
	arch = any
	bits = 32
endif

ifeq ($(ARCH),general-64)
	arch = any
endif

ifeq ($(ARCH),armv7)
	arch = armv7
	prefetch = yes
	bits = 32
	arm_version = 7
endif

ifeq ($(ARCH),armv7-neon)
	arch = armv7
	prefetch = yes
	popcnt = yes
	neon = yes
	bits = 32
	arm_version = 7
endif

ifeq ($(ARCH),armv8)
	arch = armv8
	prefetch = yes
	popcnt = yes
	neon = yes
	arm_version = 8
endif

ifeq ($(ARCH),armv8-dotprod)
	arch = armv8
	prefetch = yes
	popcnt = yes
	neon = yes
	dotprod = yes
	arm_version = 8
endif

ifeq ($(ARCH),arm64-universal)
	arch = armv8
	prefetch = yes
	popcnt = yes
	neon = yes
	arm_version = 8
endif

ifeq ($(ARCH),apple-silicon)
	arch = arm64
	prefetch = yes
	popcnt = yes
	neon = yes
	dotprod = yes
	arm_version = 8
endif

ifeq ($(ARCH),ppc-32)
	arch = ppc
	bits = 32
endif

ifeq ($(ARCH),ppc-64)
	arch = ppc64
	popcnt = yes
	prefetch = yes
endif

ifeq ($(ARCH),ppc-64-altivec)
	arch = ppc64
	popcnt = yes
	prefetch = yes
	altivec = yes
endif

ifeq ($(ARCH),ppc-64-vsx)
	arch = ppc64
	popcnt = yes
	prefetch = yes
	vsx = yes
endif

ifeq ($(findstring e2k,$(ARCH)),e2k)
	arch = e2k
	mmx = yes
	bits = 64
	sse = yes
	sse2 = yes
	ssse3 = yes
	sse41 = yes
	popcnt = yes
endif

ifeq ($(findstring riscv64,$(ARCH)),riscv64)
	arch = riscv64
endif

ifeq ($(ARCH),riscv64-rva23)
	rva23 = yes
	popcnt = yes
	prefetch = yes
endif

ifeq ($(findstring loongarch64,$(ARCH)),loongarch64)
	arch = loongarch64
	prefetch = yes

ifeq ($(findstring -lasx,$(ARCH)),-lasx)
	lsx = yes
	lasx = yes
endif

ifeq ($(findstring -lsx,$(ARCH)),-lsx)
	lsx = yes
endif

endif
endif


### ==========================================================================
### Section 3. Low-level Configuration
### ==========================================================================

### 3.1 Selecting compiler (default = gcc)
ifeq ($(MAKELEVEL),0)
       export ENV_CXXFLAGS := $(CXXFLAGS)
       export ENV_DEPENDFLAGS := $(DEPENDFLAGS)
       export ENV_LDFLAGS := $(LDFLAGS)
endif

CXXFLAGS = $(ENV_CXXFLAGS) -Wall -Wcast-qual -fno-exceptions -std=c++17 $(EXTRACXXFLAGS)

DEPENDFLAGS = $(ENV_DEPENDFLAGS) -std=c++17
LDFLAGS = $(ENV_LDFLAGS) $(EXTRALDFLAGS)

ifeq ($(COMP),)
	COMP=gcc
endif

ifeq ($(COMP),gcc)
	comp=gcc
	CXX=g++
	CXXFLAGS += -pedantic -Wextra -Wshadow -Wmissing-declarations

	ifeq ($(arch),$(filter $(arch),armv7 armv8 riscv64))
		ifeq ($(OS),Android)
			CXXFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
		ifeq ($(arch),riscv64)
			CXXFLAGS += -latomic
		endif
	else ifeq ($(arch),loongarch64)
		CXXFLAGS += -latomic
	else
		CXXFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
	endif

	ifeq ($(arch),$(filter $(arch),armv7))
		LDFLAGS += -latomic
	endif

	ifneq ($(KERNEL),Darwin)
	   LDFLAGS += -Wl,--no-as-needed
	endif
endif

ifeq ($(target_windows),yes)
	LDFLAGS += -static
endif

ifeq ($(COMP),mingw)
	comp=mingw

	ifeq ($(bits),64)
		ifeq ($(shell which x86_64-w64-mingw32-c++-posix 2> /dev/null),)
			CXX=x86_64-w64-mingw32-c++
		else
			CXX=x86_64-w64-mingw32-c++-posix
		endif
	else
		ifeq ($(shell which i686-w64-mingw32-c++-posix 2> /dev/null),)
			CXX=i686-w64-mingw32-c++
		else
			CXX=i686-w64-mingw32-c++-posix
		endif
	endif
	CXXFLAGS += -pedantic -Wextra -Wshadow -Wmissing-declarations
endif

ifeq ($(COMP),icx)
	comp=icx
	CXX=icpx
	CXXFLAGS += --intel -pedantic -Wextra -Wshadow -Wmissing-prototypes \
		-Wconditional-uninitialized -Wabi -Wdeprecated
endif

ifeq ($(COMP),clang)
	comp=clang
	CXX=clang++
	ifeq ($(target_windows),yes)
		CXX=x86_64-w64-mingw32-clang++
	endif

	CXXFLAGS += -pedantic -Wextra -Wshadow -Wmissing-prototypes \
	            -Wconditional-uninitialized

	ifeq ($(filter $(KERNEL),Darwin OpenBSD FreeBSD),)
	ifeq ($(target_windows),)
	ifneq ($(RTLIB),compiler-rt)
		LDFLAGS += -latomic
	endif
	endif
	endif

	ifeq ($(arch),$(filter $(arch),armv7 armv8 riscv64))
		ifeq ($(OS),Android)
			CXXFLAGS += -m$(bits)
			LDFLAGS += -m$(bits)
		endif
		ifeq ($(arch),riscv64)
			CXXFLAGS += -latomic
		endif
	else ifeq ($(arch),loongarch64)
		CXXFLAGS += -latomic
	else
		CXXFLAGS += -m$(bits)
		LDFLAGS += -m$(bits)
	endif
endif

### macOS: minimum version (MACOSX_VERSION_MIN, empty for the compiler's
### default) and target architecture, also used by the universal binaries
MACOSX_VERSION_MIN ?= 10.15
mac_target_flags :=

ifeq ($(TARGET_KERNEL),Darwin)
	ifneq ($(MACOSX_VERSION_MIN),)
		mac_target_flags += -mmacosx-version-min=$(MACOSX_VERSION_MIN)
	endif
	ifneq ($(arch),any)
		mac_target_flags += -arch $(arch)
	endif
	CXXFLAGS += $(mac_target_flags)
	LDFLAGS += $(mac_target_flags)
	XCRUN = xcrun
endif

# To cross-compile for Android, use NDK version r27c or later.
ifeq ($(COMP),ndk)
	CXXFLAGS += -stdlib=libc++
	comp=clang
	ifeq ($(arch),armv7)
		CXX=armv7a-linux-androideabi29-clang++
		CXXFLAGS += -mthumb -march=armv7-a -mfloat-abi=softfp -mfpu=neon
		ifneq ($(shell which arm-linux-androideabi-strip 2>/dev/null),)
			STRIP=arm-linux-androideabi-strip
		else
			STRIP=llvm-strip
		endif
	endif
	ifeq ($(arch),armv8)
		CXX=aarch64-linux-android29-clang++
		ifneq ($(shell which aarch64-linux-android-strip 2>/dev/null),)
			STRIP=aarch64-linux-android-strip
		else
			STRIP=llvm-strip
		endif
	endif
	ifeq ($(arch),x86_64)
		CXX=x86_64-linux-android29-clang++
		ifneq ($(shell which x86_64-linux-android-strip 2>/dev/null),)
			STRIP=x86_64-linux-android-strip
		else
			STRIP=llvm-strip
		endif
	endif
	LDFLAGS += -static-libstdc++
endif

### Allow overwriting CXX from command line
ifdef COMPCXX
	CXX=$(COMPCXX)
endif

# llvm-profdata must be version compatible with the specified CXX (be it clang, or the gcc alias)
# make -j profile-build CXX=clang++-20 COMP=clang
# Locate the version in the same directory as the compiler used,
# with fallback to a generic one if it can't be located
	LLVM_PROFDATA := $(dir $(realpath $(shell which $(CXX) 2> /dev/null)))llvm-profdata
# for icx
ifeq ($(wildcard $(LLVM_PROFDATA)),)
	LLVM_PROFDATA := $(dir $(realpath $(shell which $(CXX) 2> /dev/null)))/compiler/llvm-profdata
endif
ifeq ($(wildcard $(LLVM_PROFDATA)),)
	LLVM_PROFDATA := llvm-profdata
endif

ifeq ($(comp),icx)
	profile_make = icx-profile-make
	profile_use = icx-profile-use
else ifeq ($(comp),clang)
	profile_make = clang-profile-make
	profile_use = clang-profile-use
else
	profile_make = gcc-profile-make
	profile_use = gcc-profile-use
	ifeq ($(KERNEL),Darwin)
		EXTRAPROFILEFLAGS = -fvisibility=hidden
	endif
endif

### Sometimes gcc is really clang
ifeq ($(COMP),gcc)
	gccversion := $(shell $(CXX) --version 2>/dev/null)
	gccisclang := $(findstring clang,$(gccversion))
	ifneq ($(gccisclang),)
		profile_make = clang-profile-make
		profile_use = clang-profile-use
	else
		gccmajorv := $(shell $(CXX) -dumpversion 2>/dev/null)
		CXXFLAGS += -Wstack-usage=128000 -fno-ipa-cp-clone
	endif
else ifeq ($(COMP),mingw)
	CXXFLAGS += -fno-ipa-cp-clone
endif

# On Android Bionic's C library comes with its own pthread implementation bundled in
ifneq ($(OS),Android)
	# Haiku has pthreads in its libroot, so only link it in on other platforms
	ifneq ($(KERNEL),Haiku)
		ifneq ($(COMP),ndk)
			LDFLAGS += -lpthread

			add_lrt = yes
			ifeq ($(target_windows),yes)
				add_lrt = no
			endif

			ifeq ($(KERNEL),Darwin)
				add_lrt = no
			endif

			ifeq ($(add_lrt),yes)
				LDFLAGS += -lrt
			endif
		endif
	endif
endif

### 3.2.0 Network architecture
ifeq ($(nnue),att)
	CXXFLAGS += -DJIEQI_NNUE_ATT
endif
ifeq ($(threats),yes)
	CXXFLAGS += -DJIEQI_NNUE_THREATS
endif
ifeq ($(layers),2)
	CXXFLAGS += -DJIEQI_NNUE_LAYERS2
endif
ifeq ($(psqt),no)
	CXXFLAGS += -DJIEQI_NNUE_NO_PSQT
endif
ifeq ($(ctx),yes)
	CXXFLAGS += -DJIEQI_NNUE_CTX
endif
ifeq ($(lsblend),yes)
	CXXFLAGS += -DJIEQI_NNUE_BLEND
endif
ifeq ($(meta),yes)
	CXXFLAGS += -DJIEQI_NNUE_META
endif
ifeq ($(quant),2)
	CXXFLAGS += -DJIEQI_NNUE_QUANT2
endif
ifeq ($(stacks),32)
	ifeq ($(lsblend),yes)
		$(error stacks=32 and lsblend=yes cannot be combined)
	endif
	CXXFLAGS += -DJIEQI_NNUE_STACKS32
else ifneq ($(stacks),16)
	$(error stacks must be 16 or 32)
endif

### 3.2.0.1 Embedded network
# The default network src/maskfish.nnue (the copy of the network of the ref
# refs/assets/net that `make net` fetches: every build target runs it first,
# before the make that compiles) is embedded in the binary by
# nnue/embedded_nnue.S; without it the binary has no embedded network and needs
# the EvalFile option, as with embed=no.
ifeq ($(embed),yes)
ifneq (,$(wildcard maskfish.nnue))
	NNUE_EMBED_FLAGS = -DNNUE_EMBEDDING
endif
endif

### 3.2.1 Floating point
# The chance nodes average win rates computed in double precision: without
# this flag, targets with FMA (x86-64 AVX-512 builds, ARMv8) contract a*b+c
# into a fused multiply-add, which rounds differently, so that the search, and
# the bench signature, would depend on the build.
CXXFLAGS += -ffp-contract=off

### 3.2.2 Debugging
ifeq ($(debug),no)
	CXXFLAGS += -DNDEBUG
else
	CXXFLAGS += -g
	CXXFLAGS += -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG
endif

### 3.2.3 Debugging with undefined behavior sanitizers
ifneq ($(sanitize),none)
        CXXFLAGS += -g3 $(addprefix -fsanitize=,$(sanitize))
        LDFLAGS += $(addprefix -fsanitize=,$(sanitize))
endif

### 3.3 Optimization
ifeq ($(optimize),yes)

	CXXFLAGS += -O3 -funroll-loops

	ifeq ($(comp),gcc)
		ifeq ($(OS), Android)
			CXXFLAGS += -fno-gcse -mthumb -march=armv7-a -mfloat-abi=softfp
		endif
	endif

	ifeq ($(KERNEL),Darwin)
		ifeq ($(comp),$(filter $(comp),clang icx))
			CXXFLAGS += -mdynamic-no-pic
		endif

		ifeq ($(comp),gcc)
			ifneq ($(arch),arm64)
				CXXFLAGS += -mdynamic-no-pic
			endif
		endif
	endif

	ifeq ($(comp),clang)
		clangmajorversion := $(shell $(CXX) -dumpversion 2>/dev/null | cut -f1 -d.)
		ifeq ($(shell expr $(clangmajorversion) \< 16),1)
			CXXFLAGS += -fexperimental-new-pass-manager
		endif
	endif
endif

### 3.4 Bits
ifeq ($(bits),64)
	CXXFLAGS += -DIS_64BIT
endif

### 3.5 prefetch and popcount
ifeq ($(prefetch),yes)
	ifeq ($(sse),yes)
		CXXFLAGS += -msse
	endif
else
	CXXFLAGS += -DNO_PREFETCH
endif

ifeq ($(popcnt),yes)
	ifeq ($(arch),$(filter $(arch),ppc64 ppc64-altivec ppc64-vsx armv7 armv8 arm64 riscv64))
		CXXFLAGS += -DUSE_POPCNT
	else
		CXXFLAGS += -msse3 -mpopcnt -DUSE_POPCNT
	endif
endif

### 3.6 SIMD architectures
ifeq ($(avx2),yes)
	CXXFLAGS += -DUSE_AVX2
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx2 -mbmi
	endif
endif

ifeq ($(avxvnni),yes)
	CXXFLAGS += -DUSE_VNNI -DUSE_AVXVNNI
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavxvnni
	endif
endif

ifeq ($(avx512),yes)
	CXXFLAGS += -DUSE_AVX512
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx512f
	endif
endif

ifeq ($(avx512bw),yes)
	CXXFLAGS += -DUSE_AVX512BW
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx512bw
	endif
endif

ifeq ($(vnni256),yes)
	CXXFLAGS += -DUSE_VNNI
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx512vnni -mavx512dq -mavx512vl -mprefer-vector-width=256
	endif
endif

ifeq ($(vnni512),yes)
	CXXFLAGS += -DUSE_VNNI
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx512vnni -mavx512dq -mavx512vl -mprefer-vector-width=512
	endif
endif

ifeq ($(avx512icl),yes)
	CXXFLAGS += -DUSE_AVX512ICL
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mavx512cd -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vpopcntdq -mavx512bitalg -mvpclmulqdq -mgfni -mvaes
	endif
endif

ifeq ($(sse41),yes)
	CXXFLAGS += -DUSE_SSE41
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -msse4.1
	endif
endif

ifeq ($(ssse3),yes)
	CXXFLAGS += -DUSE_SSSE3
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mssse3
	endif
endif

ifeq ($(sse2),yes)
	CXXFLAGS += -DUSE_SSE2
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -msse2
	endif
endif

ifeq ($(mmx),yes)
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mmmx
	endif
endif

ifeq ($(altivec),yes)
	CXXFLAGS += -maltivec
	ifeq ($(COMP),gcc)
		CXXFLAGS += -mabi=altivec
	endif
endif

ifeq ($(vsx),yes)
	CXXFLAGS += -mvsx
	ifeq ($(COMP),gcc)
		CXXFLAGS += -DNO_WARN_X86_INTRINSICS -DUSE_SSE2
	endif
endif

ifeq ($(neon),yes)
	CXXFLAGS += -DUSE_NEON=$(arm_version)
	ifeq ($(KERNEL),Linux)
	ifneq ($(COMP),ndk)
	ifneq ($(arch),armv8)
		CXXFLAGS += -mfpu=neon
	endif
	endif
	endif
endif

ifeq ($(dotprod),yes)
	CXXFLAGS += -march=armv8.2-a+dotprod -DUSE_NEON_DOTPROD
endif

ifeq ($(rva23),yes)
	CXXFLAGS += -march=rv64gcv1p0_zba_zbb_zbs_zicbop_zicond -DUSE_RVV
endif

ifeq ($(lasx),yes)
	CXXFLAGS += -DUSE_LASX
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mlasx
	endif
endif

ifeq ($(lsx),yes)
	CXXFLAGS += -DUSE_LSX
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mlsx
	endif
endif

### 3.7 pext
ifeq ($(pext),yes)
	CXXFLAGS += -DUSE_PEXT
	ifeq ($(comp),$(filter $(comp),gcc clang mingw icx))
		CXXFLAGS += -mbmi2
	endif
endif

### 3.8.1 Try to include git commit sha for versioning
GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null | cut -c 1-8)
ifneq ($(GIT_SHA), )
	CXXFLAGS += -DGIT_SHA=$(GIT_SHA)
endif

### 3.8.2 Try to include git commit date for versioning
GIT_DATE := $(shell git show -s --date=format:'%Y%m%d' --format=%cd HEAD 2>/dev/null)
ifneq ($(GIT_DATE), )
	CXXFLAGS += -DGIT_DATE=$(GIT_DATE)
endif

### 3.8.3 Try to include architecture
ifneq ($(ARCH), )
	CXXFLAGS += -DARCH=$(ARCH)
endif

### 3.9 Link Time Optimization
### This is a mix of compile and link time options because the lto link phase
### needs access to the optimization flags.
### (GCC 14 fails to compile the RISC-V vector intrinsics with LTO: none then.)
lto = yes
ifeq ($(rva23)$(comp)$(gccisclang),yesgcc)
	lto = no
endif

ifeq ($(optimize),yes)
ifeq ($(debug),no)
ifeq ($(lto),yes)
	ifneq ($(KERNEL),Darwin)
		LLD_BIN := $(shell command -v ld.lld 2>/dev/null)
		ifeq ($(LLD_BIN),)
			LLD_BIN := $(shell command -v lld 2>/dev/null)
		endif
		ifneq ($(LLD_BIN),)
			ifeq ($(comp),clang)
				LDFLAGS += -fuse-ld=lld
			else ifeq ($(comp),gcc)
				ifneq ($(gccisclang),)
					LDFLAGS += -fuse-ld=lld
				endif
			endif
		endif
	endif

	ifeq ($(comp),$(filter $(comp),clang icx))
		CXXFLAGS += -flto=full
		ifeq ($(comp),icx)
			CXXFLAGS += -fwhole-program-vtables
		endif
		LDFLAGS += $(CXXFLAGS)

# GCC and CLANG use different methods for parallelizing LTO and CLANG pretends to be
# GCC on some systems.
	else ifeq ($(comp),gcc)
		ifeq ($(gccisclang),)
			# lto-partition=balanced is much faster on multi-core systems and
			# has not been observed to reduce performance for gcc > 13.
			# It is however not compatible with universal builds (one
			# LTO-optimized object per build).
			ifeq ($(UNIVERSAL_SLICE),true)
				CXXFLAGS += -flto -flto-partition=one
			else ifeq ($(shell test "$(gccmajorv)" -le 13 2>/dev/null && echo true),true)
				CXXFLAGS += -flto -flto-partition=one
			else
				CXXFLAGS += -flto -flto-partition=balanced
			endif
			LDFLAGS += $(CXXFLAGS) -flto=jobserver
		else
			CXXFLAGS += -flto=full
			LDFLAGS += $(CXXFLAGS)
		endif

# To use LTO and static linking on Windows,
# the tool chain requires gcc version 10.1 or later.
	else ifeq ($(comp),mingw)
		CXXFLAGS += -flto -flto-partition=one
		LDFLAGS += $(CXXFLAGS) -save-temps
	endif
endif
endif
endif

### 3.10 Android 5 can only run position independent executables. Note that this
### breaks Android 4.0 and earlier.
ifeq ($(OS), Android)
	CXXFLAGS += -fPIE
	LDFLAGS += -fPIE -pie
endif

### ==========================================================================
### Section 4. Public Targets
### ==========================================================================

help:
	@echo "" && \
	echo "To compile maskfish, type: " && \
	echo "" && \
	echo "make -j target [ARCH=arch] [COMP=compiler] [COMPCXX=cxx]" && \
	echo "" && \
	echo "Supported targets:" && \
	echo "" && \
	echo "help                    > Display architecture details" && \
	echo "profile-build           > standard build with profile-guided optimization" && \
	echo "build                   > skip profile-guided optimization" && \
	echo "macos-lipo              > macOS x86-64 + Apple silicon universal binary" && \
	echo "net                     > Fetch the default nnue net (maskfish.nnue) from the ref refs/assets/net" && \
	echo "strip                   > Strip executable" && \
	echo "install                 > Install executable" && \
	echo "clean                   > Clean up" && \
	echo "" && \
	echo "Supported archs:" && \
	echo "" && \
	echo "native                  > select the best architecture for the host processor (default)" && \
	echo "x86-64-universal        > x86 64-bit with automatic runtime selection of the best architecture" && \
	echo "x86-64-avx512icl        > x86 64-bit with minimum avx512 support of Intel Ice Lake or AMD Zen 4" && \
	echo "x86-64-vnni512          > x86 64-bit with vnni 512bit support" && \
	echo "x86-64-vnni256          > x86 64-bit with vnni 512bit support, limit operands to 256bit wide" && \
	echo "x86-64-avx512bw         > x86 64-bit with avx512bw support" && \
	echo "x86-64-avx512           > x86 64-bit with avx512 support" && \
	echo "x86-64-avxvnni          > x86 64-bit with vnni 256bit support" && \
	echo "x86-64-bmi2             > x86 64-bit with bmi2 support" && \
	echo "x86-64-avx2             > x86 64-bit with avx2 support" && \
	echo "x86-64-sse41-popcnt     > x86 64-bit with sse41 and popcnt support" && \
	echo "x86-64-modern           > deprecated, currently x86-64-sse41-popcnt" && \
	echo "x86-64-ssse3            > x86 64-bit with ssse3 support" && \
	echo "x86-64-sse3-popcnt      > x86 64-bit with sse3 compile and popcnt support" && \
	echo "x86-64                  > x86 64-bit generic (with sse2 support)" && \
	echo "x86-32-sse41-popcnt     > x86 32-bit with sse41 and popcnt support" && \
	echo "x86-32-sse2             > x86 32-bit with sse2 support" && \
	echo "x86-32                  > x86 32-bit generic (with mmx compile support)" && \
	echo "ppc-64                  > PPC 64-bit" && \
	echo "ppc-64-altivec          > PPC 64-bit with altivec support" && \
	echo "ppc-64-vsx              > PPC 64-bit with vsx support" && \
	echo "ppc-32                  > PPC 32-bit" && \
	echo "armv7                   > ARMv7 32-bit" && \
	echo "armv7-neon              > ARMv7 32-bit with popcnt and neon" && \
	echo "armv8                   > ARMv8 64-bit with popcnt and neon" && \
	echo "armv8-dotprod           > ARMv8 64-bit with popcnt, neon and dot product support" && \
	echo "arm64-universal         > ARMv8 64-bit with automatic runtime selection of the best architecture (not macOS)" && \
	echo "e2k                     > Elbrus 2000" && \
	echo "apple-silicon           > Apple silicon ARM64" && \
	echo "general-64              > unspecified 64-bit" && \
	echo "general-32              > unspecified 32-bit" && \
	echo "riscv64                 > RISC-V 64-bit" && \
	echo "riscv64-rva23           > RISC-V 64-bit RVA23 profile (vector extension)" && \
	echo "loongarch64             > LoongArch 64-bit" && \
	echo "loongarch64-lsx         > LoongArch 64-bit with SIMD eXtension" && \
	echo "loongarch64-lasx        > LoongArch 64-bit with Advanced SIMD eXtension" && \
	echo "" && \
	echo "Supported compilers:" && \
	echo "" && \
	echo "gcc                     > GNU compiler (default)" && \
	echo "mingw                   > GNU compiler with MinGW under Windows" && \
	echo "clang                   > LLVM Clang compiler" && \
	echo "icx                     > Intel oneAPI DPC++/C++ Compiler" && \
	echo "ndk                     > Google NDK to cross-compile for Android" && \
	echo "" && \
	echo "Simple examples. If you don't know what to do, you likely want to run one of: " && \
	echo "" && \
	echo "make -j profile-build ARCH=x86-64-avx2    # typically a fast compile for common systems " && \
	echo "make -j profile-build ARCH=x86-64-sse41-popcnt  # A more portable compile for 64-bit systems " && \
	echo "make -j profile-build ARCH=x86-64         # A portable compile for 64-bit systems " && \
	echo "" && \
	echo "Advanced examples, for experienced users: " && \
	echo "" && \
	echo "make -j profile-build ARCH=x86-64-avxvnni" && \
	echo "make -j profile-build ARCH=x86-64-avxvnni COMP=gcc COMPCXX=g++-12.0" && \
	echo "make -j build ARCH=x86-64-universal COMP=gcc  # one binary for every x86-64 CPU" && \
	echo 'make -j profile-build ARCH=x86-64-universal RUN_PREFIX="/path/to/sde -future --"' && \
	echo "make -j build ARCH=x86-64-ssse3 COMP=clang" && \
	echo ""
ifneq ($(SUPPORTED_ARCH), true)
	@echo "Specify a supported architecture with the ARCH option for more details"
	@echo ""
endif


.PHONY: help analyze build profile-build macos-lipo strip install clean net net-pgo \
	objclean profileclean config-sanity \
	config-sanity \
	icx-profile-use icx-profile-make \
	gcc-profile-use gcc-profile-make \
	clang-profile-use clang-profile-make FORCE \
	format analyze

analyze: net config-sanity objclean
	$(MAKE) -k ARCH=$(ARCH) COMP=$(COMP) $(notdir $(patsubst %.cpp,%.o,$(filter-out ./external/%,$(filter %.cpp,$(SRCS)))))

ifeq ($(UNIVERSAL_BUILD),false)
build: net config-sanity
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) all

profile-build: net-pgo config-sanity objclean profileclean
	@echo ""
	@echo "Step 1/4. Building instrumented executable ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) $(profile_make)
	@echo ""
	@echo "Step 2/4. Running benchmark for pgo-build ..."
	$(PGOBENCH) > PGOBENCH.out 2>&1
	tail -n 4 PGOBENCH.out
	@echo ""
	@echo "Step 3/4. Building optimized executable ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) objclean
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) $(profile_use)
	@echo ""
	@echo "Step 4/4. Deleting profile data ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) profileclean
endif

macos-lipo:
	@echo ""
	@echo "Step 1/4. Building x86-64 slice ..."
	CXXFLAGS="-arch x86_64" $(MAKE) build ARCH=x86-64-universal COMP=$(COMP)
	mv $(EXE) $(EXE)-x86_64
	@echo ""
	@echo "Step 2/4. Building Apple silicon slice ..."
	$(MAKE) build ARCH=apple-silicon COMP=$(COMP)
	mv $(EXE) $(EXE)-arm64
	@echo ""
	@echo "Step 3/4. Combining slices ..."
	lipo -create $(EXE)-x86_64 $(EXE)-arm64 -output $(EXE)
	@echo ""
	@echo "Step 4/4. Deleting slices ..."
	@rm -f $(EXE)-x86_64 $(EXE)-arm64
	@echo "Universal macOS binary built: $(EXE)"

### Objects of the builds of a universal binary (Section 6): one build is
### linked normally, then its LTO-optimized object (all the engine code, with
### every symbol internal except Maskfish_<arch>::main) becomes maskfish.o.
ifeq ($(TARGET_KERNEL),Darwin)
# Apple uses ld64 which doesn't have --save-temps
  LTO_OBJ_SUFFIX := .lto.o
  SAVE_TEMPS := -Wl,-object_path_lto,$(basename $(EXE)).lto.o
else ifeq ($(comp), clang)
  LTO_OBJ_SUFFIX := .lto.o
  ifeq ($(target_windows),yes)
    SAVE_TEMPS := -Wl,--lto-emit-asm  # ld.lld doesn't have a save temps functionality
    use_lto_emit_asm := yes
  else
    SAVE_TEMPS := -Wl,--save-temps
  endif
else
  LTO_OBJ_SUFFIX := .ltrans0.ltrans.o
  SAVE_TEMPS     := -save-temps
endif

ifeq ($(arch),armv8)
  ASM_FLAGS := -mcpu=neoverse-v2
endif

MODNAME := $(subst -,_,$(notdir $(CURDIR)))

universal-object-pgo: objclean profileclean universalclean
	@echo ""
	@echo "Step 1/4. Building instrumented executable ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) $(profile_make)
	@echo ""
	@echo "Step 2/4. Running benchmark for pgo-build ..."
	$(PGOBENCH) > PGOBENCH.out 2>&1
	tail -n 4 PGOBENCH.out
	@echo ""
	@echo "Step 3/4. Building optimized object files ..."
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) objclean
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) ENV_LDFLAGS='$(ENV_LDFLAGS) $(SAVE_TEMPS)' $(profile_use)
	@echo ""
ifeq ($(use_lto_emit_asm),yes)
	@echo "Step 4/4. Renaming ctors and extracting LTO-optimized relocatable object ..."
	awk -v MODNAME=$(MODNAME) -f ../../universal/rewrite_asm_sections.awk *.lto.s > renamed.s
# -mcpu=neoverse-v2 (or another CPU supporting dotprod) is required for clang
# to not complain when assembling
	$(CXX) -c renamed.s -o maskfish.o $(ASM_FLAGS)
else
	@echo "Step 4/4. Extracting LTO-optimized relocatable object ..."
	cp "$(basename $(EXE))$(LTO_OBJ_SUFFIX)" maskfish.o
endif

universal-object-nopgo: objclean universalclean
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) ENV_LDFLAGS='$(ENV_LDFLAGS) $(SAVE_TEMPS)' all
ifeq ($(use_lto_emit_asm),yes)
	awk -v MODNAME=$(MODNAME) -f ../../universal/rewrite_asm_sections.awk *.lto.s > renamed.s
	$(CXX) -c renamed.s -o maskfish.o $(ASM_FLAGS)
else
	cp "$(basename $(EXE))$(LTO_OBJ_SUFFIX)" maskfish.o
endif

strip:
	$(STRIP) $(EXE)

install:
	-mkdir -p -m 755 $(BINDIR)
	-cp $(EXE) $(BINDIR)
	$(STRIP) $(BINDIR)/$(EXE)

# clean all
clean: objclean profileclean universalclean
	@rm -f .depend *~ core

# clean binaries and objects
objclean:
	@rm -f maskfish maskfish.exe $(shell find . -name '*.o')

# clean auxiliary profiling files
profileclean:
	@rm -rf profdir
	@rm -f bench.txt $(shell find . -name '*.gcda' -o -name '*.gcno') *.s PGOBENCH.out
	@rm -f maskfish.profdata *.profraw
	@rm -f maskfish.*args*
	@rm -f maskfish.*lt*
	@rm -f maskfish.res
	@rm -f ./-lstdc++.res

# evaluation network (nnue)
universalclean:
# On Windows "ln -s" to a directory makes a native directory symlink that rm
# cannot unlink; rmdir unlinks it without following it. No-op elsewhere.
	@find $(TEMP_DIR) -type l -exec rmdir {} + 2>/dev/null || true
	@rm -rf $(TEMP_DIR)

net:
	@$(SHELL) ../scripts/net.sh

# The PGO bench runs the engine, which needs a network: stop before building
# anything if there is none (the engine would exit at the bench, silently)
net-pgo:
	@$(SHELL) ../scripts/net.sh --require

format:
	$(CLANG-FORMAT) -i $(filter %.cpp,$(SRCS)) $(HEADERS) -style=file

### ==========================================================================
### Section 5. Private Targets
### ==========================================================================

all: $(EXE) .depend

config-sanity: net
	@echo "" && \
	echo "Config:" && \
	echo "debug: '$(debug)'" && \
	echo "sanitize: '$(sanitize)'" && \
	echo "optimize: '$(optimize)'" && \
	echo "arch: '$(arch)'" && \
	echo "bits: '$(bits)'" && \
	echo "kernel: '$(KERNEL)'" && \
	echo "os: '$(OS)'" && \
	echo "prefetch: '$(prefetch)'" && \
	echo "popcnt: '$(popcnt)'" && \
	echo "pext: '$(pext)'" && \
	echo "sse: '$(sse)'" && \
	echo "mmx: '$(mmx)'" && \
	echo "sse2: '$(sse2)'" && \
	echo "ssse3: '$(ssse3)'" && \
	echo "sse41: '$(sse41)'" && \
	echo "avx2: '$(avx2)'" && \
	echo "avxvnni: '$(avxvnni)'" && \
	echo "avx512: '$(avx512)'" && \
	echo "avx512bw: '$(avx512bw)'" && \
	echo "vnni256: '$(vnni256)'" && \
	echo "vnni512: '$(vnni512)'" && \
	echo "avx512icl: '$(avx512icl)'" && \
	echo "altivec: '$(altivec)'" && \
	echo "vsx: '$(vsx)'" && \
	echo "neon: '$(neon)'" && \
	echo "dotprod: '$(dotprod)'" && \
	echo "arm_version: '$(arm_version)'" && \
	echo "lsx: '$(lsx)'" && \
	echo "lasx: '$(lasx)'" && \
	echo "target_windows: '$(target_windows)'" && \
	echo "embedded net: $(if $(filter yes,$(embed)),$$(test -f maskfish.nnue && echo maskfish.nnue || echo none),none)" && \
	echo "" && \
	echo "Flags:" && \
	echo "CXX: $(CXX)" && \
	echo "CXXFLAGS: $(CXXFLAGS)" && \
	echo "LDFLAGS: $(LDFLAGS)" && \
	echo "" && \
	echo "Testing config sanity. If this fails, try 'make help' ..." && \
	echo "" && \
	(test "$(debug)" = "yes" || test "$(debug)" = "no") && \
	(test "$(optimize)" = "yes" || test "$(optimize)" = "no") && \
	(test "$(SUPPORTED_ARCH)" = "true") && \
	(test "$(arch)" = "any" || test "$(arch)" = "x86_64" || test "$(arch)" = "i386" || \
	 test "$(arch)" = "ppc64" || test "$(arch)" = "ppc" || test "$(arch)" = "e2k" || \
	 test "$(arch)" = "armv7" || test "$(arch)" = "armv8" || test "$(arch)" = "arm64" || \
	 test "$(arch)" = "riscv64" || test "$(arch)" = "loongarch64") && \
	(test "$(bits)" = "32" || test "$(bits)" = "64") && \
	(test "$(prefetch)" = "yes" || test "$(prefetch)" = "no") && \
	(test "$(popcnt)" = "yes" || test "$(popcnt)" = "no") && \
	(test "$(pext)" = "yes" || test "$(pext)" = "no") && \
	(test "$(sse)" = "yes" || test "$(sse)" = "no") && \
	(test "$(mmx)" = "yes" || test "$(mmx)" = "no") && \
	(test "$(sse2)" = "yes" || test "$(sse2)" = "no") && \
	(test "$(ssse3)" = "yes" || test "$(ssse3)" = "no") && \
	(test "$(sse41)" = "yes" || test "$(sse41)" = "no") && \
	(test "$(avx2)" = "yes" || test "$(avx2)" = "no") && \
	(test "$(avx512)" = "yes" || test "$(avx512)" = "no") && \
	(test "$(avx512bw)" = "yes" || test "$(avx512bw)" = "no") && \
	(test "$(vnni256)" = "yes" || test "$(vnni256)" = "no") && \
	(test "$(vnni512)" = "yes" || test "$(vnni512)" = "no") && \
	(test "$(avx512icl)" = "yes" || test "$(avx512icl)" = "no") && \
	(test "$(altivec)" = "yes" || test "$(altivec)" = "no") && \
	(test "$(vsx)" = "yes" || test "$(vsx)" = "no") && \
	(test "$(neon)" = "yes" || test "$(neon)" = "no") && \
	(test "$(lsx)" = "yes" || test "$(lsx)" = "no") && \
	(test "$(lasx)" = "yes" || test "$(lasx)" = "no") && \
	(test "$(comp)" = "gcc" || test "$(comp)" = "icx" || test "$(comp)" = "mingw" || \
	 test "$(comp)" = "clang" || test "$(comp)" = "armv7a-linux-androideabi16-clang" || \
	 test "$(comp)" = "aarch64-linux-android21-clang")

$(EXE): $(OBJS)
	+$(CXX) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.S
	$(CXX) $(CXXFLAGS)   -c -o $@ $<

# Always assembled again (it is quick), so that the binary has the current
# src/maskfish.nnue, or none after the file was removed or with embed=no
embedded_nnue.o: embedded_nnue.S FORCE
	$(CXX) $(CXXFLAGS) $(NNUE_EMBED_FLAGS) -c -o $@ $<

# Force recompilation to ensure version info is up-to-date
misc.o: FORCE
FORCE:

clang-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-generate ' \
	EXTRALDFLAGS=' -fprofile-generate' \
	all

clang-profile-use:
	$(XCRUN) $(LLVM_PROFDATA) merge -output=maskfish.profdata *.profraw
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-use=maskfish.profdata' \
	EXTRALDFLAGS='-fprofile-use ' \
	all

# -fprofile-update=atomic: the PGO bench runs speedtest with several threads;
# without atomic counters the profile can be inconsistent, and GCC 13 then
# stops with "corrupted profile info" at the -fprofile-use step.
gcc-profile-make:
	@mkdir -p profdir
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-generate=profdir -fprofile-update=atomic' \
	EXTRACXXFLAGS+=$(EXTRAPROFILEFLAGS) \
	EXTRALDFLAGS='-lgcov' \
	all

gcc-profile-use:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-use=profdir -fno-peel-loops -fno-tracer' \
	EXTRACXXFLAGS+=$(EXTRAPROFILEFLAGS) \
	EXTRALDFLAGS='-lgcov' \
	all

icx-profile-make:
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-instr-generate ' \
	EXTRALDFLAGS=' -fprofile-instr-generate' \
	all

icx-profile-use:
	$(XCRUN) $(LLVM_PROFDATA) merge -output=maskfish.profdata *.profraw
	$(MAKE) ARCH=$(ARCH) COMP=$(COMP) \
	EXTRACXXFLAGS='-fprofile-instr-use=maskfish.profdata' \
	EXTRALDFLAGS='-fprofile-use ' \
	all

.depend: $(SRCS)
	-@$(CXX) $(DEPENDFLAGS) -MM $(SRCS) > $@ 2> /dev/null

ifeq (, $(filter $(MAKECMDGOALS), help strip install clean net net-pgo objclean profileclean universalclean format config-sanity))
ifeq ($(UNIVERSAL_BUILD),false)
-include .depend
endif
endif

### ==========================================================================
### Section 6. Universal binary driver
### ==========================================================================
#
# A universal binary holds one build of the engine per instruction set and
# selects the best one for the CPU when it starts (src/universal/). Usage:
#   make -j build ARCH=x86-64-universal          # without PGO
#   make -j profile-build ARCH=x86-64-universal  # PGO (every build runs the bench:
#                                                # RUN_PREFIX="/path/to/sde -future --"
#                                                # for the builds the CPU lacks)
#   make -j build ARCH=arm64-universal
# Each build is made in temp_builds/<arch>/ with the namespace Maskfish renamed
# to Maskfish_<arch>; MASKFISH_UNIVERSAL_ARCH=<arch> at run time forces a build
# (MASKFISH_UNIVERSAL_ARCH=list lists them). The network (embed=yes) is
# embedded once: nnue/embedded_nnue.S is assembly, so its object is linked from
# the baseline build only, and every build uses it.
#
# On Linux the C++ runtime is linked statically (libstdc++, or libc++ with
# UNIVERSAL_LIBCXX=yes and COMP=clang, as Pikafish's release builds).

TEMP_DIR             := $(CURDIR)/temp_builds
UNIVERSAL_EXE        := $(CURDIR)/$(EXE)

ifeq ($(UNIVERSAL_BUILD),true)

ifneq ($(optimize)$(debug),yesno)
$(error A universal binary needs optimize=yes and debug=no (it uses the LTO-optimized objects of its builds))
endif

# The arm64 entry point reads the CPU features with getauxval() (Linux,
# Android) or IsProcessorFeaturePresent() (Windows), and finds the
# initializers of each build in ELF/PE sections: it does not support macOS,
# where every Apple silicon CPU runs the same build anyway.
ifeq ($(ARCH)$(TARGET_KERNEL),arm64-universalDarwin)
ifneq (,$(filter build profile-build,$(MAKECMDGOALS)))
$(error ARCH=arm64-universal is not supported on macOS: use ARCH=apple-silicon, or make macos-lipo for an Intel + Apple silicon binary)
endif
endif

UNIVERSAL_SRC_DIR    := $(CURDIR)/universal
NET_SENTINEL         := $(TEMP_DIR)/.net-done

ifeq ($(arch),x86_64)
  UNIVERSAL_ARCHES := x86-64 x86-64-sse41-popcnt x86-64-avx2 x86-64-bmi2 \
                      x86-64-avxvnni x86-64-avx512bw x86-64-vnni512 x86-64-avx512icl
  UNIVERSAL_ENTRY_OBJ  := $(TEMP_DIR)/entry_x86.o
  UNIVERSAL_ENTRY_SRC  := $(UNIVERSAL_SRC_DIR)/entry_x86.cpp
  BASELINE_ARCH := x86-64
else
  UNIVERSAL_ARCHES := armv8 armv8-dotprod
  UNIVERSAL_ENTRY_OBJ  := $(TEMP_DIR)/entry_arm64.o
  UNIVERSAL_ENTRY_SRC  := $(UNIVERSAL_SRC_DIR)/entry_arm64.cpp
  BASELINE_ARCH := armv8
endif

# Top-level tracked items in src/ (directories and files)
TRACKED_TOP := $(sort $(foreach f,$(patsubst ./%,%,$(SRCS) $(HEADERS)) Makefile,$(firstword $(subst /, ,$(f)))))

# Baseline build must come first in the final link (Windows --allow-multiple-definition
# uses the first copy of inline duplicates)
ARCH_OBJS := $(TEMP_DIR)/$(BASELINE_ARCH)/maskfish.o \
             $(patsubst %,$(TEMP_DIR)/%/maskfish.o,$(filter-out $(BASELINE_ARCH),$(UNIVERSAL_ARCHES)))

# Objects that are not part of the LTO (assembly): those of the baseline build
UNIVERSAL_COMMON_OBJ := $(patsubst %,$(TEMP_DIR)/$(BASELINE_ARCH)/%,$(notdir $(patsubst %.S,%.o,$(filter %.S,$(SRCS)))))

UNIVERSAL_FINAL_FLAGS := -fno-exceptions -Os -pthread

ifeq ($(UNIVERSAL_LIBCXX),yes)
  ifneq ($(comp),clang)
$(error UNIVERSAL_LIBCXX=yes needs COMP=clang)
  endif
  # The builds use libc++ too
  UNIVERSAL_SLICE_CXXFLAGS := -stdlib=libc++
endif

ifeq ($(TARGET_KERNEL),Darwin)
  UNIVERSAL_FINAL_FLAGS += $(mac_target_flags)
else ifeq ($(TARGET_KERNEL),Linux)
  ifeq ($(COMP),ndk)
    UNIVERSAL_FINAL_FLAGS += -static
  else ifeq ($(UNIVERSAL_LIBCXX),yes)
    UNIVERSAL_FINAL_FLAGS += -fuse-ld=lld -nostdlib++ -stdlib=libc++ -rtlib=compiler-rt -unwindlib=libunwind -Wl,-Bstatic -lc++ -lc++abi -lunwind -Wl,-Bdynamic
  else
    UNIVERSAL_FINAL_FLAGS += -static-libstdc++ -static-libgcc
  endif
  ifneq ($(COMP),ndk)
    UNIVERSAL_FINAL_FLAGS += -Wl,--no-as-needed -lpthread -lrt
  endif
else    # windows
  UNIVERSAL_FINAL_FLAGS += -Wl,--allow-multiple-definition -static
  ifeq ($(COMP),clang)
    UNIVERSAL_FINAL_FLAGS += -lpthread -fuse-ld=lld
  endif
endif

# $(call arch-namespace,x86-64-avx2) => Maskfish_x86_64_avx2
arch-suffix     = $(subst -,_,$(1))
arch-namespace  = Maskfish_$(call arch-suffix,$(1))
# Itanium ABI mangled name for Maskfish_<suffix>::main(int, char**)
arch-mangled    = _ZN$(shell printf %s '$(call arch-namespace,$(1))' | wc -c | tr -d ' ')$(call arch-namespace,$(1))4mainEiPPc
# Mach-O initializer section name: strip "x86-64" or "x86-64-" and prefix with "_i_"
# (Mach-O section names are restricted to <=16 chars, hence this naming scheme)
arch-mac-section = _i_$(subst -,_,$(patsubst x86-64,,$(patsubst x86-64-%,%,$(1))))

ifeq ($(TARGET_KERNEL),Darwin)
  arch-defmain = -Wl,-alias,_$(call arch-mangled,$(1)),_main
# lld-link, used on Windows+clang, doesn't accept --defsym, so
# we define main directly in main.cpp instead and delete it later.
else ifeq ($(target_windows)+$(COMP),yes+clang)
  arch-defmain = -DUNIVERSAL_NEEDS_MAIN_SHIM
else
  arch-defmain = -Wl,--defsym=main=$(call arch-mangled,$(1))
endif

arch-cxxflags   = -DMaskfish=$(call arch-namespace,$(1)) -DUNIVERSAL_BINARY $(call arch-defmain,$(1)) \
                  $(UNIVERSAL_SLICE_CXXFLAGS) $(if $(filter clang,$(comp)),-Wno-unused-command-line-argument)

UOBJ_TGT ?= universal-object-pgo
# objcopy for the objects of the target: llvm-objcopy with clang (and the
# NDK), the MinGW one when cross-compiling for Windows
ifeq ($(COMP),$(filter $(COMP),clang ndk))
	OBJCOPY ?= llvm-objcopy
else ifeq ($(COMP)$(shell which x86_64-w64-mingw32-objcopy 2> /dev/null),mingw)
	OBJCOPY ?= objcopy
else ifeq ($(COMP),mingw)
	OBJCOPY ?= x86_64-w64-mingw32-objcopy
else
	OBJCOPY ?= objcopy
endif

.PHONY: build profile-build universalclean
build:         UOBJ_TGT := universal-object-nopgo
build:         $(UNIVERSAL_EXE)
profile-build: UOBJ_TGT := universal-object-pgo
profile-build: $(UNIVERSAL_EXE)

$(TEMP_DIR):
	@mkdir -p $@

# Configuration of the universal binary. The builds in temp_builds/ are made by
# sub-makes, so the targets below do not see their dependencies on their own:
# every build depends on the sources, the Makefile and this file, which is
# rewritten (and so rebuilds everything) only when one of these variables
# changes: compiler, network architecture options, PGO or not, flags, ... It is
# recursively expanded, so that UOBJ_TGT (set for build / profile-build) is the
# one of the goal.
UNIVERSAL_CONFIG := $(TEMP_DIR)/.config
universal_config = arch=$(ARCH) comp=$(COMP) cxx=$(CXX) target=$(UOBJ_TGT) \
                   nnue=$(nnue) threats=$(threats) layers=$(layers) psqt=$(psqt) ctx=$(ctx) lsblend=$(lsblend) meta=$(meta) quant=$(quant) stacks=$(stacks) \
                   embed=$(embed) \
                   libcxx=$(UNIVERSAL_LIBCXX) macos=$(MACOSX_VERSION_MIN) \
                   cxxflags=$(ENV_CXXFLAGS) extracxxflags=$(EXTRACXXFLAGS) \
                   ldflags=$(ENV_LDFLAGS) extraldflags=$(EXTRALDFLAGS) \
                   top=$(TRACKED_TOP)

.PHONY: universal-config-check
universal-config-check:

# printf rather than $(file ...), which make 3.81 (macOS) does not have
$(UNIVERSAL_CONFIG): universal-config-check | $(TEMP_DIR)
	@printf '%s\n' '$(subst ','\'',$(strip $(universal_config)))' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

# Fetch the network of the ref refs/assets/net into src/ at every make
# (scripts/net.sh). The sentinel holds a copy of it and changes only when the
# network changes: the builds, which embed it, are then made again.
$(NET_SENTINEL): FORCE | $(TEMP_DIR)
	@cd $(CURDIR) && $(SHELL) ../scripts/net.sh
	@if [ -f $(CURDIR)/maskfish.nnue ]; then \
	    cmp -s $(CURDIR)/maskfish.nnue $@ || cp $(CURDIR)/maskfish.nnue $@; \
	elif [ ! -e $@ ] || [ -s $@ ]; then \
	    rm -f $@ && touch $@; \
	fi

# With PGO, check that there is a network before any build starts (an
# order-only prerequisite of every build: it runs at each make, but does not
# rebuild them).
.PHONY: universal-net-check
universal-net-check: $(NET_SENTINEL)
	@$(if $(filter universal-object-pgo,$(UOBJ_TGT)),cd $(CURDIR) && $(SHELL) ../scripts/net.sh --require,:)

$(UNIVERSAL_ENTRY_OBJ): $(UNIVERSAL_ENTRY_SRC) $(UNIVERSAL_SRC_DIR)/select.h $(UNIVERSAL_CONFIG) | $(TEMP_DIR)
	$(CXX) -O2 -std=c++20 $(mac_target_flags) $(UNIVERSAL_SLICE_CXXFLAGS) $(EXTRACXXFLAGS) -c $< -o $@

# Symlink tracked top-level items from src/ (and the network, for the PGO
# bench) into temp_builds/<arch>/. Done again when the configuration changes
# (it holds the list of these items).
$(TEMP_DIR)/%/.setup: $(UNIVERSAL_CONFIG) | $(TEMP_DIR)
	@rm -rf $(TEMP_DIR)/$*
	@mkdir -p $(TEMP_DIR)/$*
	@$(foreach f,$(TRACKED_TOP),ln -s $(CURDIR)/$(f) $(TEMP_DIR)/$*/$(f);)
	@ln -s $(CURDIR)/maskfish.nnue $(TEMP_DIR)/$*/maskfish.nnue
	@touch $@

# Per-arch maskfish.o
.SECONDARY: $(patsubst %,$(TEMP_DIR)/%/.setup,$(UNIVERSAL_ARCHES))

# (the sub-make starts with objclean: a change rebuilds the whole build)
# (and on the network, which they embed: NET_SENTINEL)
$(TEMP_DIR)/%/maskfish.o: $(TEMP_DIR)/%/.setup $(NET_SENTINEL) $(UNIVERSAL_CONFIG) \
                          $(SRCS) $(HEADERS) Makefile | universal-net-check
	+ENV_CXXFLAGS='$(ENV_CXXFLAGS) $(call arch-cxxflags,$*)' \
	    $(MAKE) -C $(TEMP_DIR)/$* $(UOBJ_TGT) \
		        ARCH=$* CXX=$(CXX) COMP=$(COMP) UNIVERSAL_SLICE=true
ifeq ($(TARGET_KERNEL),Darwin)
# Rename the initializer section
	ld -r -rename_section __DATA __mod_init_func __DATA $(call arch-mac-section,$*) $@ -o $@.tmp
	mv $@.tmp $@
else
# Drop COMDAT groups; clang LTO leaves them around internalized LOCAL
# template instantiations which causes problems at the final link
	$(OBJCOPY) -R .group $@
ifeq ($(use_lto_emit_asm),yes)
# Delete per-arch main shim
	$(OBJCOPY) -N main $@
else
# Rename .init_array (Linux) or .ctors (Windows) so we can manually invoke them
	$(OBJCOPY) --rename-section .init_array=$(call arch-suffix,$*)_init $@
	$(OBJCOPY) --rename-section .ctors=$(call arch-suffix,$*)_init $@
# Make the array inert
	$(OBJCOPY) --set-section-flags $(call arch-suffix,$*)_init=alloc,data $@
endif
endif

$(UNIVERSAL_EXE): $(UNIVERSAL_ENTRY_OBJ) $(ARCH_OBJS) $(UNIVERSAL_CONFIG)
	$(CXX) -o $@ $(UNIVERSAL_ENTRY_OBJ) $(ARCH_OBJS) $(UNIVERSAL_COMMON_OBJ) $(UNIVERSAL_FINAL_FLAGS) $(EXTRALDFLAGS)
	@echo "Universal binary built: $@"

endif  # UNIVERSAL_BUILD
