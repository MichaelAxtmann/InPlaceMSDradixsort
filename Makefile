CC=gcc
CFLAGS=-O3
CLIBS=-lpthread -lnuma -lm -mavx
SRC_PATH=src

all:	lsb_32 cmp_32 msb_32 lsb_64 cmp_64 msb_64

lsb_32:	${SRC_PATH}/lsb_32.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c
	${CC} ${CFLAGS} -o lsb_32 ${SRC_PATH}/lsb_32.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c ${CLIBS}

msb_32: ${SRC_PATH}/msb_32.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c
	${CC} ${CFLAGS} -o msb_32 ${SRC_PATH}/msb_32.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c ${CLIBS}

cmp_32: ${SRC_PATH}/cmp_32.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c
	${CC} ${CFLAGS} -o cmp_32 ${SRC_PATH}/cmp_32.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${SRC_PATH}/shuffle.c ${CLIBS}

lsb_64: ${SRC_PATH}/lsb_64.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c
	${CC} ${CFLAGS} -o lsb_64 ${SRC_PATH}/lsb_64.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${CLIBS}

msb_64: ${SRC_PATH}/benchmark_msb_64.c ${SRC_PATH}/msb_64.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c
	${CC} ${CFLAGS} -o msb_64 ${SRC_PATH}/benchmark_msb_64.c ${SRC_PATH}/msb_64.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${CLIBS} -Iinclude

cmp_64: ${SRC_PATH}/cmp_64.c ${SRC_PATH}/init.c ${SRC_PATH}/rand.c ${SRC_PATH}/zipf.c
	${CC} ${CFLAGS} -o cmp_64 ${SRC_PATH}/cmp_64.c ${SRC_PATH}/rand.c ${SRC_PATH}/init.c ${SRC_PATH}/zipf.c ${CLIBS}

clean:
	rm -f lsb_32 msb_32 cmp_32 lsb_64 msb_64 cmp_64
