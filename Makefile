CC=gcc
CFLAGS=-O3
CLIBS=-lpthread -lnuma -lm -mavx

all:	lsb_32 cmp_32 msb_32 lsb_64 cmp_64 msb_64

lsb_32:	lsb_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o lsb_32 lsb_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

msb_32: msb_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o msb_32 msb_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

cmp_32: cmp_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o cmp_32 cmp_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

lsb_64: lsb_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o lsb_64 lsb_64.c rand.c init.c zipf.c ${CLIBS}

msb_64: msb_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o msb_64 msb_64.c rand.c init.c zipf.c ${CLIBS}

cmp_64: cmp_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o cmp_64 cmp_64.c rand.c init.c zipf.c ${CLIBS}

clean:
	rm -f lsb_32 msb_32 cmp_32 lsb_64 msb_64 cmp_64
