// moon_phase.c
// Draw an ASCII representation of the current moon phase.

#include <stdio.h>
#include <time.h>
#include <math.h>

int main(void) {
    const int radius = 44;            // radius of the "moon"
    const int diameter = radius * 2;  // width of the output

    // Length of synodic month in seconds (~29.53 days)
    const long lunar_cycle = 2551443L;

    // Original code used an offset 592531 to line up the phase.
    time_t now = time(NULL);
    long phase_raw = (long)((now - 592531L) % lunar_cycle);

    // Scale into 0..511 (<<9 then divide by a)
    unsigned int z = (unsigned int)((phase_raw << 9) / lunar_cycle);

    for (int y = 2 - radius; y <= radius; y += 4) {
        for (int x = -radius; x < radius; ) {

            if (x < 0) {
                // Points on the left of center:
                // decide whether we are inside the circle.
                int inside = (x * x + y * y) < radius * radius;

                if (inside) {
                    // The original code does:
                    // a = 1 - x; return -1; (for next x)
                    // which effectively "jumps" to the right side.
                    x = 1 - x;   // mirrored position on the right
                } else {
                    x += 1;
                }

                putchar(inside ? '#' : ' ');
                continue;
            }

            // We are on the right side: choose '#' or '.'
            // Expression from original:
            // "#."[(x < a*(~z & 255)>>8) ^ (z >> 8)]
            //
            // Here "a" is effectively diameter.
            int threshold = diameter * (int)(~z & 0xFF) >> 8;
            int index = ((x < threshold) ^ (z >> 8)) & 1;
            char ch = "#."[index];

            putchar(ch);
            ++x;
        }
        putchar('\n');
    }

    return 0;
}
