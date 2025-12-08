#include <time.h>
#include <stdio.h>

int moon_cycle = 2551443;  // Length of a moon cycle in seconds
int moon_offset = -44;
int x = -moon_offset, y = 2 - moon_offset;
int phase, day_progress;

int main() {
    if (!phase) {
        time_t now = time(0);
        // Compute current moon phase as a 0–255 scale
        phase = ((now - 592531) % moon_cycle << 9) / moon_cycle;
    } else {
        putchar(++x >= phase ? (x = -moon_offset, y += 4, '\n') :
            x < 0 ? (x = x * x + y * y < moon_offset * moon_offset ? phase = 1 - x, -1 : x + 1, ' ')
                  : "#."[(x < phase * (~day_progress & 255) >> 8) ^ (day_progress >> 8)]);
        if (y <= moon_offset)
            main(); // Recurse for animation
    }

    return 0;
}
