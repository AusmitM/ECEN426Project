#include <stdio.h>
#include <unistd.h>
#include <string.h>

#define MAX_W 9999
#define MAX_H 9999

char canvas[MAX_H][MAX_W] = {0};
char buffer[MAX_H][MAX_W];
int width, height, offset_x = 0, offset_y = 0;
long seed = 0, mod = 33333331, prime = 33331;
int canvas_width = 40, shuffle_seed = 0, temp_var = 0;

// Function to rotate ASCII character if needed
void clean_canvas(int c) {
    for (int i = 0; i < canvas_width * height; i++) {
        int x = i % canvas_width;
        int y = i / canvas_width;
        int ch = canvas[y][x];
        canvas[y][x] = ch ? ch : 32; // Replace null with space
    }
}

// Transformation logic for text shifting and distortion
void transform(int t, int d) {
    for (int i = 0; i < (canvas_width / 2 * 3 + 3); ++i) {
        long long offset = temp_var * canvas_width;
        int is_col = t;
        int range = is_col ? canvas_width : height;
        int direction = (2 * i - (i / 2) * 3 - (i / 3) * 5) * d;

        for (int j = 0; j < range; ++j) {
            int source_x = is_col ? j : i + offset;
            int source_y = is_col ? i + offset : j;

            int src_idx_x = (source_x + direction) % canvas_width;
            int src_idx_y = (source_y + direction) % height;

            src_idx_x = (src_idx_x + canvas_width) % canvas_width;
            src_idx_y = (src_idx_y + height) % height;

            buffer[j][i] = canvas[src_idx_y][src_idx_x];
        }
    }

    // Copy back to canvas
    for (int i = 0; i < height; ++i)
        memcpy(canvas[i], buffer[i], canvas_width);
}

int main(int argc, char** argv) {
    char* input;
    int c;
    int freq[256] = {0};

    mod = 33333331;
    prime = 33331;
    canvas_width = 40;

    // Optional command-line seed
    if (argc > 1 && sscanf(argv[1], "%ld", &canvas_width)) {
        argv++; argc--;
    }

    if (argc > 1) {
        input = argv[1];
        while (*input)
            shuffle_seed = (shuffle_seed * prime + *input++) % mod;
    }

    // Read stdin into canvas
    int x = 0, y = 0;
    while ((c = getchar()) != EOF) {
        freq[c]++;
        if (c == '\n') {
            x = 0;
            y++;
        } else {
            canvas[y][x++] = c;
        }
    }

    height = 2 * canvas_width;

    // Calculate transformation seed
    for (int i = 33; i < 127; i++) {
        temp_var = (prime * temp_var + freq[i]) % prime;
    }

    // Animation loop
    for (int loop = 0; loop <= 32; ++loop) {
        // Trim whitespace at end of each row
        for (int j = MAX_H; j--;) {
            char* line = canvas[j];
            while (*line) line++;
            while (*--line == ' ') *line = 0;
        }

        clean_canvas(0);

        // Print canvas
        if (loop)
            printf("\033[H");
        else
            printf("\033[H\033[2J");

        for (int i = 0; i < height; ++i)
            puts(canvas[i]);

        usleep(prime << 1);

        if (loop < 32)
            transform(loop, loop < temp_var ? 1 : -1);
    }

    return 0;
}
