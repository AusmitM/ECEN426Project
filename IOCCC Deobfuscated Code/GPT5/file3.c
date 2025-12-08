// text_wave.c
// Reads text from stdin and animates it with a simple horizontal wave.

#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINES 2000
#define MAX_COLS  2000

static char lines[MAX_LINES][MAX_COLS];
static int  line_count = 0;

static void trim_trailing_spaces(char *s) {
    int len = (int)strlen(s);
    while (len > 0 && s[len - 1] == ' ') {
        s[--len] = '\0';
    }
}

static void read_input(void) {
    while (line_count < MAX_LINES &&
           fgets(lines[line_count], MAX_COLS, stdin) != NULL) {
        char *nl = strchr(lines[line_count], '\n');
        if (nl) *nl = '\0';
        trim_trailing_spaces(lines[line_count]);
        ++line_count;
    }
}

static void clear_screen(void) {
    fputs("\033[2J\033[H", stdout);
}

int main(void) {
    read_input();
    if (line_count == 0) {
        fprintf(stderr, "No input.\n");
        return 1;
    }

    int frame = 0;

    for (;;) {
        clear_screen();

        for (int i = 0; i < line_count; ++i) {
            int shift = (int)(3 * sin((frame + i) / 5.0));
            if (shift < 0) shift = 0;

            // Print leading spaces
            for (int s = 0; s < shift; ++s) putchar(' ');

            puts(lines[i]);
        }

        fflush(stdout);
        usleep(70000); // ~70ms

        ++frame;
    }

    return 0;
}
