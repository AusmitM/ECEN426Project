// wordle_helper.c
// Simple Wordle-style helper: you provide guess + feedback pattern,
// it suggests remaining candidates.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORDS 80000
#define WORD_LEN  5

static char words[MAX_WORDS][WORD_LEN + 1];
static int  word_count = 0;
static int  alive[MAX_WORDS];

static int load_dict(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        return 0;
    }
    char buf[256];
    while (fgets(buf, sizeof(buf), f) && word_count < MAX_WORDS) {
        char *nl = strchr(buf, '\n');
        if (nl) *nl = '\0';
        if (strlen(buf) != WORD_LEN) continue;
        int ok = 1;
        for (int i = 0; i < WORD_LEN; ++i) {
            if (!islower((unsigned char)buf[i])) {
                ok = 0; break;
            }
        }
        if (!ok) continue;
        strcpy(words[word_count], buf);
        alive[word_count] = 1;
        ++word_count;
    }
    fclose(f);
    return 1;
}

static int matches_pattern(const char *word,
                           const char *guess,
                           const char *pattern) {
    // First pass: handle greens.
    int used[WORD_LEN] = {0};

    for (int i = 0; i < WORD_LEN; ++i) {
        if (pattern[i] == 'g') {
            if (word[i] != guess[i]) return 0;
            used[i] = 1;
        }
    }

    // Second pass: yellows and blacks
    for (int i = 0; i < WORD_LEN; ++i) {
        if (pattern[i] == 'y') {
            int found = 0;
            for (int j = 0; j < WORD_LEN; ++j) {
                if (!used[j] && word[j] == guess[i]) {
                    used[j] = 1;
                    found = 1;
                    break;
                }
            }
            if (!found) return 0;
        } else if (pattern[i] == 'b') {
            // letter must not appear in any unused position
            for (int j = 0; j < WORD_LEN; ++j) {
                if (!used[j] && word[j] == guess[i]) {
                    return 0;
                }
            }
        }
    }

    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s dict\n", argv[0]);
        return 1;
    }

    if (!load_dict(argv[1])) {
        return 1;
    }

    setbuf(stdout, NULL);

    char guess[WORD_LEN + 16];
    char pattern[WORD_LEN + 16];

    while (1) {
        printf("? ");

        if (!fgets(guess, sizeof(guess), stdin)) break;
        char *nl = strchr(guess, '\n');
        if (nl) *nl = '\0';

        if (strlen(guess) != WORD_LEN) {
            fprintf(stderr, "Input: guess must be %d letters\n", WORD_LEN);
            continue;
        }

        printf("pattern (g/y/b): ");
        if (!fgets(pattern, sizeof(pattern), stdin)) break;
        nl = strchr(pattern, '\n');
        if (nl) *nl = '\0';

        if (strlen(pattern) != WORD_LEN) {
            fprintf(stderr, "Input: pattern must be %d chars\n", WORD_LEN);
            continue;
        }

        // Filter candidates
        int alive_count = 0;
        for (int i = 0; i < word_count; ++i) {
            if (!alive[i]) continue;
            if (!matches_pattern(words[i], guess, pattern)) {
                alive[i] = 0;
            } else {
                alive_count++;
            }
        }

        if (alive_count == 0) {
            printf("No candidates left.\n");
            break;
        }

        // Suggest the first surviving word
        for (int i = 0; i < word_count; ++i) {
            if (alive[i]) {
                printf("Suggestion: %s\n", words[i]);
                break;
            }
        }
    }

    return 0;
}
