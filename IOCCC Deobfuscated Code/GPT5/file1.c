// peace_animation.c
// Draws a small banner in the center of the terminal and animates
// some colored particles around it.

#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <time.h>

// Simple wrappers for writing to stdout
static void write_str(const char *s) {
    write(STDOUT_FILENO, s, (size_t)strlen(s));
}
static void write_n(const char *s, size_t n) {
    write(STDOUT_FILENO, s, n);
}

// Terminal size / layout
static int term_rows;
static int term_cols;
static int banner_row;
static int banner_col;

// PRNG state used in the original
static unsigned char prng_state = 0;

// Simple 0–7 pseudo-random generator (mimics original r())
static unsigned random_3bit(void) {
    prng_state ^= 8;
    prng_state++;
    return prng_state % 8;
}

// Move cursor to (row, col), 1-based
static void move_cursor(int row, int col) {
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "\033[%d;%dH", row, col);
    write_n(buf, (size_t)n);
}

// Select SGR color (0–7 => standard foreground colors)
static void set_color(int color_index) {
    char buf[16];
    int n = snprintf(buf, sizeof(buf), "\033[3%dm", color_index % 8);
    write_n(buf, (size_t)n);
}

// Turn bold on / off
static void set_bold(int on) {
    write_str(on ? "\033[1m" : "\033[0m");
}

// Switch to alternate screen, clear, home
static void enter_alt_screen(void) {
    write_str("\033[?1049h");  // alternate screen buffer
    write_str("\033[2J");      // clear screen
    write_str("\033[H");       // home cursor
}

// Leave alternate screen
static void leave_alt_screen(void) {
    write_str("\033[0m");      // reset attributes
    write_str("\033[?1049l");  // normal screen
}

// Query terminal size
static void detect_terminal_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1) {
        term_rows = 24;
        term_cols = 80;
    } else {
        term_rows = ws.ws_row;
        term_cols = ws.ws_col;
    }

    // Banner layout: 3 rows, width of longest line
    static const char *banner[] = {
        "  /\\  PEACE  /\\  ",
        " <  >      <  > ",
        "  \\/        \\/  ",
    };
    int banner_height = 3;
    int banner_width  = (int)strlen(banner[0]);

    banner_row = term_rows / 2 - banner_height / 2;
    if (banner_row < 1) banner_row = 1;
    banner_col = term_cols / 2 - banner_width / 2;
    if (banner_col < 1) banner_col = 1;
}

// Seed PRNG from current time (similar to original u())
static void seed_prng(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    prng_state = (unsigned char)(tv.tv_usec & 0xFF);
}

// Draw the static banner
static void draw_banner(void) {
    static const char *banner[] = {
        "  /\\  PEACE  /\\  ",
        " <  >      <  > ",
        "  \\/        \\/  ",
    };

    for (int i = 0; i < 3; ++i) {
        move_cursor(banner_row + i, banner_col);
        set_bold(1);
        set_color(2 + i); // greenish variations
        write_str(banner[i]);
    }
    set_bold(0);
}

// Particle structure
typedef struct {
    int row;
    int col;
    int dx;
    int dy;
    int color;
} Particle;

#define MAX_PARTICLES 40

static void init_particle(Particle *p) {
    p->row   = banner_row + random_3bit() - 4;
    p->col   = banner_col + random_3bit() * 2;
    p->dx    = (random_3bit() & 1) ? 1 : -1;
    p->dy    = (random_3bit() & 1) ? 1 : -1;
    p->color = 1 + (random_3bit() % 6);
}

// Draw a single frame of particles
static void draw_particles(Particle *particles, int count) {
    for (int i = 0; i < count; ++i) {
        Particle *p = &particles[i];

        // Update position
        p->row += p->dy;
        p->col += p->dx;

        // Bounce on edges
        if (p->row <= 1 || p->row >= term_rows - 1) {
            p->dy = -p->dy;
            p->row += p->dy;
        }
        if (p->col <= 1 || p->col >= term_cols - 1) {
            p->dx = -p->dx;
            p->col += p->dx;
        }

        // Randomly change color (original q()/S())
        if (random_3bit() < 2) {
            set_bold(1);
        } else {
            set_bold(0);
        }

        set_color(p->color);
        move_cursor(p->row, p->col);
        write_str("*");
    }
    set_bold(0);
}

// Sleep ~40ms between frames (original used nanosleep)
static void frame_sleep(void) {
    struct timespec ts = {0, 40 * 1000 * 1000};
    nanosleep(&ts, NULL);
}

int main(void) {
    Particle particles[MAX_PARTICLES];

    detect_terminal_size();
    seed_prng();
    enter_alt_screen();

    draw_banner();

    // Initialize particles
    for (int i = 0; i < MAX_PARTICLES; ++i) {
        init_particle(&particles[i]);
    }

    // Basic animation loop
    for (int frame = 0; frame < 300; ++frame) {
        draw_particles(particles, MAX_PARTICLES);
        frame_sleep();
    }

    leave_alt_screen();
    return 0;
}
