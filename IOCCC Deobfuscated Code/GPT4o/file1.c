#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <time.h>

#define write_to_stdout write
extern long write_to_stdout(int, const void*, size_t);
extern double O[];

int term_rows, term_cols;
int col_offset, row_offset;
int state = 0;
int v = 0;

// Terminal manipulation helpers
void write_str(const char* str, long len) {
    write_to_stdout(1, str, len);
}

unsigned char rand_bit_cycle() {
    v ^= 8;
    return ++v % 8;
}

int get_char_at_offset(int index) {
    return ((char*)&O[index])[rand_bit_cycle()];
}

void set_fg_color(int color_index) {
    char* base = (char*)(O + 14);
    write_str(base + color_index * 2, 2);
}

void set_bg_color(int color_index) {
    char* base = (char*)(O + 16);
    write_str(base + color_index * 3, 3);
}

int digit_from_double(int base_index, int offset) {
    return ((int)(O[base_index]) / *((char*)(O + 18) + offset)) % 10 + '0';
}

void seed_random() {
    char buf[16];
    gettimeofday((struct timeval*)buf, NULL);
    v = buf[0];
}

void get_terminal_size() {
    struct winsize ws;
    ioctl(1, TIOCGWINSZ, &ws);
    term_cols = ws.ws_col;
    term_rows = ws.ws_row;
    col_offset = term_rows / 2 - 12;
    row_offset = term_cols - 4;
}

void esc() { write_str("\033[", 2); }
void bold() { esc(); write_str("1m", 2); }
void reset() { esc(); write_str("0m", 2); }
void switch_alt_screen() { esc(); write_str("?1049h", 6); esc(); write_str("2J", 2); esc(); write_str("H", 1); }

void move_cursor(int row, int col) {
    char pos[6] = {
        digit_from_double(row, 2),
        digit_from_double(row, 1),
        digit_from_double(row, 0),
        ';',
        digit_from_double(col, 2),
        digit_from_double(col, 1),
    };
    write_str(pos, 6);
    write_str("H", 1);
}

void apply_color(int color) {
    esc();
    set_fg_color(color);
    write_str("m", 1);
}

void draw_ui() {
    for (int i = 0; i < 18; ++i) {
        move_cursor(row_offset + i / 6, col_offset + (i % 6) * 4);
        apply_color(((char*)&O)[state++]);
        write_str(((char*)&O) + (i << 2) + 24, 4);
    }
}

void delay() {
    struct timespec ts = {0, 1000000};  // ~1ms
    nanosleep(&ts, NULL);
}

void apply_random_style(int index) {
    int styles[] = {3, 0, 1, 2, 0, 1};
    if (rand_bit_cycle() < 2) {
        bold();
        apply_color(styles[index - 2]);
    } else {
        apply_color(styles[index + 1]);
    }
}

void select_style(int code, int x, int y, int* dx, int* dy) {
    switch (code) {
        case 0:
            if (x <= 2 || y < 4) {
                *dx = get_char_at_offset(12);
                *dy = 0;
            } else if (x < 15) {
                *dx = get_char_at_offset(12);
                *dy = -(x % 2);
            } else {
                *dx = get_char_at_offset(12);
                *dy = get_char_at_offset(12) / 2;
            }
            break;
        case 3:
            *dx = get_char_at_offset(12) * 1.5;
            *dy = get_char_at_offset(13);
            break;
        case 4:
            *dx = get_char_at_offset(12) / 2;
            *dy = get_char_at_offset(12) / 2;
            break;
        default:
            *dx = get_char_at_offset(13) * 1.5 * (1 - (code - 1) * 2);
            *dy = get_char_at_offset(13) / 2;
            break;
    }
}

void draw_pattern(int x, int y, int style, int steps) {
    int dx, dy;
    int alternate = 5;

    while (steps-- > 0) {
        int C = 35 - steps;
        select_style(style, steps, C, &dx, &dy);

        if (dy > 0 && y > row_offset - 2) dy--;

        if (steps < 3 || (style == 0 && steps % 3 == 0)) {
            if (--alternate <= 0) {
                alternate = 3;
                draw_pattern(x, y + 5, (state++ % 2) + 1, steps);
            }
        }

        x += dx;
        y += dy;

        if (x < 0) x = 0;
        if (y < 0) y = 0;

        move_cursor(y, x);
        apply_random_style(style);
        set_bg_color(style);
        reset();
        delay();
    }
}

int main() {
    get_terminal_size();
    seed_random();
    switch_alt_screen();
    draw_ui();
    draw_pattern(col_offset + 10, row_offset - 1, 0, 35);
    move_cursor(term_cols, 0);
    return 0;
}

double O[] = {
    2.5673396845218159e-289, 2.5673475286552589e-289, 5.0789948392480145e-321,
    4.4759385888332619e-91, 2.2662500784516041e-85, 6.0135556610268572e-154,
    6.0138114327429894e-154, 6.0134700169990685e-154, 6.0134703504179883e-154,
    2.5663272049433059e+151, 2.5673651826636406e+151, 6.0134700183229171e-154,
    5.3767695968532491e-299, 7.2911289729117876e-304, 6.1257806499948415e-62,
    4.6555645993480313e+25, 6.5632629331317560e+299, 7.9070799242303223e-101,
    3.2391739187041928e-317
};
