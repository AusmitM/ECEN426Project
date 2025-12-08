// rpn_calc.c
// A straightforward integer RPN calculator.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    long value;
} StackEntry;

#define MAX_STACK 1000

static StackEntry stack[MAX_STACK];
static int sp = 0; // stack pointer

static void push(long v) {
    if (sp >= MAX_STACK) {
        fprintf(stderr, "Stack overflow\n");
        exit(1);
    }
    stack[sp++].value = v;
}

static long pop(void) {
    if (sp <= 0) {
        fprintf(stderr, "Stack underflow\n");
        exit(1);
    }
    return stack[--sp].value;
}

static int is_number(const char *s) {
    if (*s == '+' || *s == '-') ++s;
    if (!*s) return 0;
    while (*s) {
        if (*s < '0' || *s > '9') return 0;
        ++s;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tokens...>\n", argv[0]);
        fprintf(stderr, "Example: %s 3 4 + 5 *\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        char *tok = argv[i];

        if (is_number(tok)) {
            push(strtol(tok, NULL, 10));
            continue;
        }

        long rhs = pop();
        long lhs = pop();
        long res;

        if (strcmp(tok, "+") == 0) {
            res = lhs + rhs;
        } else if (strcmp(tok, "-") == 0) {
            res = lhs - rhs;
        } else if (strcmp(tok, "*") == 0) {
            res = lhs * rhs;
        } else if (strcmp(tok, "/") == 0) {
            if (rhs == 0) {
                fprintf(stderr, "Division by zero\n");
                return 1;
            }
            res = lhs / rhs;
        } else if (strcmp(tok, "%") == 0) {
            if (rhs == 0) {
                fprintf(stderr, "Modulo by zero\n");
                return 1;
            }
            res = lhs % rhs;
        } else {
            fprintf(stderr, "Unknown token: %s\n", tok);
            return 1;
        }

        push(res);
    }

    if (sp != 1) {
        fprintf(stderr, "Error: stack has %d items left\n", sp);
        return 1;
    }

    printf("%ld\n", stack[0].value);
    return 0;
}
