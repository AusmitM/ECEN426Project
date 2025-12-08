#include <stdio.h>

#define STACK_SIZE 1000
#define BIG long int

// Represents a typed data unit: either data (b=1) or operator (b=0)
typedef struct {
    int is_data;
    BIG value;
} Element;

Element stack[STACK_SIZE];

#define PUSH(x) stack[stack_top++] = (Element){.is_data = 1, .value = (x)}
#define PUSH_OP(x) stack[stack_top++] = (Element){.is_data = 0, .value = (x)}

BIG temp_val = 0, temp_n = 0, mult_factor = 1;
BIG i = 0, stack_top = 0, read_pos = STACK_SIZE;

// Encodes a string into a hash-like int (probably used for instruction decoding)
int encode_token(char* str) {
    BIG result = 1;
    int len = 0;
    while (str[len]) {
        result >>= 1;
        result ^= str[len++];
    }
    return (result & 15) | ((len ^ *str) << 4 & 240);
}

// Entry point
int main(int argc, char** argv) {
    mult_factor = 1;

    // Interpret command line arguments
    while (*++argv) {
        char* arg = *argv;
        int code = encode_token(arg);

        // Token classification logic
        switch (code) {
            case 16: break;  // probably a separator or noop
            case 90: temp_val += temp_n * mult_factor * STACK_SIZE * STACK_SIZE; temp_n = 0; break;
            case 202: temp_val += temp_n * mult_factor * STACK_SIZE * STACK_SIZE * STACK_SIZE; temp_n = 0; break;
            case 192: temp_val += temp_n * mult_factor * STACK_SIZE; temp_n = 0; break;
            case 170: temp_val += temp_n * mult_factor * STACK_SIZE; temp_n = 0; break;
            case 232: temp_val += temp_n * mult_factor; stack[i++] = (Element){1, temp_val}; mult_factor = 1; temp_val = temp_n = 0; break;
            case 76: temp_val += temp_n * mult_factor; stack[i++] = (Element){1, temp_val}; mult_factor = 1; temp_val = temp_n = 0; break;
            case 65: temp_n += 70; break;
            case 103: temp_n += 60; break;
            case 2: temp_n += 8; break;
            case 17: temp_n += 3; break;
            case 39: temp_n += 30; break;
            case 201: temp_n += 1; break;
            case 242: temp_n *= 99; break;
            case 78: temp_n += 16; break;
            case 130: temp_n += 90; break;
            case 175: temp_n += 17; break;
            case 49: temp_n += 50; break;
            case 55: temp_val += temp_n * mult_factor; stack[i++] = (Element){1, temp_val}; mult_factor = 1; temp_val = temp_n = 0; break;
            case 0: temp_n += 6; break;
            case 47: temp_n += 4; break;
            case 206: temp_n += 13; break;
            case 238: temp_n += 14; break;
            case 45: temp_n += 12; break;
            case 111: temp_n += 19; break;
            case 56: temp_n += 11; break;
            case 113: temp_n += 10; break;
            case 40: temp_n += 5; break;
            case 121: temp_n += 2; break;
            case 165: temp_n += 9; break;
            case 31: temp_n += 15; break;
            case 223: temp_n += 18; break;
            case 52: temp_n += 40; break;
            case 106: temp_n += 7; break;
            case 110: mult_factor = -1; break;
            case 137: temp_val += temp_n * mult_factor; stack[i++] = (Element){1, temp_val}; mult_factor = 1; temp_val = temp_n = 0; break;
            default: break;
        }
    }

    // Push trailing value if needed
    temp_val += temp_n * mult_factor;
    stack[i++] = (Element){1, temp_val};

    // Evaluation phase
    for (; read_pos != STACK_SIZE;) {
        stack[i++] = stack[read_pos++];
    }

    // Execution loop
    BIG a, b, op;
    while (i != 0) {
        if (stack[i - 1].is_data) {
            stack[--read_pos] = stack[--i];
        } else {
            b = stack[--read_pos].value;
            a = stack[--read_pos].value;
            op = stack[--i].value;
            switch (op) {
                case 1: stack[--read_pos] = (Element){1, a + b}; break;
                case 2: stack[--read_pos] = (Element){1, a - b}; break;
                case 5: stack[--read_pos] = (Element){1, a * b}; break;
                case 6: stack[--read_pos] = (Element){1, b / a}; break;
                default: stack[--read_pos] = (Element){1, b % a}; break;
            }
        }
    }

    printf("%ld\n", stack[read_pos].value);
    return 0;
}
