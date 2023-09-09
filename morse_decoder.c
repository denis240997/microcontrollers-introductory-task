#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define CARRIER_FREQUENCY_HZ 800
#define DOT_DURATION_MS 50

#define AVERAGE_WINDOW_SIZE_PERIODS 8

#define POINTS_PER_DOT 2

#define DASH_THRESHOLD_MS 95
#define SYMBOL_THRESHOLD_MS 100
#define SPACER_THRESHOLD_MS 250

#define MAX_CODE_LENGTH 6

#define MORSE_TABLE_FILEPATH "morse_table.txt"
#define MORSE_TABLE_LENGTH 114


double *read_signal_from_file(char *file_name, int *sample_size, int *sampling_rate)
{
    FILE *input_file;
    input_file = fopen(file_name, "r");

    fscanf(input_file, "%d %d", sample_size, sampling_rate);

    double *signal = malloc(*sample_size * sizeof(double));

    for (int i = 0; i < *sample_size; i++)
    {
        fscanf(input_file, "%le", &signal[i]);
    }

    fclose(input_file);

    return signal;
}


bool *digitalize_signal(double *signal, int sample_size, int sampling_rate, int *digital_signal_size)
{
    // Calculate averaging window size to capture AVERAGE_WINDOW_SIZE_PERIODS periods of carrier frequency
    int averaging_window_size = AVERAGE_WINDOW_SIZE_PERIODS * sampling_rate / CARRIER_FREQUENCY_HZ;

    // Averaged signal will be shorter than original one
    int averaged_signal_size = sample_size - averaging_window_size + 1;

    double *averaged_signal = malloc(averaged_signal_size * sizeof(double));

    // Calculate averaged signal using linear algorithm
    double sum = 0;
    for (int i = 0; i < averaging_window_size; i++)
    {
        sum += fabs(signal[i]);
    }
    averaged_signal[0] = sum / averaging_window_size;

    for (int i = 1; i < averaged_signal_size; i++)
    {
        sum += fabs(signal[i + averaging_window_size - 1]) - fabs(signal[i - 1]);
        averaged_signal[i] = sum / averaging_window_size;
    }

    // Threshold value is mean between min and max values of averaged signal
    // TODO: calculate min and max values online during averaging
    double min = averaged_signal[0];
    double max = averaged_signal[0];
    for (int i = 1; i < averaged_signal_size; i++)
    {
        if (averaged_signal[i] < min)
        {
            min = averaged_signal[i];
        }
        if (averaged_signal[i] > max)
        {
            max = averaged_signal[i];
        }
    }

    double threshold = (min + max) / 2;

    // Digitalize averaged signal using threshold
    *digital_signal_size = averaged_signal_size;
    bool *digitalized_signal = malloc(*digital_signal_size * sizeof(bool));

    for (int i = 0; i < *digital_signal_size; i++)
    {
        if (averaged_signal[i] > threshold)
        {
            digitalized_signal[i] = true;
        }
        else
        {
            digitalized_signal[i] = false;
        }
    }

    free(averaged_signal);

    return digitalized_signal;
}


bool *compress_digital_signal(bool *digital_signal, int digital_signal_size, int sampling_rate, int *compressed_signal_size)
{
    double min_points_per_dot = POINTS_PER_DOT;
    double dot_duration_ms = DOT_DURATION_MS;

    double min_freq = min_points_per_dot * 1000 / dot_duration_ms;
    int step = (int) (sampling_rate / min_freq);

    *compressed_signal_size = (int) (digital_signal_size / step);

    bool *compressed_signal = malloc(*compressed_signal_size * sizeof(bool));

    for (int i = 0; i < *compressed_signal_size; i++)
    {
        compressed_signal[i] = digital_signal[i * step];
    }

    return compressed_signal;
}


double compute_interval_duration_ms(int sample_size, int sampling_rate, int compressed_signal_size)
{
    return (double) 1000 * sample_size / sampling_rate / compressed_signal_size;
}


char *recognize_signal(bool *compressed_signal, int compressed_signal_size, double interval_duration_ms, int *final_code_length)
{
    int max_code_length = (int) (2 * interval_duration_ms * compressed_signal_size / DOT_DURATION_MS);
    char *code = malloc(max_code_length * sizeof(char));

    int code_length = 0;

    int peak_start = 0, spacer_start = -1;
    bool state = false;

    for (int i = 0; i < compressed_signal_size; i++)
    {
        bool point = compressed_signal[i];

        // rising edge
        if (!state && point)
        {
            peak_start = i;
            if (spacer_start != -1)
            {
                double spacer_duration = (i - spacer_start) * interval_duration_ms;
                                
                // end of symbol
                if (spacer_duration > SYMBOL_THRESHOLD_MS)
                {
                    code[code_length] = ' ';
                    code_length++;

                    // end of word
                    if (spacer_duration > SPACER_THRESHOLD_MS)
                    {
                        code[code_length] = '#';
                        code_length++;
                    }
                }
            }
        }

        // falling edge
        if (state && !point)
        {
            spacer_start = i;
            double peak_duration = (i - peak_start) * interval_duration_ms;
            code[code_length] = (peak_duration > DASH_THRESHOLD_MS) ? '-' : '.';
            code_length++;
        }

        state = point;
    }

    code[code_length] = '\0';
    *final_code_length = code_length - 1;

    return code;
}


// The Morse code is represented as a sequence of zeros (dots) and ones (dashes) with a leading one, 
// this binary record is converted to decimal with a shift of 2 (so that the indexes start from zero)
int morse_code_to_decimal(char *morse_code)
{
    bool *binary = malloc(sizeof(bool) * (MAX_CODE_LENGTH + 1));
    binary[0] = true;

    int i = 0;
    while (morse_code[i] != '\0')
    {
        i++;
        binary[i] = (morse_code[i-1] == '-') ? true : false;
    }

    int decimal = 0;
    int n_digits = i;
    while (i >= 0)
    {
        decimal += binary[i] * (1 << (n_digits - i));
        i--;
    }

    free(binary);
    
    return decimal - 2;
}


char *get_morse_table()
{
    FILE *morse_table_file;
    morse_table_file = fopen(MORSE_TABLE_FILEPATH, "r");

    char *morse_table = malloc(sizeof(char) * MORSE_TABLE_LENGTH + 1);

    int i = 0;
    char c;
    while (fscanf(morse_table_file, "%c", &c) != EOF)
    {
        if (c != '\n')
        {
            morse_table[i] = c;
            i++;
        }
    }
    morse_table[i] = '\0';

    fclose(morse_table_file);

    return morse_table;
}


char *decode_message(char *code, int code_length, char *morse_table)
{
    char *message = malloc(sizeof(char) * code_length);
    char *symbol = malloc(sizeof(char) * (MAX_CODE_LENGTH + 1));
    int i = 0, j = 0, k = 0;
    while (code[i] != '\0')
    {
        if (code[i] == ' ')
        {
            symbol[k] = '\0';
            message[j] = morse_table[morse_code_to_decimal(symbol)];
            k = 0;
            j++;
        }
        else if (code[i] == '#')
        {
            message[j] = ' ';
            j++;
        }
        else
        {
            symbol[k] = code[i];
            k++;
        }
        i++;
    }

    if (k > 0)
    {
        symbol[k] = '\0';
        message[j] = morse_table[morse_code_to_decimal(symbol)];
        j++;
    }

    message[j] = '\0';
    free(symbol);

    return message;
}


int main(int argc, char *argv[])
{
    // argv[1] - input file name where first line is two numbers: sample_size and sampling_rate. Rest of the file is a signal values

    int sample_size;
    int sampling_rate;

    double *signal = read_signal_from_file(argv[1], &sample_size, &sampling_rate);

    int digital_signal_size;
    bool *digital_signal = digitalize_signal(signal, sample_size, sampling_rate, &digital_signal_size);
    free(signal);

    int compressed_signal_size;
    bool *compressed_signal = compress_digital_signal(digital_signal, digital_signal_size, sampling_rate, &compressed_signal_size);
    free(digital_signal);

    double interval_duration_ms = compute_interval_duration_ms(sample_size, sampling_rate, compressed_signal_size);

    int code_length;
    char *code = recognize_signal(compressed_signal, compressed_signal_size, interval_duration_ms, &code_length);
    free(compressed_signal);

    char *morse_table = get_morse_table();

    char *message = decode_message(code, code_length, morse_table);
    printf("%s\n", message);

    free(message);
    free(code);
    free(morse_table);

    return 0;
}
