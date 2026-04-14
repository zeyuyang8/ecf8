// clang-format off
#define THREAD_ID           threadIdx.x
#define BLOCK_ID            blockIdx.x
#define N_THREADS           blockDim.x
#define BYTES_PER_THREAD    8
#define BYTES_LOOK_AHEAD    2  // 2 bytes for look-ahead for FP8
#define GAP_BIT_SIZE        4  // 4 bits for gap because 2^4 = 16 bits

typedef unsigned char       uint8_t;
typedef unsigned short      uint16_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;


extern "C"
__global__ void decode(
    const uint8_t* __restrict__  luts,                // (n_luts, 256)
    const uint8_t* __restrict__  encoded,             // (n_bytes,)
    const uint8_t* __restrict__  packed_other_4bits,  // (n_elements / 2,)
    const uint64_t* __restrict__ output_positions,    // one entry per thread block
    const uint8_t* __restrict__  gaps,                // one entry per thread
    uint8_t* __restrict__        outputs,             // (n_elements,)
    const int                    n_luts,
    const int                    n_bytes,
    const int                    n_elements
) {
    // Thread-local register buffer
    const int n_bytes_register = BYTES_LOOK_AHEAD + BYTES_PER_THREAD;  // 10 bytes
    alignas(8) uint8_t register_buffer[n_bytes_register];
    // Shared memory across all threads in block
    extern __shared__ volatile uint8_t shared_mem[];

    // Accumulators for all threads starting from the first 4 bytes in shared memory
    volatile uint32_t* accumulators = (volatile uint32_t*)shared_mem;
    // Write buffer for all threads starting after the accumulators
    volatile uint8_t* write_buffer = (volatile uint8_t*)(shared_mem + N_THREADS * 4 + 4);

    // Global thread ID
    const int global_thread_id = BLOCK_ID * N_THREADS + THREAD_ID;

    // Load data from global memory to thread-local register buffer
    for (int i = 0; i < n_bytes_register; i++) {
        if (global_thread_id * BYTES_PER_THREAD + i < n_bytes) {
            register_buffer[i] = encoded[global_thread_id * BYTES_PER_THREAD + i];
        }
    }
    __syncthreads();

    // Create memory views
    alignas(8) uint8_t buffer[n_bytes_register];

    // On little-endian systems (which is almost always the case for CUDA on x86 and most ARM),
    // the least significant byte of long_buffer is buffer[0], and the most significant byte is buffer[7].
    // so from left (most significant) to right (least significant), it will be byte 7 to 0 of `buffer`
    uint64_t &long_buffer = *reinterpret_cast<uint64_t*>(buffer);
    // Likewise, byte 9 to 8 of `buffer`
    uint16_t &short_buffer = *reinterpret_cast<uint16_t*>(buffer + 8);

    // Gap value indicates bits to skip (already decoded)
    buffer[8] = gaps[global_thread_id * GAP_BIT_SIZE / 8];
    const int bit_position = global_thread_id * GAP_BIT_SIZE % 8;
    const uint8_t gap = (buffer[8] >> (4 - bit_position)) & 0x0f;

    // Decoding to get the output position
    uint32_t thread_counter = 0;

    // Here the order is reversed because of the reinterpret cast of `buffer`
    // Now the left most byte of `long_buffer` is `buffer[7]`, and the right most byte is `buffer[0]`
    // So we let `buffer[7]` be `register_buffer[0]`, and so on for the rest bytes in `buffer`
    // Then now the left most byte of `long_buffer` is `buffer[7]` which is `register_buffer[0]`
    // From left to right in `long_buffer` will be byte 0 to 7 in `register_buffer`
    buffer[0] = register_buffer[7];
    buffer[1] = register_buffer[6];
    buffer[2] = register_buffer[5];
    buffer[3] = register_buffer[4];
    buffer[4] = register_buffer[3];
    buffer[5] = register_buffer[2];
    buffer[6] = register_buffer[1];
    buffer[7] = register_buffer[0];

    long_buffer <<= gap;
    uint8_t free_bits = gap;
    uint8_t decoded;

    while (free_bits < 16) {
        decoded = __ldg(&luts[long_buffer >> 56]);
        if (decoded >= 240) {
            decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)]);
        }
        thread_counter += 1;
        decoded = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    // Likewise, from left to right in `short_buffer` will be byte 8 to 9 in `register_buffer`
    buffer[8] = register_buffer[9];
    buffer[9] = register_buffer[8];

    long_buffer |= static_cast<uint64_t>(short_buffer) << (free_bits - 16);
    free_bits -= 16;

    while (2 + free_bits / 8 < BYTES_PER_THREAD) {
        decoded = __ldg(&luts[long_buffer >> 56]);
        if (decoded >= 240) {
            decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)]);
        }
        thread_counter += 1;
        decoded = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    // Efficiently do the cumulative sum of the thread counters starts here
    if (THREAD_ID == 0) {
        accumulators[0] = output_positions[BLOCK_ID] + thread_counter;
    } else {
        accumulators[THREAD_ID] = thread_counter;
    }
    __syncthreads();

    int i;
    for (i = 2; i <= N_THREADS; i <<= 1) {
        if (((THREAD_ID + 1) & (i - 1)) == 0) {
            accumulators[THREAD_ID] += accumulators[THREAD_ID - (i >> 1)];
        }
        __syncthreads();
    }

    if (THREAD_ID == 0) {
        accumulators[N_THREADS - 1] = 0;
    }
    __syncthreads();

    for (i = N_THREADS; i >= 2; i >>= 1) {
        if (((THREAD_ID + 1) & (i - 1)) == 0) {
            accumulators[THREAD_ID] += accumulators[THREAD_ID - (i >> 1)];
            accumulators[THREAD_ID - (i >> 1)] = accumulators[THREAD_ID] - accumulators[THREAD_ID - (i >> 1)];
        }
        __syncthreads();
    }

    if (THREAD_ID == 0) {
        accumulators[0] = output_positions[BLOCK_ID];
        accumulators[N_THREADS] = output_positions[BLOCK_ID + 1];
    }
    __syncthreads();
    // Cumulative sum ends here

    // Get the output position of the current thread block
    uint32_t output_idx = accumulators[THREAD_ID], write_offset = accumulators[0];
    const uint32_t end_output_idx = min(output_idx + thread_counter, n_elements);

    // Now we have known the output position of the current thread block and the current thread
    // Do the decoding again and write the result to the write buffer in shared memory
    buffer[0] = register_buffer[7];
    buffer[1] = register_buffer[6];
    buffer[2] = register_buffer[5];
    buffer[3] = register_buffer[4];
    buffer[4] = register_buffer[3];
    buffer[5] = register_buffer[2];
    buffer[6] = register_buffer[1];
    buffer[7] = register_buffer[0];

    long_buffer <<= gap;
    free_bits = gap;

    while (free_bits < 16 && output_idx < end_output_idx) {
        decoded = __ldg(&luts[long_buffer >> 56]);
        if (decoded >= 240) {
            decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)]);
        }
        // Only want the leftmost 4 bits which is sign and mantissa
        buffer[8] = packed_other_4bits[output_idx / 2] << (output_idx % 2 * 4);
        buffer[8] = decoded << 3 | (buffer[8] & 0x80) | (buffer[8] >> 4 & 0x7);
        write_buffer[output_idx - write_offset] = buffer[8];

        output_idx += 1;
        decoded = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buffer <<= decoded;
        free_bits += decoded;
    }

    buffer[8] = register_buffer[9];
    buffer[9] = register_buffer[8];

    long_buffer |= static_cast<uint64_t>(short_buffer) << (free_bits - 16);
    free_bits -= 16;

    while (output_idx < end_output_idx) {
        decoded = __ldg(&luts[long_buffer >> 56]);
        if (decoded >= 240) {
            decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buffer >> 48) & 0xff)]);
        }

        buffer[8] = packed_other_4bits[output_idx / 2] << (output_idx % 2 * 4);
        buffer[8] = decoded << 3 | (buffer[8] & 0x80) | (buffer[8] >> 4 & 0x7);
        write_buffer[output_idx - write_offset] = buffer[8];

        output_idx += 1;
        decoded = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buffer <<= decoded;
        free_bits += decoded;
    }
    __syncthreads();

    // Write the decoding result from shared memory to global memory
    for (i = THREAD_ID; i < min(accumulators[N_THREADS] - write_offset, n_elements - write_offset); i += N_THREADS) {
        outputs[i + write_offset] = write_buffer[i];
    }
}
