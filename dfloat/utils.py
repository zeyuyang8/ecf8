import re
from copy import copy
from sys import stderr

import numpy as np
import torch
from dahuffman import HuffmanCodec
from tqdm import tqdm


def param_to_bytes(param):
    if param.dtype in [torch.uint8, torch.int8, torch.float8_e4m3fn]:
        return param.numel()
    elif param.dtype in [torch.float16, torch.bfloat16, torch.int16, torch.uint16]:
        return param.numel() * 2
    elif param.dtype in [torch.float32, torch.int32, torch.uint32]:
        return param.numel() * 4
    elif param.dtype in [torch.float64, torch.int64, torch.uint64]:
        return param.numel() * 8
    else:
        raise ValueError(f"Unrecognized parameter data type {param.dtype}.")


def report_model_size(model, include_pattern: list[str] | dict[str, list[str]] | None):
    include_bytes = 0
    exclude_bytes = 0

    # Handle different input types
    if isinstance(include_pattern, list):
        # Original behavior: list of regex patterns
        for param_name, param in model.state_dict().items():
            if any(re.search(pattern, param_name) for pattern in include_pattern):
                include_bytes += param_to_bytes(param)
            else:
                exclude_bytes += param_to_bytes(param)
    elif isinstance(include_pattern, dict):
        # New behavior: dict with regex patterns as keys and attribute lists as values
        for param_name, param in model.state_dict().items():
            included = False
            for pattern, attributes in include_pattern.items():
                if re.search(pattern, param_name):
                    # Check if the parameter name matches any of the specified attributes
                    for attr in attributes:
                        if attr in param_name or param_name.endswith(attr):
                            include_bytes += param_to_bytes(param)
                            included = True
                            break
                    if included:
                        break
            if not included:
                exclude_bytes += param_to_bytes(param)
    elif include_pattern is None:
        for param_name, param in model.state_dict().items():
            include_bytes += param_to_bytes(param)
    else:
        raise ValueError(
            "`include_pattern` must be either a list of strings or a dictionary, or None"
        )

    model_bytes = include_bytes + exclude_bytes
    include_gb = include_bytes / (1024**3)
    exclude_gb = exclude_bytes / (1024**3)
    model_gb = model_bytes / (1024**3)
    print(f"Included model size: {include_gb:0.4f} GB", file=stderr)
    print(f"Excluded model size: {exclude_gb:0.4f} GB", file=stderr)
    print(f"Total model size: {model_gb:0.4f} GB", file=stderr)
    return include_gb, exclude_gb, model_gb


def repo_id_to_model_id(repo_id: str) -> str:
    temp = repo_id.split("/")
    return "models--" + "--".join(temp)


def locate_block_by_regex(model, pattern):
    matched_blocks = {}
    for full_name, sub_module in model.named_modules():
        if re.fullmatch(pattern, full_name):
            matched_blocks[full_name] = sub_module
    return matched_blocks


def find_layers_by_selection(module, select: list[str], name=""):
    # `name` 'text_encoder_2.encoder.block.0'
    # `selected` ['layer.0.SelfAttention.k', 'layer.0.SelfAttention.o']
    # Want to return a dictionary {name: module} where `name` is the full name of the selected layer
    res = {}

    # Check if current module matches any selection pattern
    for pattern in select:
        # If name is empty, we're at the root, so just use the pattern
        # Otherwise, combine the base name with the pattern
        if name == "":
            full_pattern = pattern
        else:
            full_pattern = name + "." + pattern

        # Try to get the module at this path
        try:
            target_module = module
            parts = pattern.split(".")
            for part in parts:
                target_module = getattr(target_module, part)

            # If we successfully found the module, add it to results
            res[full_pattern] = target_module
        except AttributeError:
            # Pattern doesn't match, continue to next pattern
            continue

    # Recursively search children
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name != "" else name1
        res.update(find_layers_by_selection(child, select, name=child_name))

    # Sort the keys alphabetically
    return dict(sorted(res.items()))


def find_layers_by_layer_type(module, layers=[torch.nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers_by_layer_type(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )

    # Sort the keys alphabetically
    return dict(sorted(res.items()))


def find_layers_by_data_type(module, target_dtype=torch.bfloat16, name=""):
    # Check if module has weights and if they match target dtype
    if (
        hasattr(module, "weight")
        and hasattr(module.weight, "dtype")
        and module.weight.dtype == target_dtype
    ):
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers_by_data_type(
                child,
                target_dtype=target_dtype,
                name=name + "." + name1 if name != "" else name1,
            )
        )

    # Sort the keys alphabetically
    return dict(sorted(res.items()))


def find_layers_by_data_type_and_layer_type(
    module,
    target_dtype=torch.bfloat16,
    layers=[torch.nn.Linear, torch.nn.Embedding],
    name="",
):
    # Check if module has weights and if they match target dtype
    if (
        type(module) in layers
        and hasattr(module, "weight")
        and hasattr(module.weight, "dtype")
        and module.weight.dtype == target_dtype
    ):
        return {name: module}

    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers_by_data_type_and_layer_type(
                child,
                target_dtype=target_dtype,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )

    # Sort the keys alphabetically
    return dict(sorted(res.items()))


def flatten_block_of_weights(layers, dtype=torch.float8_e4m3fn):
    numel_by_block = []
    total_numel = 0
    for sub_module in layers.values():
        numel_by_block.append(sub_module.weight.numel())
        total_numel += sub_module.weight.numel()

    flat_weights = torch.empty(total_numel, dtype=dtype)
    offset = 0
    for sub_module in layers.values():
        w = sub_module.weight.data.detach().flatten()
        flat_weights[offset : offset + w.numel()] = w
        offset += w.numel()

    split_positions = torch.cumsum(torch.LongTensor(numel_by_block), dim=0)[:-1]

    return flat_weights, split_positions


def get_exponents_and_other_4bits_fp8_e4m3(data: torch.Tensor):
    assert data.ndim == 1
    assert data.dtype == torch.float8_e4m3fn

    # First view the fp8 weights as a uint8 tensor
    W = data.view(torch.uint8)

    # Shift right by 3 bits to get the exponent bits
    # `& 0xF` masks to keep only the 4 exponent bits (0xF is binary 1111)
    Wexp = (W >> 3) & 0xF

    # Other 4 bits
    # `& 0x7` masks to keep only the rightmost 3 bits (0x7 is binary 0111)
    # `>> 4` shifts right by 4 bits and `& 0x8` masks to keep only the leftmost 1 bit (0x8 is binary 1000)
    # `|` combines the two results
    Wother_4bits = ((W >> 4) & 0x8 | (W & 0x7)).to(torch.uint8)

    paired = Wother_4bits.view(-1, 2)
    packed_other_4bits = (paired[:, 0] << 4) | paired[:, 1]

    return Wexp, packed_other_4bits


def compute_shannon_entropy(data: torch.Tensor):
    assert data.ndim == 1
    vals, freqs = torch.unique(data, return_counts=True)
    probabilities = freqs / freqs.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities))

    return entropy.item()


def get_nbit_codec(counter, max_n_bits=16):
    codec = HuffmanCodec.from_frequencies(counter)
    table = codec.get_code_table()
    max_len = 0
    for _, (length, _) in table.items():
        max_len = max(max_len, length)

    compressed_codec = codec
    compressed_counter = counter

    min_k = 2
    freq = np.array(list(counter.values()))
    while max_len > max_n_bits:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = np.array(list(counter.keys()))[min_indices]

        compressed_counter = copy(counter)
        for k in min_keys:
            compressed_counter[k] = 1
        compressed_codec = HuffmanCodec.from_frequencies(compressed_counter)
        table = compressed_codec.get_code_table()
        max_len = 0
        for _, (length, _) in table.items():
            max_len = max(max_len, length)

        print(min_k - 1, max_len)

    return compressed_codec, compressed_counter, table


def get_luts_from_huffman_table(table):
    prefixes = [""]

    for symbol, (bits, decimal_val) in table.items():
        # Ignore the `_EOF` symbol
        if not isinstance(symbol, int):
            continue

        # `rjust` is used to pad 0s to the left of the string to make the string the 8-bit long
        binary_val = bin(decimal_val)[2:].rjust(bits, "0")

        # The prefix is n bytes long where n = (bits - 1) // 8
        prefix_len = (bits - 1) - (bits - 1) % 8
        prefix = binary_val[:prefix_len]
        # print(f'bits: {bits:3d} | binary: {bin(decimal_val):>15} | decimal: {decimal_val:4d} | padded binary: {binary_val:>15} | prefix: {prefix:>8}')

        if prefix not in prefixes:
            prefixes.append(prefix)

    prefixes.sort(key=len)
    # print()
    # print(f'prefixes: {prefixes}')
    # print()

    # Initialize a lookup table where each row corresponds to a prefix
    luts = np.zeros((len(prefixes), 2**8), dtype=np.uint8)

    # Iterate over each prefix to fill the lookup table
    for prefix_idx, prefix in enumerate(prefixes):
        bytes_dict = {}
        prefix_n_bytes = len(prefix) // 8
        prefix_n_bits = prefix_n_bytes * 8

        # Iterate over each code in the Huffman tree
        for symbol, (bits, decimal_val) in table.items():
            # Ignore the `_EOF` symbol
            if not isinstance(symbol, int):
                continue

            # If the code starts with the current prefix, then there are two cases
            binary_val = bin(decimal_val)[2:].rjust(bits, "0")
            if binary_val.startswith(prefix):
                # Case 1: When the first `prefix_n_bytes` bits of the code are the same as the prefix
                # time to map the code to the symbol
                if (bits - 1) // 8 == prefix_n_bytes:
                    # The suffix is the remaining bits after the prefix
                    # `ljust` is used to pad 0s to the right of the string to make the string the 8-bit long
                    suffix = binary_val[prefix_n_bits:]
                    padded_suffix = suffix.ljust(8, "0")
                    padded_suffix_int = int(padded_suffix, 2)

                    dict_key = padded_suffix_int
                    dict_value = symbol

                    # print(f'prefix: {prefix:>8} | binary: {binary_val:>17} | suffix: {suffix:>8} | padded suffix: {padded_suffix:>8} | padded suffix int: {padded_suffix_int:>3d}')

                # Case 2: When the current prefix is a subset of the first n bytes of the code, and that n > `prefix_n_bytes`
                # map the next byte in the code to the longer prefix
                else:
                    assert (bits - 1) // 8 > prefix_n_bytes
                    next_byte = binary_val[prefix_n_bits : prefix_n_bits + 8]
                    next_byte_int = int(next_byte, 2)

                    # NOTE: due to the nature of the Huffman coding, if the current prefix is a subset of the first n bytes of the code,
                    # then the current prefix including the next byte MUST be in the prefixes list
                    prefix_and_next_byte = binary_val[: prefix_n_bits + 8]
                    prefix_next_byte_index = prefixes.index(prefix_and_next_byte)

                    dict_key = next_byte_int
                    dict_value = 2**8 - prefix_next_byte_index

                    # print(f'index: {prefix_next_byte_index}')
                    # print(f'prefix: {prefix:>8} | binary: {binary_val:>17} | next byte: {next_byte:>8} | next byte int: {next_byte_int:>3d}')

                # Make sure that the mappings in case 1 and case 2 have no duplicates
                if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                    raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                else:
                    bytes_dict[dict_key] = dict_value

        # print(bytes_dict)

        curr_val = 0
        for byte_node_int in range(2**8):
            # `byte_node_int` could be either a suffix or a byte in the prefix
            if byte_node_int in bytes_dict:
                curr_val = bytes_dict[byte_node_int]
            luts[prefix_idx, byte_node_int] = curr_val

    # The last row of `luts` is the length of binary code for each symbol
    lens = np.zeros((1, 2**8), dtype=np.uint8)
    for symbol, (bits, decimal_val) in table.items():
        if isinstance(symbol, int):
            lens[-1, symbol] = bits

    # print(lens)

    # In summary, `luts` is a 2D array of shape `(len(prefixes) + 1, 256)`
    # Each row corresponds to a prefix, and the last row corresponds to the length of binary code for each possible symbol
    luts = torch.from_numpy(np.concatenate((luts, lens), axis=0))
    return luts


def _encode_exponents_for_cuda(
    exponents_lst, codec, bytes_per_thread=8, threads_per_block=512
):
    bits_per_thread = 8 * bytes_per_thread
    bits_per_block = bits_per_thread * threads_per_block
    encoded = []
    gaps = []
    output_positions = []

    # Python int is 64 bits
    buffer = 0
    size = 0
    total_size = 0
    element_count = 0
    for symbol in tqdm(exponents_lst, leave=False):
        # Record the gaps for each thread, where gaps are the number of unaligned bits coming from the previous thread
        # `total_size // bits_per_thread` is the number of threads that have already encoded data
        # `+ 1` is to account for the current thread
        # So `total_size // bits_per_thread + 1` is the index of the current thread
        if total_size // bits_per_thread + 1 > len(gaps):
            # `gap` represents the number of unaligned bits coming from the previous thread
            # See the example below (if `bits_per_thread` is 8):
            # total size: 0 | gap: 0
            # total size: 8 | gap: 0
            # total size: 17 | gap: 1
            # total size: 24 | gap: 0
            gap = total_size % bits_per_thread
            gaps.append(gap)
            # print(f'total size: {total_size} | gap: {gap}')

        # Record the output positions for each block, where output positions are the number of elements that have been encoded,
        # so that during decoding, we can know where to write the decoded data
        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)

        bits, decimal_val = codec._table[symbol]
        # print(f'symbol: {symbol} | bits: {bits} | code int: {decimal_val}')

        # Shift code bits for the current symbol into the buffer at the rightmost position
        buffer = (buffer << bits) + decimal_val
        # print(f'buffer binary: {bin(buffer)}')

        # Update the size of the buffer and the number of encoded elements
        size += bits
        total_size += bits
        element_count += 1

        # When the buffer is larger than 8 bits, extract the leftmost 8 bits
        # from the buffer and write it to the encoded list
        while size >= 8:
            # Extract the leftmost 8 bits from the buffer
            byte = buffer >> (size - 8)
            # print(f'byte binary: {bin(byte)}')

            # Write the extracted byte to the encoded list
            encoded.append(byte)

            # Remove the leftmost 8 bits from the buffer
            buffer = buffer - (byte << (size - 8))
            size -= 8

    # Handling of the final sub-byte chunk
    # The encoded bit stream may not align with byte boundaries at the end
    # We need an EOF marker to prevent the decoder from interpreting the remaining bits as valid data
    # However, we only need to encode up to the current byte boundary
    # since the decoder knows it's the end of data, saving us from creating a new byte for the `_EOF` symbol
    if size > 0:
        if total_size // (8 * bytes_per_thread) + 1 > len(gaps):
            gap = total_size % bits_per_thread
            gaps.append(gap)

        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)

        bit, decimal_val = codec._table[codec._eof]
        # Shift the EOF bits into the buffer (including the remaining bits) at the rightmost position
        buffer = (buffer << bit) + decimal_val
        size += bit
        # If the buffer is larger than 8 bits, extract the leftmost 8 bits from the buffer
        # The leftmost 8 bits are the remaining bits for the encoded data and some bits of the EOF symbol
        if size >= 8:
            byte = buffer >> (size - 8)
        # If the buffer is smaller than 8 bits, shift the remaining bits to the left
        # But actually this case should never happen because the bit length of the EOF symbol is quite large in most cases
        else:
            byte = buffer << (8 - size)
        encoded.append(byte)

    # The last position marks the total length of the input data
    # Helps locate the output positions of the last block
    output_positions.append(len(exponents_lst))

    # Compute the number of blocks needed to decode the encoded data
    # Total number of bytes divided by the number of bytes per block
    # `np.ceil` is used to round up to the nearest integer
    # As a result, some threads in the last block could not be used
    # and those unused threads do not have gaps marked since we only mark gaps for threads that encode data
    n_bytes_total = len(encoded)
    blocks_per_grid = int(
        np.ceil(n_bytes_total / (threads_per_block * bytes_per_thread))
    )

    # Pads the gaps array with 0s to ensure it has the correct length for all threads
    n_used_threads = len(gaps)
    n_unused_threads = blocks_per_grid * threads_per_block - n_used_threads
    gaps.extend([0] * n_unused_threads)
    # The maximum possible number of bits for a symbol is 16 which is 2 ** 4
    # So the maximum number of bits for a gap is 4,
    # and it happens when a 16-bit code placed at the end of a thread
    binary_str_gaps = [format(gap, "04b") for gap in gaps]
    binary_gaps = [int(bit) for binary in binary_str_gaps for bit in binary]

    # Convert the encoded data to a numpy array
    # `bytes(encoded)` converts the list of bytes to a bytes object
    # `np.frombuffer` converts the bytes object to a numpy array,
    # Use `np.array(encoded, dtype=np.uint8)` would also work,
    # but it creates a new array by copying the data, converts each element individually
    # so it has more memory overhead and is slower for large data
    np_encoded = np.frombuffer(bytes(encoded), dtype=np.uint8)

    # `np.packbits` packs the binary gaps into a single array
    packed_binary_gaps = np.packbits(binary_gaps)

    # Convert the output positions to a numpy array
    output_positions = np.array(output_positions, dtype=np.uint64)

    return np_encoded, packed_binary_gaps, output_positions


def encode_exponents_for_cuda(
    exponents_4bits,
    codec,
    bytes_per_thread=8,
    threads_per_block=512,
):
    # NumPy arrays are optimized for vectorized operations, not iteration
    # Each iteration requires Python to create a new Python object for each element
    # The overhead of converting from NumPy's internal representation to Python objects is significant
    # That's why we convert the NumPy array to a Python list
    # Get the encoded exponents, gaps, and output positions,
    # and convert them to torch tensors
    encoded_exponents, gaps, output_positions = _encode_exponents_for_cuda(
        exponents_4bits.tolist(),
        codec,
        bytes_per_thread,
        threads_per_block,
    )

    import warnings

    # Suppress the warning about the NumPy array being not writable
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        encoded_exponents, gaps, output_positions = list(
            map(torch.from_numpy, (encoded_exponents, gaps, output_positions))
        )

    return encoded_exponents, gaps, output_positions
