#include <stdint.h>

#include "../../model/exported/resnet18/resnet18_graph.h"
#include "../../model/exported/resnet18/resnet18_quant_params.h"
#include "../kernels/qgemm.h"

#ifndef RESNET18_EXECUTION_MODE
#define RESNET18_EXECUTION_MODE 0
#endif

#if defined(__riscv) && (__riscv_xlen == 64)
#include "../kernels/xq_ops.h"
#endif

static int8_t requantize_channel(uint32_t op, uint32_t channel,
                                 int64_t accumulator);
static int32_t weight_sum(uint32_t op, uint32_t channel);

static int8_t requantize_fixed(int64_t accumulator, int32_t multiplier,
                               uint8_t shift, int8_t zero_point)
{
#if defined(__riscv) && (__riscv_xlen == 64)
    if (RESNET18_EXECUTION_MODE == 3) {
        uint64_t parameters = xq_pack_requant_params(
            multiplier, shift, zero_point);
        return (int8_t)xqrequant8_emit(accumulator, parameters);
    }
#endif
    return requantize_reference(accumulator, multiplier, shift, zero_point);
}

static int32_t round_divide(int64_t value, int64_t divisor)
{
    int64_t magnitude = value < 0 ? -value : value;
    int64_t rounded = (magnitude + divisor / 2) / divisor;
    return (int32_t)(value < 0 ? -rounded : rounded);
}

static int8_t clamp_int8_value(int32_t value)
{
    if (value > 127)
        return 127;
    if (value < -128)
        return -128;
    return (int8_t)value;
}

static __attribute__((noinline)) void fully_connected_scalar(uint32_t op, const int8_t *input,
                                   const int8_t *weights, const int32_t *bias,
                                   int8_t *output, uint32_t input_size,
                                   uint32_t output_size, int32_t input_zp,
                                   int32_t output_zp)
{
    uint32_t oc;
    for (oc = 0; oc < output_size; ++oc) {
        int64_t accumulator = bias[oc];
        uint32_t i;
        for (i = 0; i < input_size; ++i)
            accumulator += (int64_t)input[i] * weights[oc * input_size + i];
        accumulator -= (int64_t)input_zp * weight_sum(op, oc);
        output[oc] = requantize_channel(op, oc, accumulator);
        if (resnet18_operators[op].fused_activation == 1 &&
            output[oc] < output_zp)
            output[oc] = (int8_t)output_zp;
    }
}

static int8_t requantize_channel(uint32_t op, uint32_t channel,
                                 int64_t accumulator)
{
#define REQUANTIZE_CASE(n)                                                   \
    case n:                                                                   \
        if (resnet18_op_##n##_use_scalar_fallback[channel]) {                \
            return requantize_scalar_wide(                                    \
                accumulator, resnet18_op_##n##_fallback_multipliers[channel], \
                resnet18_op_##n##_fallback_shifts[channel],                  \
                resnet18_op_##n##_output_zero_point);                        \
        }                                                                      \
        return requantize_fixed(                                               \
            accumulator, resnet18_op_##n##_multipliers[channel],             \
            resnet18_op_##n##_shifts[channel],                               \
            resnet18_op_##n##_output_zero_point)

    switch (op) {
        REQUANTIZE_CASE(2);
        REQUANTIZE_CASE(5);
        REQUANTIZE_CASE(6);
        REQUANTIZE_CASE(8);
        REQUANTIZE_CASE(9);
        REQUANTIZE_CASE(12);
        REQUANTIZE_CASE(13);
        REQUANTIZE_CASE(14);
        REQUANTIZE_CASE(16);
        REQUANTIZE_CASE(17);
        REQUANTIZE_CASE(20);
        REQUANTIZE_CASE(21);
        REQUANTIZE_CASE(22);
        REQUANTIZE_CASE(24);
        REQUANTIZE_CASE(25);
        REQUANTIZE_CASE(28);
        REQUANTIZE_CASE(29);
        REQUANTIZE_CASE(30);
        REQUANTIZE_CASE(32);
        REQUANTIZE_CASE(33);
        REQUANTIZE_CASE(36);
    default:
        return 0;
    }
#undef REQUANTIZE_CASE
}

static int32_t weight_sum(uint32_t op, uint32_t channel)
{
#define WEIGHT_SUM_CASE(n) case n: return resnet18_op_##n##_weight_sums[channel]
    switch (op) {
        WEIGHT_SUM_CASE(2); WEIGHT_SUM_CASE(5); WEIGHT_SUM_CASE(6);
        WEIGHT_SUM_CASE(8); WEIGHT_SUM_CASE(9); WEIGHT_SUM_CASE(12);
        WEIGHT_SUM_CASE(13); WEIGHT_SUM_CASE(14); WEIGHT_SUM_CASE(16);
        WEIGHT_SUM_CASE(17); WEIGHT_SUM_CASE(20); WEIGHT_SUM_CASE(21);
        WEIGHT_SUM_CASE(22); WEIGHT_SUM_CASE(24); WEIGHT_SUM_CASE(25);
        WEIGHT_SUM_CASE(28); WEIGHT_SUM_CASE(29); WEIGHT_SUM_CASE(30);
        WEIGHT_SUM_CASE(32); WEIGHT_SUM_CASE(33); WEIGHT_SUM_CASE(36);
    default: return 0;
    }
#undef WEIGHT_SUM_CASE
}

static int32_t weight_zero_point(uint32_t op, uint32_t channel)
{
#define WEIGHT_ZP_CASE(n) case n: return resnet18_op_##n##_weight_zero_points[channel]
    switch (op) {
        WEIGHT_ZP_CASE(2); WEIGHT_ZP_CASE(5); WEIGHT_ZP_CASE(6);
        WEIGHT_ZP_CASE(8); WEIGHT_ZP_CASE(9); WEIGHT_ZP_CASE(12);
        WEIGHT_ZP_CASE(13); WEIGHT_ZP_CASE(14); WEIGHT_ZP_CASE(16);
        WEIGHT_ZP_CASE(17); WEIGHT_ZP_CASE(20); WEIGHT_ZP_CASE(21);
        WEIGHT_ZP_CASE(22); WEIGHT_ZP_CASE(24); WEIGHT_ZP_CASE(25);
        WEIGHT_ZP_CASE(28); WEIGHT_ZP_CASE(29); WEIGHT_ZP_CASE(30);
        WEIGHT_ZP_CASE(32); WEIGHT_ZP_CASE(33); WEIGHT_ZP_CASE(36);
    default: return 0;
    }
#undef WEIGHT_ZP_CASE
}

static void transpose_nchw_to_nhwc(const int8_t *input, int8_t *output,
                                   uint32_t channels, uint32_t height,
                                   uint32_t width)
{
    uint32_t c;
    uint32_t y;
    uint32_t x;
    for (c = 0; c < channels; ++c) {
        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
                output[(y * width + x) * channels + c] =
                    input[(c * height + y) * width + x];
            }
        }
    }
}

static void pad_tensor(const int8_t *input, int8_t *output,
                       uint32_t height, uint32_t width, uint32_t channels,
                       const int32_t *padding, int8_t value)
{
    uint32_t top = (uint32_t)padding[2];
    uint32_t bottom = (uint32_t)padding[3];
    uint32_t left = (uint32_t)padding[4];
    uint32_t right = (uint32_t)padding[5];
    uint32_t out_height = height + top + bottom;
    uint32_t out_width = width + left + right;
    uint32_t y;
    uint32_t x;
    uint32_t c;
    for (y = 0; y < out_height; ++y) {
        for (x = 0; x < out_width; ++x) {
            for (c = 0; c < channels; ++c) {
                uint32_t index = (y * out_width + x) * channels + c;
                if (y < top || y >= top + height ||
                    x < left || x >= left + width) {
                    output[index] = value;
                } else {
                    uint32_t source = ((y - top) * width + x - left) *
                                      channels + c;
                    output[index] = input[source];
                }
            }
        }
    }
}

static __attribute__((noinline)) void conv2d_scalar(uint32_t op, const int8_t *input, const int8_t *weights,
                          const int32_t *bias, int8_t *output,
                          uint32_t height, uint32_t width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width,
                          uint32_t output_channels, uint32_t kernel_height,
                          uint32_t kernel_width, uint32_t stride,
                          int32_t input_zero_point)
{
    uint32_t oy;
    uint32_t ox;
    uint32_t oc;
    for (oy = 0; oy < output_height; ++oy) {
        for (ox = 0; ox < output_width; ++ox) {
            for (oc = 0; oc < output_channels; ++oc) {
                int64_t accumulator = bias[oc];
                int64_t input_sum = 0;
                uint32_t ky;
                uint32_t kx;
                for (ky = 0; ky < kernel_height; ++ky) {
                    for (kx = 0; kx < kernel_width; ++kx) {
                        uint32_t total_pad_y = output_height > 0 &&
                            (output_height - 1U) * stride + kernel_height > height ?
                            (output_height - 1U) * stride + kernel_height - height : 0;
                        uint32_t total_pad_x = output_width > 0 &&
                            (output_width - 1U) * stride + kernel_width > width ?
                            (output_width - 1U) * stride + kernel_width - width : 0;
                        int32_t iy = (int32_t)(oy * stride + ky) -
                                     (int32_t)(total_pad_y / 2U);
                        int32_t ix = (int32_t)(ox * stride + kx) -
                                     (int32_t)(total_pad_x / 2U);
                        uint32_t ic;
                        for (ic = 0; ic < input_channels; ++ic) {
                            int32_t input_value = input_zero_point;
                            int32_t weight_value = weights[((oc * kernel_height +
                                ky) * kernel_width + kx) * input_channels + ic];
                            if (iy >= 0 && ix >= 0 && (uint32_t)iy < height &&
                                (uint32_t)ix < width) {
                                input_value = input[((uint32_t)iy * width +
                                                     (uint32_t)ix) *
                                                    input_channels + ic];
                            }

                            accumulator += (int64_t)input_value * weight_value;
                            input_sum += input_value;
                        }
                    }
                }
                accumulator -= (int64_t)input_zero_point * weight_sum(op, oc);
                accumulator -= (int64_t)weight_zero_point(op, oc) * input_sum;
                accumulator += (int64_t)kernel_height * kernel_width *
                               input_channels * input_zero_point *
                               weight_zero_point(op, oc);
                        output[(oy * output_width + ox) * output_channels + oc] =
                    requantize_channel(op, oc, accumulator);
                        if (resnet18_operators[op].fused_activation == 1 &&
                            output[(oy * output_width + ox) * output_channels + oc] <
                            resnet18_tensors[resnet18_operators[op].outputs[0]].zero_point)
                            output[(oy * output_width + ox) * output_channels + oc] =
                                (int8_t)resnet18_tensors[
                                    resnet18_operators[op].outputs[0]].zero_point;
            }
        }
    }
}

static __attribute__((noinline)) void conv2d_b1(uint32_t op, const int8_t *input, const int8_t *weights,
                      const int32_t *bias, int8_t *output,
                      uint32_t height, uint32_t width, uint32_t input_channels,
                      uint32_t output_height, uint32_t output_width,
                      uint32_t output_channels, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride,
                      int32_t input_zero_point)
{
    uint32_t pad_y = (output_height - 1U) * stride + kernel_height > height ?
        (output_height - 1U) * stride + kernel_height - height : 0;
    uint32_t pad_x = (output_width - 1U) * stride + kernel_width > width ?
        (output_width - 1U) * stride + kernel_width - width : 0;
    uint32_t oy;
    uint32_t ox;
    uint32_t oc;
    for (oy = 0; oy < output_height; ++oy) {
        for (ox = 0; ox < output_width; ++ox) {
            int32_t base_y = (int32_t)(oy * stride) - (int32_t)(pad_y / 2U);
            int32_t base_x = (int32_t)(ox * stride) - (int32_t)(pad_x / 2U);
            for (oc = 0; oc < output_channels; ++oc) {
                int64_t accumulator = bias[oc];
                int64_t input_sum = 0;
                uint32_t ky;
                for (ky = 0; ky < kernel_height; ++ky) {
                    uint32_t kx;
                    for (kx = 0; kx < kernel_width; ++kx) {
                        int32_t iy = base_y + (int32_t)ky;
                        int32_t ix = base_x + (int32_t)kx;
                        uint32_t ic;
                        for (ic = 0; ic < input_channels; ++ic) {
                            int8_t value = (int8_t)input_zero_point;
                            if (iy >= 0 && ix >= 0 && (uint32_t)iy < height &&
                                (uint32_t)ix < width)
                                value = input[((uint32_t)iy * width +
                                              (uint32_t)ix) * input_channels + ic];
                            accumulator += (int64_t)value *
                                weights[((oc * kernel_height + ky) * kernel_width +
                                         kx) * input_channels + ic];
                            input_sum += value;
                        }
                    }
                }
                accumulator -= (int64_t)input_zero_point * weight_sum(op, oc);
                accumulator -= (int64_t)weight_zero_point(op, oc) * input_sum;
                accumulator += (int64_t)kernel_height * kernel_width *
                    input_channels * input_zero_point * weight_zero_point(op, oc);
                output[(oy * output_width + ox) * output_channels + oc] =
                    requantize_channel(op, oc, accumulator);
                if (resnet18_operators[op].fused_activation == 1 &&
                    output[(oy * output_width + ox) * output_channels + oc] <
                    resnet18_tensors[resnet18_operators[op].outputs[0]].zero_point)
                    output[(oy * output_width + ox) * output_channels + oc] =
                        (int8_t)resnet18_tensors[
                            resnet18_operators[op].outputs[0]].zero_point;
            }
        }
    }
}

#if defined(__riscv) && (__riscv_xlen == 64)
static inline int8_t conv2d_input_value(const int8_t *input,
                                        uint32_t height, uint32_t width,
                                        uint32_t input_channels,
                                        int32_t base_y, int32_t base_x,
                                        uint32_t kernel_width,
                                        uint32_t position,
                                        int32_t input_zero_point)
{
    uint32_t plane = position / input_channels;
    uint32_t ic = position % input_channels;
    uint32_t ky = plane / kernel_width;
    uint32_t kx = plane % kernel_width;
    int32_t iy = base_y + (int32_t)ky;
    int32_t ix = base_x + (int32_t)kx;

    if (iy < 0 || ix < 0 || (uint32_t)iy >= height ||
        (uint32_t)ix >= width)
        return (int8_t)input_zero_point;
    return input[((uint32_t)iy * width + (uint32_t)ix) *
                 input_channels + ic];
}

static __attribute__((noinline)) void conv2d_custom(uint32_t op, const int8_t *input, const int8_t *weights,
                          const int32_t *bias, int8_t *output,
                          uint32_t height, uint32_t width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width,
                          uint32_t output_channels, uint32_t kernel_height,
                          uint32_t kernel_width, uint32_t stride,
                          int32_t input_zero_point)
{
    uint32_t pad_y = (output_height - 1U) * stride + kernel_height > height ?
        (output_height - 1U) * stride + kernel_height - height : 0;
    uint32_t pad_x = (output_width - 1U) * stride + kernel_width > width ?
        (output_width - 1U) * stride + kernel_width - width : 0;
    uint32_t oy;
    uint32_t ox;
    uint32_t oc;
    for (oy = 0; oy < output_height; ++oy) {
        for (ox = 0; ox < output_width; ++ox) {
            int32_t base_y = (int32_t)(oy * stride) - (int32_t)(pad_y / 2U);
            int32_t base_x = (int32_t)(ox * stride) - (int32_t)(pad_x / 2U);
            uint32_t reduction = kernel_height * kernel_width * input_channels;
            uint32_t output_base = (oy * output_width + ox) * output_channels;
            for (oc = 0; oc < output_channels; oc += 8U) {
                uint64_t accumulators[8];
                uint32_t block_channels = output_channels - oc;
                int64_t input_sum = 0;
                uint32_t lane;
                uint32_t position;

                if (block_channels > 8U)
                    block_channels = 8U;
                for (lane = 0; lane < block_channels; ++lane)
                    accumulators[lane] = (uint64_t)bias[oc + lane];

                for (position = 0; position + 8U <= reduction;
                     position += 8U) {
                    int8_t activation_lanes[8];
                    uint64_t packed_activation;

                    for (lane = 0; lane < 8U; ++lane) {
                        activation_lanes[lane] = conv2d_input_value(
                            input, height, width, input_channels, base_y,
                            base_x, kernel_width, position + lane,
                            input_zero_point);
                        input_sum += activation_lanes[lane];
                    }
                    packed_activation = xq_pack_i8(activation_lanes, 8U, 1U);

                    for (lane = 0; lane < block_channels; ++lane) {
                        uint64_t packed_weights = xq_pack_i8(
                            &weights[(oc + lane) * reduction + position],
                            8U, 1U);
                        accumulators[lane] = (uint64_t)xqmac8_emit(
                            (int64_t)accumulators[lane],
                            packed_activation, packed_weights);
                    }
                }

                for (; position < reduction; ++position) {
                    int8_t value = conv2d_input_value(
                        input, height, width, input_channels, base_y, base_x,
                        kernel_width, position, input_zero_point);
                    input_sum += value;
                    for (lane = 0; lane < block_channels; ++lane)
                        accumulators[lane] += (uint64_t)((int64_t)value *
                            weights[(oc + lane) * reduction + position]);
                }

                for (lane = 0; lane < block_channels; ++lane) {
                    uint32_t channel = oc + lane;
                    int64_t accumulator = (int64_t)accumulators[lane];
                    int8_t result;

                    accumulator -= (int64_t)input_zero_point *
                                   weight_sum(op, channel);
                    accumulator -= (int64_t)weight_zero_point(op, channel) *
                                   input_sum;
                    accumulator += (int64_t)reduction * input_zero_point *
                                   weight_zero_point(op, channel);
                    result = requantize_channel(op, channel, accumulator);
                    if (resnet18_operators[op].fused_activation == 1 &&
                        result < resnet18_tensors[
                            resnet18_operators[op].outputs[0]].zero_point)
                        result = (int8_t)resnet18_tensors[
                            resnet18_operators[op].outputs[0]].zero_point;
                    output[output_base + channel] = result;
                }
            }
        }
    }
}
#endif

static void max_pool(const int8_t *input, int8_t *output, uint32_t height,
                     uint32_t width, uint32_t channels, uint32_t out_height,
                     uint32_t out_width, uint32_t filter)
{
    uint32_t oy;
    uint32_t ox;
    uint32_t c;
    for (oy = 0; oy < out_height; ++oy) {
        for (ox = 0; ox < out_width; ++ox) {
            for (c = 0; c < channels; ++c) {
                int8_t value = -128;
                uint32_t ky;
                uint32_t kx;
                for (ky = 0; ky < filter; ++ky) {
                    for (kx = 0; kx < filter; ++kx) {
                        uint32_t iy = oy * 2U + ky;
                        uint32_t ix = ox * 2U + kx;
                        if (iy < height && ix < width) {
                            int8_t candidate = input[(iy * width + ix) *
                                                     channels + c];
                            if (candidate > value) value = candidate;
                        }
                    }
                }
                output[(oy * out_width + ox) * channels + c] = value;
            }
        }
    }
}

static void average_pool(const int8_t *input, int8_t *output, uint32_t height,
                         uint32_t width, uint32_t channels, uint32_t filter)
{
    (void)height;
    uint32_t c;
    for (c = 0; c < channels; ++c) {
        int64_t sum = 0;
        uint32_t y;
        uint32_t x;
        for (y = 0; y < filter; ++y) {
            for (x = 0; x < filter; ++x)
                sum += input[(y * width + x) * channels + c] -
                       resnet18_tensors[73].zero_point;
        }
        output[c] = (int8_t)(round_divide(sum, (int32_t)(filter * filter)) +
                             resnet18_tensors[82].zero_point);
    }
}

static void add_scalar(uint32_t op, const int8_t *left, const int8_t *right,
                       int8_t *output, uint32_t elements)
{
#define ADD_PARAMS_CASE(n) case n: left_zp = resnet18_op_##n##_input_zero_points[0]; right_zp = resnet18_op_##n##_input_zero_points[1]; left_m = resnet18_op_##n##_add_multipliers[0]; right_m = resnet18_op_##n##_add_multipliers[1]; left_s = resnet18_op_##n##_add_shifts[0]; right_s = resnet18_op_##n##_add_shifts[1]; output_zp = resnet18_op_##n##_output_zero_point; break
    int32_t left_zp;
    int32_t right_zp;
    int32_t left_m;
    int32_t right_m;
    uint8_t left_s;
    uint8_t right_s;
    int32_t output_zp;
    switch (op) {
        ADD_PARAMS_CASE(7); ADD_PARAMS_CASE(10); ADD_PARAMS_CASE(15);
        ADD_PARAMS_CASE(18); ADD_PARAMS_CASE(23); ADD_PARAMS_CASE(26);
        ADD_PARAMS_CASE(31); ADD_PARAMS_CASE(34);
    default:
        return;
    }
#undef ADD_PARAMS_CASE
    uint32_t i;
    uint8_t common_shift = left_s > right_s ? left_s : right_s;
    for (i = 0; i < elements; ++i) {
        int64_t left_product = (int64_t)(left[i] - left_zp) * left_m;
        int64_t right_product = (int64_t)(right[i] - right_zp) * right_m;
        int64_t scaled_sum = (left_product << (common_shift - left_s)) +
                             (right_product << (common_shift - right_s));
        output[i] = clamp_int8_value(
            round_divide(scaled_sum, 1LL << common_shift) + output_zp);
        if (resnet18_operators[op].fused_activation == 1 &&
            output[i] < output_zp)
            output[i] = (int8_t)output_zp;
    }
}

void resnet18_run(void)
{
    uint32_t op;
    for (op = 0; op < resnet18_operator_count; ++op) {
        resnet18_operator_t current = resnet18_operators[op];
        int32_t input_id = current.inputs[0];
        int32_t output_id = current.outputs[0];
        resnet18_tensor_t *input = &resnet18_tensors[input_id];
        resnet18_tensor_t *output = &resnet18_tensors[output_id];
        const int8_t *input_data = input->constant_data ?
            (const int8_t *)input->constant_data : input->data;
        switch (current.operator_kind) {
        case 7:
            transpose_nchw_to_nhwc(input_data, output->data, 3, 224, 224);
            break;
        case 6:
            pad_tensor(input_data, output->data, (uint32_t)input->shape[1],
                       (uint32_t)input->shape[2], (uint32_t)input->shape[3],
                       (const int32_t *)resnet18_tensors[current.inputs[1]].constant_data,
                       (int8_t)resnet18_tensors[input_id].zero_point);
            break;
        case 3: {
            const int8_t *weights = (const int8_t *)resnet18_tensors[current.inputs[1]].constant_data;
            const int32_t *bias = (const int32_t *)resnet18_tensors[current.inputs[2]].constant_data;
            if (op == 2U)
                __asm__ volatile (".globl resnet18_initial_conv_start\nresnet18_initial_conv_start:");
#if defined(__riscv) && (__riscv_xlen == 64)
            if (RESNET18_EXECUTION_MODE == 2 ||
                RESNET18_EXECUTION_MODE == 3) {
                conv2d_custom(op, input_data, weights, bias, output->data,
                              input->shape[1], input->shape[2], input->shape[3],
                              output->shape[1], output->shape[2],
                              output->shape[3],
                              resnet18_tensors[current.inputs[1]].shape[1],
                              resnet18_tensors[current.inputs[1]].shape[2],
                              current.stride_h, input->zero_point);
            } else
#endif
            if (RESNET18_EXECUTION_MODE == 1) {
                conv2d_b1(op, input_data, weights, bias, output->data,
                          input->shape[1], input->shape[2], input->shape[3],
                          output->shape[1], output->shape[2],
                          output->shape[3],
                          resnet18_tensors[current.inputs[1]].shape[1],
                          resnet18_tensors[current.inputs[1]].shape[2],
                          current.stride_h, input->zero_point);
            } else {
                conv2d_scalar(op, input_data, weights, bias, output->data,
                              input->shape[1], input->shape[2], input->shape[3],
                              output->shape[1], output->shape[2],
                              output->shape[3],
                              resnet18_tensors[current.inputs[1]].shape[1],
                              resnet18_tensors[current.inputs[1]].shape[2],
                              current.stride_h, input->zero_point);
            }
            if (op == 2U)
                __asm__ volatile (".globl resnet18_initial_conv_end\nresnet18_initial_conv_end:");
            break;
        }
        case 5:
            max_pool(input_data, output->data, input->shape[1], input->shape[2],
                     input->shape[3], output->shape[1], output->shape[2],
                     current.filter_h);
            break;
        case 1:
            add_scalar(op, input_data,
                       (const int8_t *)resnet18_tensors[current.inputs[1]].data,
                       output->data, output->elements);
            break;
        case 2:
            average_pool(input_data, output->data, input->shape[1],
                         input->shape[2], input->shape[3], current.filter_h);
            break;
        case 4: {
            const int8_t *weights = (const int8_t *)resnet18_tensors[current.inputs[1]].constant_data;
            const int32_t *bias = (const int32_t *)resnet18_tensors[current.inputs[2]].constant_data;
            __asm__ volatile (".globl resnet18_fc_start\nresnet18_fc_start:");
            fully_connected_scalar(op, input_data, weights, bias, output->data,
                                   input->elements, output->elements,
                                   input->zero_point, output->zero_point);
            __asm__ volatile (".globl resnet18_fc_end\nresnet18_fc_end:");
            break;
        }
        default:
            break;
        }
    }
}

void resnet18_load_input(const int8_t *input, uint32_t elements)
{
    uint32_t index;
    for (index = 0; index < elements; ++index)
        resnet18_tensors[85].data[index] = input[index];
}

uint64_t resnet18_output_checksum(void)
{
    uint64_t result = 14695981039346656037ULL;
    uint32_t index;
    for (index = 0; index < resnet18_tensors[41].elements; ++index) {
        result ^= (uint64_t)(uint8_t)resnet18_tensors[41].data[index];
        result *= 1099511628211ULL;
    }
    return result;
}

void resnet18_copy_output(int8_t *output, uint32_t elements)
{
    uint32_t index;
    for (index = 0; index < elements && index < resnet18_tensors[41].elements;
         ++index)
        output[index] = resnet18_tensors[41].data[index];
}

void resnet18_copy_tensor(uint32_t tensor_id, int8_t *output, uint32_t elements)
{
    uint32_t index;
    if (tensor_id >= resnet18_tensor_count)
        return;
    for (index = 0; index < elements &&
                  index < resnet18_tensors[tensor_id].elements; ++index)
        output[index] = resnet18_tensors[tensor_id].data[index];
}
