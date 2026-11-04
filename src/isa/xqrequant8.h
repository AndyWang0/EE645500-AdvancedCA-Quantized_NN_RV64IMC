// EE645500 begin
// The custom RV64 instruction applies the project fixed-point contract:
// signed 128-bit product, ties-away-from-zero rounding, zero-point addition,
// and signed INT8 saturation.
require_rv64;

const int64_t accumulator = static_cast<int64_t>(RS1);
const int32_t multiplier = static_cast<int32_t>(RS2 & 0xffffffffu);
const unsigned shift = static_cast<unsigned>((RS2 >> 32) & 0x3fu);
const int8_t zero_point = static_cast<int8_t>((RS2 >> 38) & 0xffu);
const __int128 product = static_cast<__int128>(accumulator) *
                         static_cast<__int128>(multiplier);

__int128 scaled;
if (shift == 0) {
  scaled = product;
} else {
  const bool negative = product < 0;
  const __int128 magnitude = negative ? -product : product;
  const __int128 rounded =
      (magnitude + (static_cast<__int128>(1) << (shift - 1))) >> shift;
  scaled = negative ? -rounded : rounded;
}

__int128 quantized = scaled + static_cast<__int128>(zero_point);
if (quantized > 127) {
  quantized = 127;
} else if (quantized < -128) {
  quantized = -128;
}

WRITE_RD(static_cast<reg_t>(static_cast<int64_t>(static_cast<int8_t>(quantized))));
// EE645500 end
