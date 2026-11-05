// EE645500 begin
// The custom RV64 instruction treats both operands as eight packed signed
// INT8 values and performs a read-modify-write update of the destination.
require_rv64;

int64_t dot = 0;
for (unsigned i = 0; i < 8; ++i) {
  int64_t a = static_cast<int64_t>((RS1 >> (i * 8)) & 0xff);
  int64_t w = static_cast<int64_t>((RS2 >> (i * 8)) & 0xff);
  if ((a & 0x80) != 0) {
    a -= 0x100;
  }
  if ((w & 0x80) != 0) {
    w -= 0x100;
  }
  dot += a * w;
}

WRITE_RD(static_cast<reg_t>(RD + static_cast<reg_t>(dot)));
// EE645500 end
