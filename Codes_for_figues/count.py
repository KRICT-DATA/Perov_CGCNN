import math
binary_config = 0
for i in range(1, 16):
    binary_config += math.comb(16, i)
binary_config *= 135

ternary_config = 0
for i in range(1, 16):
    for j in range(1, 16-i):
        ternary_config += math.comb(16, i)*math.comb(16-i, j)
ternary_config *= 180

quaternary_config = 0
for i in range(1, 16):
    for j in range(1, 16-i):
        for k in range(1, 16-i-j):
            quaternary_config += math.comb(16,i)*math.comb(16-i, j)*math.comb(16-i-j, k)
quaternary_config *= 135
config = binary_config + ternary_config + quaternary_config
print(f"{config:.1e}")
