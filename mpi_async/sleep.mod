V27 0x4 sleep
9 sleep.cuf S582 0
05/02/2014  13:31:34
use cudadevice public 0 direct
use iso_c_binding public 0 indirect
use pgi_acc_common public 0 indirect
use cudafor public 0 direct
enduse
D 444 24 2099 8 2098 7
D 450 24 2101 8 2100 7
D 462 24 2099 8 2098 7
D 480 24 2174 8 2173 7
D 1110 21 6 1 1049 1048 0 1 0 0 1
 1043 1046 1047 1043 1046 1044
D 1113 21 6 1 0 69 0 0 0 0 0
 0 69 0 3 69 0
S 582 24 0 0 0 8 1 0 4658 10005 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 sleep
S 585 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 591 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 592 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 593 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 594 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 595 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 2098 25 6 iso_c_binding c_ptr
R 2099 5 7 iso_c_binding val c_ptr
R 2100 25 8 iso_c_binding c_funptr
R 2101 5 9 iso_c_binding val c_funptr
R 2134 6 42 iso_c_binding c_null_ptr$ac
R 2136 6 44 iso_c_binding c_null_funptr$ac
R 2137 26 45 iso_c_binding ==
R 2139 26 47 iso_c_binding !=
R 2173 25 5 pgi_acc_common c_devptr
R 2174 5 6 pgi_acc_common cptr c_devptr
R 2176 6 8 pgi_acc_common c_null_devptr$ac
R 2182 26 14 pgi_acc_common =
S 6833 6 4 0 0 6 1 582 9285 40800006 0 A 0 0 0 0 0 0 0 0 0 0 0 0 6839 0 0 0 0 0 0 0 0 0 0 582 0 0 0 0 z_b_0
S 6834 7 6 0 0 1110 1 582 31139 10a00004 51 A 0 0 0 0 0 0 6836 0 0 0 6838 0 0 0 0 0 0 0 0 6835 0 0 6837 582 0 0 0 0 streams
S 6835 8 4 0 0 1113 6833 582 31147 40822004 1020 A 0 0 0 0 0 0 0 0 0 0 0 0 6839 0 0 0 0 0 0 0 0 0 0 582 0 0 0 0 streams$sd
S 6836 6 4 0 0 7 6837 582 31158 40802001 1020 A 0 0 0 0 0 0 0 0 0 0 0 0 6839 0 0 0 0 0 0 0 0 0 0 582 0 0 0 0 streams$p
S 6837 6 4 0 0 7 6835 582 31168 40802000 1020 A 0 0 0 0 0 0 0 0 0 0 0 0 6839 0 0 0 0 0 0 0 0 0 0 582 0 0 0 0 streams$o
S 6838 22 1 0 0 8 1 582 31178 40000000 1000 A 0 0 0 0 0 0 0 6834 0 0 0 0 6835 0 0 0 0 0 0 0 0 0 0 582 0 0 0 0 streams$arrdsc
S 6839 11 0 0 0 8 2188 582 31193 40800000 805000 A 0 0 0 0 0 92 0 0 6836 6833 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _sleep$0
S 6840 23 5 0 4 0 6841 582 4658 0 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 sleep
S 6841 14 5 0 4 0 1 6840 4658 0 400000 A 0 0 0 0 0 0 0 4241 1 0 0 0 0 0 0 0 0 0 0 0 0 11 0 582 0 0 0 0 sleep
F 6841 1 6842
S 6842 1 3 0 0 7 1 6840 31202 4 7000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _V_num_cycles
S 6843 23 5 0 0 7 6846 582 31216 4 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 get_cycles
S 6844 1 3 1 0 9 1 6843 31227 4 3000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 seconds
S 6845 1 3 0 0 7 1 6843 31235 4 1003000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num_cycles
S 6846 14 5 0 0 7 1 6843 31216 4 1400000 A 0 0 0 0 0 0 0 4243 1 0 0 6845 0 0 0 0 0 0 0 0 0 24 0 582 0 0 0 0 get_cycles
F 6846 1 6844
S 6847 23 5 0 0 0 6849 582 31246 0 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 create_streams
S 6848 1 3 0 0 6 1 6847 31261 4 3000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num_streams
S 6849 14 5 0 0 0 1 6847 31246 0 400000 A 0 0 0 0 0 0 0 4245 1 0 0 0 0 0 0 0 0 0 0 0 0 35 0 582 0 0 0 0 create_streams
F 6849 1 6848
S 6850 23 5 0 0 0 6853 582 31273 0 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 sleep_kernel
S 6851 1 3 0 0 7 1 6850 31286 4 3000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num_cycles
S 6852 1 3 0 0 6 1 6850 31297 4 3000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 stream_id
S 6853 14 5 0 0 0 1 6850 31273 0 400000 A 0 0 0 0 0 0 0 4247 2 0 0 0 0 0 0 0 0 0 0 0 0 46 0 582 0 0 0 0 sleep_kernel
F 6853 2 6851 6852
S 6854 23 5 0 0 0 6856 582 31307 0 0 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 destroy_streams
S 6855 1 3 0 0 6 1 6854 31261 4 3000 A 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 num_streams
S 6856 14 5 0 0 0 1 6854 31307 0 400000 A 0 0 0 0 0 0 0 4250 1 0 0 0 0 0 0 0 0 0 0 0 0 58 0 582 0 0 0 0 destroy_streams
F 6856 1 6855
A 69 2 0 0 0 6 591 0 0 0 69 0 0 0 0 0 0 0 0 0
A 71 2 0 0 0 6 585 0 0 0 71 0 0 0 0 0 0 0 0 0
A 73 2 0 0 0 6 592 0 0 0 73 0 0 0 0 0 0 0 0 0
A 75 2 0 0 0 6 593 0 0 0 75 0 0 0 0 0 0 0 0 0
A 79 2 0 0 0 6 594 0 0 0 79 0 0 0 0 0 0 0 0 0
A 81 2 0 0 0 6 595 0 0 0 81 0 0 0 0 0 0 0 0 0
A 336 1 0 0 0 444 2134 0 0 0 0 0 0 0 0 0 0 0 0 0
A 339 1 0 0 46 450 2136 0 0 0 0 0 0 0 0 0 0 0 0 0
A 356 1 0 0 316 480 2176 0 0 0 0 0 0 0 0 0 0 0 0 0
A 1042 1 0 1 0 1113 6835 0 0 0 0 0 0 0 0 0 0 0 0 0
A 1043 10 0 0 0 6 1042 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 73
A 1044 10 0 0 1043 6 1042 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 75
A 1045 4 0 0 560 6 1044 0 3 0 0 0 0 2 0 0 0 0 0 0
A 1046 4 0 0 266 6 1043 0 1045 0 0 0 0 1 0 0 0 0 0 0
A 1047 10 0 0 1044 6 1042 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 79
A 1048 10 0 0 1047 6 1042 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 81
A 1049 10 0 0 1048 6 1042 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 71
Z
J 149 1 1
V 336 444 7 0
S 0 444 0 0 0
A 0 6 0 0 1 2 0
J 150 1 1
V 339 450 7 0
S 0 450 0 0 0
A 0 6 0 0 1 2 0
J 30 1 1
V 356 480 7 0
S 0 480 0 0 0
A 0 462 0 0 1 336 0
Z
