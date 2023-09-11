###################################################################

# Created by write_sdc on Mon Sep 11 18:05:32 2023

###################################################################
set sdc_version 2.1

set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA
set_max_area 0
create_clock -name vclk  -period 0.6  -waveform {0 0.3}
set_input_delay -clock vclk  0  [get_ports {codeword[71]}]
set_input_delay -clock vclk  0  [get_ports {codeword[70]}]
set_input_delay -clock vclk  0  [get_ports {codeword[69]}]
set_input_delay -clock vclk  0  [get_ports {codeword[68]}]
set_input_delay -clock vclk  0  [get_ports {codeword[67]}]
set_input_delay -clock vclk  0  [get_ports {codeword[66]}]
set_input_delay -clock vclk  0  [get_ports {codeword[65]}]
set_input_delay -clock vclk  0  [get_ports {codeword[64]}]
set_input_delay -clock vclk  0  [get_ports {codeword[63]}]
set_input_delay -clock vclk  0  [get_ports {codeword[62]}]
set_input_delay -clock vclk  0  [get_ports {codeword[61]}]
set_input_delay -clock vclk  0  [get_ports {codeword[60]}]
set_input_delay -clock vclk  0  [get_ports {codeword[59]}]
set_input_delay -clock vclk  0  [get_ports {codeword[58]}]
set_input_delay -clock vclk  0  [get_ports {codeword[57]}]
set_input_delay -clock vclk  0  [get_ports {codeword[56]}]
set_input_delay -clock vclk  0  [get_ports {codeword[55]}]
set_input_delay -clock vclk  0  [get_ports {codeword[54]}]
set_input_delay -clock vclk  0  [get_ports {codeword[53]}]
set_input_delay -clock vclk  0  [get_ports {codeword[52]}]
set_input_delay -clock vclk  0  [get_ports {codeword[51]}]
set_input_delay -clock vclk  0  [get_ports {codeword[50]}]
set_input_delay -clock vclk  0  [get_ports {codeword[49]}]
set_input_delay -clock vclk  0  [get_ports {codeword[48]}]
set_input_delay -clock vclk  0  [get_ports {codeword[47]}]
set_input_delay -clock vclk  0  [get_ports {codeword[46]}]
set_input_delay -clock vclk  0  [get_ports {codeword[45]}]
set_input_delay -clock vclk  0  [get_ports {codeword[44]}]
set_input_delay -clock vclk  0  [get_ports {codeword[43]}]
set_input_delay -clock vclk  0  [get_ports {codeword[42]}]
set_input_delay -clock vclk  0  [get_ports {codeword[41]}]
set_input_delay -clock vclk  0  [get_ports {codeword[40]}]
set_input_delay -clock vclk  0  [get_ports {codeword[39]}]
set_input_delay -clock vclk  0  [get_ports {codeword[38]}]
set_input_delay -clock vclk  0  [get_ports {codeword[37]}]
set_input_delay -clock vclk  0  [get_ports {codeword[36]}]
set_input_delay -clock vclk  0  [get_ports {codeword[35]}]
set_input_delay -clock vclk  0  [get_ports {codeword[34]}]
set_input_delay -clock vclk  0  [get_ports {codeword[33]}]
set_input_delay -clock vclk  0  [get_ports {codeword[32]}]
set_input_delay -clock vclk  0  [get_ports {codeword[31]}]
set_input_delay -clock vclk  0  [get_ports {codeword[30]}]
set_input_delay -clock vclk  0  [get_ports {codeword[29]}]
set_input_delay -clock vclk  0  [get_ports {codeword[28]}]
set_input_delay -clock vclk  0  [get_ports {codeword[27]}]
set_input_delay -clock vclk  0  [get_ports {codeword[26]}]
set_input_delay -clock vclk  0  [get_ports {codeword[25]}]
set_input_delay -clock vclk  0  [get_ports {codeword[24]}]
set_input_delay -clock vclk  0  [get_ports {codeword[23]}]
set_input_delay -clock vclk  0  [get_ports {codeword[22]}]
set_input_delay -clock vclk  0  [get_ports {codeword[21]}]
set_input_delay -clock vclk  0  [get_ports {codeword[20]}]
set_input_delay -clock vclk  0  [get_ports {codeword[19]}]
set_input_delay -clock vclk  0  [get_ports {codeword[18]}]
set_input_delay -clock vclk  0  [get_ports {codeword[17]}]
set_input_delay -clock vclk  0  [get_ports {codeword[16]}]
set_input_delay -clock vclk  0  [get_ports {codeword[15]}]
set_input_delay -clock vclk  0  [get_ports {codeword[14]}]
set_input_delay -clock vclk  0  [get_ports {codeword[13]}]
set_input_delay -clock vclk  0  [get_ports {codeword[12]}]
set_input_delay -clock vclk  0  [get_ports {codeword[11]}]
set_input_delay -clock vclk  0  [get_ports {codeword[10]}]
set_input_delay -clock vclk  0  [get_ports {codeword[9]}]
set_input_delay -clock vclk  0  [get_ports {codeword[8]}]
set_input_delay -clock vclk  0  [get_ports {codeword[7]}]
set_input_delay -clock vclk  0  [get_ports {codeword[6]}]
set_input_delay -clock vclk  0  [get_ports {codeword[5]}]
set_input_delay -clock vclk  0  [get_ports {codeword[4]}]
set_input_delay -clock vclk  0  [get_ports {codeword[3]}]
set_input_delay -clock vclk  0  [get_ports {codeword[2]}]
set_input_delay -clock vclk  0  [get_ports {codeword[1]}]
set_input_delay -clock vclk  0  [get_ports {codeword[0]}]
set_output_delay -clock vclk  0  [get_ports {message[63]}]
set_output_delay -clock vclk  0  [get_ports {message[62]}]
set_output_delay -clock vclk  0  [get_ports {message[61]}]
set_output_delay -clock vclk  0  [get_ports {message[60]}]
set_output_delay -clock vclk  0  [get_ports {message[59]}]
set_output_delay -clock vclk  0  [get_ports {message[58]}]
set_output_delay -clock vclk  0  [get_ports {message[57]}]
set_output_delay -clock vclk  0  [get_ports {message[56]}]
set_output_delay -clock vclk  0  [get_ports {message[55]}]
set_output_delay -clock vclk  0  [get_ports {message[54]}]
set_output_delay -clock vclk  0  [get_ports {message[53]}]
set_output_delay -clock vclk  0  [get_ports {message[52]}]
set_output_delay -clock vclk  0  [get_ports {message[51]}]
set_output_delay -clock vclk  0  [get_ports {message[50]}]
set_output_delay -clock vclk  0  [get_ports {message[49]}]
set_output_delay -clock vclk  0  [get_ports {message[48]}]
set_output_delay -clock vclk  0  [get_ports {message[47]}]
set_output_delay -clock vclk  0  [get_ports {message[46]}]
set_output_delay -clock vclk  0  [get_ports {message[45]}]
set_output_delay -clock vclk  0  [get_ports {message[44]}]
set_output_delay -clock vclk  0  [get_ports {message[43]}]
set_output_delay -clock vclk  0  [get_ports {message[42]}]
set_output_delay -clock vclk  0  [get_ports {message[41]}]
set_output_delay -clock vclk  0  [get_ports {message[40]}]
set_output_delay -clock vclk  0  [get_ports {message[39]}]
set_output_delay -clock vclk  0  [get_ports {message[38]}]
set_output_delay -clock vclk  0  [get_ports {message[37]}]
set_output_delay -clock vclk  0  [get_ports {message[36]}]
set_output_delay -clock vclk  0  [get_ports {message[35]}]
set_output_delay -clock vclk  0  [get_ports {message[34]}]
set_output_delay -clock vclk  0  [get_ports {message[33]}]
set_output_delay -clock vclk  0  [get_ports {message[32]}]
set_output_delay -clock vclk  0  [get_ports {message[31]}]
set_output_delay -clock vclk  0  [get_ports {message[30]}]
set_output_delay -clock vclk  0  [get_ports {message[29]}]
set_output_delay -clock vclk  0  [get_ports {message[28]}]
set_output_delay -clock vclk  0  [get_ports {message[27]}]
set_output_delay -clock vclk  0  [get_ports {message[26]}]
set_output_delay -clock vclk  0  [get_ports {message[25]}]
set_output_delay -clock vclk  0  [get_ports {message[24]}]
set_output_delay -clock vclk  0  [get_ports {message[23]}]
set_output_delay -clock vclk  0  [get_ports {message[22]}]
set_output_delay -clock vclk  0  [get_ports {message[21]}]
set_output_delay -clock vclk  0  [get_ports {message[20]}]
set_output_delay -clock vclk  0  [get_ports {message[19]}]
set_output_delay -clock vclk  0  [get_ports {message[18]}]
set_output_delay -clock vclk  0  [get_ports {message[17]}]
set_output_delay -clock vclk  0  [get_ports {message[16]}]
set_output_delay -clock vclk  0  [get_ports {message[15]}]
set_output_delay -clock vclk  0  [get_ports {message[14]}]
set_output_delay -clock vclk  0  [get_ports {message[13]}]
set_output_delay -clock vclk  0  [get_ports {message[12]}]
set_output_delay -clock vclk  0  [get_ports {message[11]}]
set_output_delay -clock vclk  0  [get_ports {message[10]}]
set_output_delay -clock vclk  0  [get_ports {message[9]}]
set_output_delay -clock vclk  0  [get_ports {message[8]}]
set_output_delay -clock vclk  0  [get_ports {message[7]}]
set_output_delay -clock vclk  0  [get_ports {message[6]}]
set_output_delay -clock vclk  0  [get_ports {message[5]}]
set_output_delay -clock vclk  0  [get_ports {message[4]}]
set_output_delay -clock vclk  0  [get_ports {message[3]}]
set_output_delay -clock vclk  0  [get_ports {message[2]}]
set_output_delay -clock vclk  0  [get_ports {message[1]}]
set_output_delay -clock vclk  0  [get_ports {message[0]}]
set_output_delay -clock vclk  0  [get_ports {error_type[1]}]
set_output_delay -clock vclk  0  [get_ports {error_type[0]}]
set_input_transition -max 0.1  [get_ports {codeword[71]}]
set_input_transition -min 0.1  [get_ports {codeword[71]}]
set_input_transition -max 0.1  [get_ports {codeword[70]}]
set_input_transition -min 0.1  [get_ports {codeword[70]}]
set_input_transition -max 0.1  [get_ports {codeword[69]}]
set_input_transition -min 0.1  [get_ports {codeword[69]}]
set_input_transition -max 0.1  [get_ports {codeword[68]}]
set_input_transition -min 0.1  [get_ports {codeword[68]}]
set_input_transition -max 0.1  [get_ports {codeword[67]}]
set_input_transition -min 0.1  [get_ports {codeword[67]}]
set_input_transition -max 0.1  [get_ports {codeword[66]}]
set_input_transition -min 0.1  [get_ports {codeword[66]}]
set_input_transition -max 0.1  [get_ports {codeword[65]}]
set_input_transition -min 0.1  [get_ports {codeword[65]}]
set_input_transition -max 0.1  [get_ports {codeword[64]}]
set_input_transition -min 0.1  [get_ports {codeword[64]}]
set_input_transition -max 0.1  [get_ports {codeword[63]}]
set_input_transition -min 0.1  [get_ports {codeword[63]}]
set_input_transition -max 0.1  [get_ports {codeword[62]}]
set_input_transition -min 0.1  [get_ports {codeword[62]}]
set_input_transition -max 0.1  [get_ports {codeword[61]}]
set_input_transition -min 0.1  [get_ports {codeword[61]}]
set_input_transition -max 0.1  [get_ports {codeword[60]}]
set_input_transition -min 0.1  [get_ports {codeword[60]}]
set_input_transition -max 0.1  [get_ports {codeword[59]}]
set_input_transition -min 0.1  [get_ports {codeword[59]}]
set_input_transition -max 0.1  [get_ports {codeword[58]}]
set_input_transition -min 0.1  [get_ports {codeword[58]}]
set_input_transition -max 0.1  [get_ports {codeword[57]}]
set_input_transition -min 0.1  [get_ports {codeword[57]}]
set_input_transition -max 0.1  [get_ports {codeword[56]}]
set_input_transition -min 0.1  [get_ports {codeword[56]}]
set_input_transition -max 0.1  [get_ports {codeword[55]}]
set_input_transition -min 0.1  [get_ports {codeword[55]}]
set_input_transition -max 0.1  [get_ports {codeword[54]}]
set_input_transition -min 0.1  [get_ports {codeword[54]}]
set_input_transition -max 0.1  [get_ports {codeword[53]}]
set_input_transition -min 0.1  [get_ports {codeword[53]}]
set_input_transition -max 0.1  [get_ports {codeword[52]}]
set_input_transition -min 0.1  [get_ports {codeword[52]}]
set_input_transition -max 0.1  [get_ports {codeword[51]}]
set_input_transition -min 0.1  [get_ports {codeword[51]}]
set_input_transition -max 0.1  [get_ports {codeword[50]}]
set_input_transition -min 0.1  [get_ports {codeword[50]}]
set_input_transition -max 0.1  [get_ports {codeword[49]}]
set_input_transition -min 0.1  [get_ports {codeword[49]}]
set_input_transition -max 0.1  [get_ports {codeword[48]}]
set_input_transition -min 0.1  [get_ports {codeword[48]}]
set_input_transition -max 0.1  [get_ports {codeword[47]}]
set_input_transition -min 0.1  [get_ports {codeword[47]}]
set_input_transition -max 0.1  [get_ports {codeword[46]}]
set_input_transition -min 0.1  [get_ports {codeword[46]}]
set_input_transition -max 0.1  [get_ports {codeword[45]}]
set_input_transition -min 0.1  [get_ports {codeword[45]}]
set_input_transition -max 0.1  [get_ports {codeword[44]}]
set_input_transition -min 0.1  [get_ports {codeword[44]}]
set_input_transition -max 0.1  [get_ports {codeword[43]}]
set_input_transition -min 0.1  [get_ports {codeword[43]}]
set_input_transition -max 0.1  [get_ports {codeword[42]}]
set_input_transition -min 0.1  [get_ports {codeword[42]}]
set_input_transition -max 0.1  [get_ports {codeword[41]}]
set_input_transition -min 0.1  [get_ports {codeword[41]}]
set_input_transition -max 0.1  [get_ports {codeword[40]}]
set_input_transition -min 0.1  [get_ports {codeword[40]}]
set_input_transition -max 0.1  [get_ports {codeword[39]}]
set_input_transition -min 0.1  [get_ports {codeword[39]}]
set_input_transition -max 0.1  [get_ports {codeword[38]}]
set_input_transition -min 0.1  [get_ports {codeword[38]}]
set_input_transition -max 0.1  [get_ports {codeword[37]}]
set_input_transition -min 0.1  [get_ports {codeword[37]}]
set_input_transition -max 0.1  [get_ports {codeword[36]}]
set_input_transition -min 0.1  [get_ports {codeword[36]}]
set_input_transition -max 0.1  [get_ports {codeword[35]}]
set_input_transition -min 0.1  [get_ports {codeword[35]}]
set_input_transition -max 0.1  [get_ports {codeword[34]}]
set_input_transition -min 0.1  [get_ports {codeword[34]}]
set_input_transition -max 0.1  [get_ports {codeword[33]}]
set_input_transition -min 0.1  [get_ports {codeword[33]}]
set_input_transition -max 0.1  [get_ports {codeword[32]}]
set_input_transition -min 0.1  [get_ports {codeword[32]}]
set_input_transition -max 0.1  [get_ports {codeword[31]}]
set_input_transition -min 0.1  [get_ports {codeword[31]}]
set_input_transition -max 0.1  [get_ports {codeword[30]}]
set_input_transition -min 0.1  [get_ports {codeword[30]}]
set_input_transition -max 0.1  [get_ports {codeword[29]}]
set_input_transition -min 0.1  [get_ports {codeword[29]}]
set_input_transition -max 0.1  [get_ports {codeword[28]}]
set_input_transition -min 0.1  [get_ports {codeword[28]}]
set_input_transition -max 0.1  [get_ports {codeword[27]}]
set_input_transition -min 0.1  [get_ports {codeword[27]}]
set_input_transition -max 0.1  [get_ports {codeword[26]}]
set_input_transition -min 0.1  [get_ports {codeword[26]}]
set_input_transition -max 0.1  [get_ports {codeword[25]}]
set_input_transition -min 0.1  [get_ports {codeword[25]}]
set_input_transition -max 0.1  [get_ports {codeword[24]}]
set_input_transition -min 0.1  [get_ports {codeword[24]}]
set_input_transition -max 0.1  [get_ports {codeword[23]}]
set_input_transition -min 0.1  [get_ports {codeword[23]}]
set_input_transition -max 0.1  [get_ports {codeword[22]}]
set_input_transition -min 0.1  [get_ports {codeword[22]}]
set_input_transition -max 0.1  [get_ports {codeword[21]}]
set_input_transition -min 0.1  [get_ports {codeword[21]}]
set_input_transition -max 0.1  [get_ports {codeword[20]}]
set_input_transition -min 0.1  [get_ports {codeword[20]}]
set_input_transition -max 0.1  [get_ports {codeword[19]}]
set_input_transition -min 0.1  [get_ports {codeword[19]}]
set_input_transition -max 0.1  [get_ports {codeword[18]}]
set_input_transition -min 0.1  [get_ports {codeword[18]}]
set_input_transition -max 0.1  [get_ports {codeword[17]}]
set_input_transition -min 0.1  [get_ports {codeword[17]}]
set_input_transition -max 0.1  [get_ports {codeword[16]}]
set_input_transition -min 0.1  [get_ports {codeword[16]}]
set_input_transition -max 0.1  [get_ports {codeword[15]}]
set_input_transition -min 0.1  [get_ports {codeword[15]}]
set_input_transition -max 0.1  [get_ports {codeword[14]}]
set_input_transition -min 0.1  [get_ports {codeword[14]}]
set_input_transition -max 0.1  [get_ports {codeword[13]}]
set_input_transition -min 0.1  [get_ports {codeword[13]}]
set_input_transition -max 0.1  [get_ports {codeword[12]}]
set_input_transition -min 0.1  [get_ports {codeword[12]}]
set_input_transition -max 0.1  [get_ports {codeword[11]}]
set_input_transition -min 0.1  [get_ports {codeword[11]}]
set_input_transition -max 0.1  [get_ports {codeword[10]}]
set_input_transition -min 0.1  [get_ports {codeword[10]}]
set_input_transition -max 0.1  [get_ports {codeword[9]}]
set_input_transition -min 0.1  [get_ports {codeword[9]}]
set_input_transition -max 0.1  [get_ports {codeword[8]}]
set_input_transition -min 0.1  [get_ports {codeword[8]}]
set_input_transition -max 0.1  [get_ports {codeword[7]}]
set_input_transition -min 0.1  [get_ports {codeword[7]}]
set_input_transition -max 0.1  [get_ports {codeword[6]}]
set_input_transition -min 0.1  [get_ports {codeword[6]}]
set_input_transition -max 0.1  [get_ports {codeword[5]}]
set_input_transition -min 0.1  [get_ports {codeword[5]}]
set_input_transition -max 0.1  [get_ports {codeword[4]}]
set_input_transition -min 0.1  [get_ports {codeword[4]}]
set_input_transition -max 0.1  [get_ports {codeword[3]}]
set_input_transition -min 0.1  [get_ports {codeword[3]}]
set_input_transition -max 0.1  [get_ports {codeword[2]}]
set_input_transition -min 0.1  [get_ports {codeword[2]}]
set_input_transition -max 0.1  [get_ports {codeword[1]}]
set_input_transition -min 0.1  [get_ports {codeword[1]}]
set_input_transition -max 0.1  [get_ports {codeword[0]}]
set_input_transition -min 0.1  [get_ports {codeword[0]}]
