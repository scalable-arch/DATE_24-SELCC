#!/bin/csh -f

cd /home/xyz123479/verilog/DATE24_SC_ECC/2_Hardware_overheads/IP-DAEC/Decoder/sim/output

#This ENV is used to avoid overriding current script in next vcselab run 
setenv SNPS_VCSELAB_SCRIPT_NO_OVERRIDE  1

/usr/synopsys/vcs/R-2020.12-SP1-1/linux64/bin/vcselab $* \
    -o \
    simv \
    -nobanner \

cd -

