module SCC_4LC_syndrome_gen(input [70:0] codeword, output [6:0] syndrome);

    assign syndrome[6] = ^(codeword&71'b0001010001001100100110011110111010010110011100010111000001111101_1000000);
    assign syndrome[5] = ^(codeword&71'b0111001110101010110101010001000111011101011010011100100001000011_0100000);
    assign syndrome[4] = ^(codeword&71'b0011000010011011111100110110011001111000111001011001010001011100_0010000);
    assign syndrome[3] = ^(codeword&71'b1000111000101111111110011011001100111100010100101100101000101110_0001000);
    assign syndrome[2] = ^(codeword&71'b1100110011110101111111001101100110011110001010010110010100010111_0000100);
    assign syndrome[1] = ^(codeword&71'b0000101001110110011001111000101001011001010001011100001011110110_0000010);
    assign syndrome[0] = ^(codeword&71'b0101010100011001001100111100110100101100101000101110000111111011_0000001);

endmodule