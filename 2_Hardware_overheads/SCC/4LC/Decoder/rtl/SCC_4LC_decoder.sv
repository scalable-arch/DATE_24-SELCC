module SCC_4LC_syndrome_gen(input [70:0] codeword, output [6:0] syndrome);

    assign syndrome[6] = ^(codeword&71'b0001010001001100100110011110111010010110011100010111000001111101_1000000);
    assign syndrome[5] = ^(codeword&71'b0111001110101010110101010001000111011101011010011100100001000011_0100000);
    assign syndrome[4] = ^(codeword&71'b0011000010011011111100110110011001111000111001011001010001011100_0010000);
    assign syndrome[3] = ^(codeword&71'b1000111000101111111110011011001100111100010100101100101000101110_0001000);
    assign syndrome[2] = ^(codeword&71'b1100110011110101111111001101100110011110001010010110010100010111_0000100);
    assign syndrome[1] = ^(codeword&71'b0000101001110110011001111000101001011001010001011100001011110110_0000010);
    assign syndrome[0] = ^(codeword&71'b0101010100011001001100111100110100101100101000101110000111111011_0000001);

endmodule


module SCC_4LC_errorinfo_gen(input [6:0] syndrome, output reg [6:0] error_addr, output reg [1:0] error_type);

    always @(syndrome) begin
        case (syndrome)
            // No error
            7'b00000000 : begin error_type = 2'b00; error_addr = 7'd0; end

            // SE
            7'b0001100 : begin error_type = 2'b01; error_addr = 7'd70; end
            7'b0100101 : begin error_type = 2'b01; error_addr = 7'd69; end
            7'b0110000 : begin error_type = 2'b01; error_addr = 7'd68; end
            7'b1110001 : begin error_type = 2'b01; error_addr = 7'd67; end
            7'b0001110 : begin error_type = 2'b01; error_addr = 7'd66; end
            7'b1001101 : begin error_type = 2'b01; error_addr = 7'd65; end
            7'b0101010 : begin error_type = 2'b01; error_addr = 7'd64; end
            7'b0100001 : begin error_type = 2'b01; error_addr = 7'd63; end
            7'b0110100 : begin error_type = 2'b01; error_addr = 7'd62; end
            7'b1000110 : begin error_type = 2'b01; error_addr = 7'd61; end
            7'b0101110 : begin error_type = 2'b01; error_addr = 7'd60; end
            7'b0010111 : begin error_type = 2'b01; error_addr = 7'd59; end
            7'b1111001 : begin error_type = 2'b01; error_addr = 7'd58; end
            7'b1001110 : begin error_type = 2'b01; error_addr = 7'd57; end
            7'b0111010 : begin error_type = 2'b01; error_addr = 7'd56; end
            7'b0011101 : begin error_type = 2'b01; error_addr = 7'd55; end
            7'b1111100 : begin error_type = 2'b01; error_addr = 7'd54; end
            7'b0111110 : begin error_type = 2'b01; error_addr = 7'd53; end
            7'b0011111 : begin error_type = 2'b01; error_addr = 7'd52; end
            7'b1111101 : begin error_type = 2'b01; error_addr = 7'd51; end
            7'b1001100 : begin error_type = 2'b01; error_addr = 7'd50; end
            7'b0100110 : begin error_type = 2'b01; error_addr = 7'd49; end
            7'b0010011 : begin error_type = 2'b01; error_addr = 7'd48; end
            7'b1111011 : begin error_type = 2'b01; error_addr = 7'd47; end
            7'b1001111 : begin error_type = 2'b01; error_addr = 7'd46; end
            7'b1010101 : begin error_type = 2'b01; error_addr = 7'd45; end
            7'b1011000 : begin error_type = 2'b01; error_addr = 7'd44; end
            7'b0101100 : begin error_type = 2'b01; error_addr = 7'd43; end
            7'b1000111 : begin error_type = 2'b01; error_addr = 7'd42; end
            7'b1010001 : begin error_type = 2'b01; error_addr = 7'd41; end
            7'b1011010 : begin error_type = 2'b01; error_addr = 7'd40; end
            7'b0101101 : begin error_type = 2'b01; error_addr = 7'd39; end
            7'b1100100 : begin error_type = 2'b01; error_addr = 7'd38; end
            7'b0110010 : begin error_type = 2'b01; error_addr = 7'd37; end
            7'b0011001 : begin error_type = 2'b01; error_addr = 7'd36; end
            7'b1111110 : begin error_type = 2'b01; error_addr = 7'd35; end
            7'b0111111 : begin error_type = 2'b01; error_addr = 7'd34; end
            7'b1101101 : begin error_type = 2'b01; error_addr = 7'd33; end
            7'b1000100 : begin error_type = 2'b01; error_addr = 7'd32; end
            7'b0100010 : begin error_type = 2'b01; error_addr = 7'd31; end
            7'b0010001 : begin error_type = 2'b01; error_addr = 7'd30; end
            7'b1111010 : begin error_type = 2'b01; error_addr = 7'd29; end
            7'b1110101 : begin error_type = 2'b01; error_addr = 7'd28; end
            7'b1001000 : begin error_type = 2'b01; error_addr = 7'd27; end
            7'b0100100 : begin error_type = 2'b01; error_addr = 7'd26; end
            7'b0010010 : begin error_type = 2'b01; error_addr = 7'd25; end
            7'b0001001 : begin error_type = 2'b01; error_addr = 7'd24; end
            7'b1110110 : begin error_type = 2'b01; error_addr = 7'd23; end
            7'b0111011 : begin error_type = 2'b01; error_addr = 7'd22; end
            7'b1101111 : begin error_type = 2'b01; error_addr = 7'd21; end
            7'b1000101 : begin error_type = 2'b01; error_addr = 7'd20; end
            7'b1010000 : begin error_type = 2'b01; error_addr = 7'd19; end
            7'b0101000 : begin error_type = 2'b01; error_addr = 7'd18; end
            7'b0010100 : begin error_type = 2'b01; error_addr = 7'd17; end
            7'b0001010 : begin error_type = 2'b01; error_addr = 7'd16; end
            7'b0000101 : begin error_type = 2'b01; error_addr = 7'd15; end
            7'b0000011 : begin error_type = 2'b01; error_addr = 7'd14; end
            7'b1110011 : begin error_type = 2'b01; error_addr = 7'd13; end
            7'b1001011 : begin error_type = 2'b01; error_addr = 7'd12; end
            7'b1010111 : begin error_type = 2'b01; error_addr = 7'd11; end
            7'b1011001 : begin error_type = 2'b01; error_addr = 7'd10; end
            7'b1011110 : begin error_type = 2'b01; error_addr = 7'd9; end
            7'b0101111 : begin error_type = 2'b01; error_addr = 7'd8; end
            7'b1100101 : begin error_type = 2'b01; error_addr = 7'd7; end
            7'b1000000 : begin error_type = 2'b01; error_addr = 7'd6; end
            7'b0100000 : begin error_type = 2'b01; error_addr = 7'd5; end
            7'b0010000 : begin error_type = 2'b01; error_addr = 7'd4; end
            7'b0001000 : begin error_type = 2'b01; error_addr = 7'd3; end
            7'b0000100 : begin error_type = 2'b01; error_addr = 7'd2; end
            7'b0000010 : begin error_type = 2'b01; error_addr = 7'd1; end
            7'b0000001 : begin error_type = 2'b01; error_addr = 7'd0; end

            // DE (DAE)
            7'b0101001 : begin error_type = 2'b10; error_addr = 7'd69; end
            7'b1000001 : begin error_type = 2'b10; error_addr = 7'd67; end
            7'b1000011 : begin error_type = 2'b10; error_addr = 7'd65; end
            7'b0001011 : begin error_type = 2'b10; error_addr = 7'd63; end
            7'b1110010 : begin error_type = 2'b10; error_addr = 7'd61; end
            7'b0111001 : begin error_type = 2'b10; error_addr = 7'd59; end
            7'b0110111 : begin error_type = 2'b10; error_addr = 7'd57; end
            7'b0100111 : begin error_type = 2'b10; error_addr = 7'd55; end
            7'b1000010 : begin error_type = 2'b10; error_addr = 7'd53; end
            7'b1100010 : begin error_type = 2'b10; error_addr = 7'd51; end
            7'b1101010 : begin error_type = 2'b10; error_addr = 7'd49; end
            7'b1101000 : begin error_type = 2'b10; error_addr = 7'd47; end
            7'b0011010 : begin error_type = 2'b10; error_addr = 7'd45; end
            7'b1110100 : begin error_type = 2'b10; error_addr = 7'd43; end
            7'b0010110 : begin error_type = 2'b10; error_addr = 7'd41; end
            7'b1110111 : begin error_type = 2'b10; error_addr = 7'd39; end
            7'b1010110 : begin error_type = 2'b10; error_addr = 7'd37; end
            7'b1100111 : begin error_type = 2'b10; error_addr = 7'd35; end
            7'b1010010 : begin error_type = 2'b10; error_addr = 7'd33; end
            7'b1100110 : begin error_type = 2'b10; error_addr = 7'd31; end
            7'b1101011 : begin error_type = 2'b10; error_addr = 7'd29; end
            7'b0111101 : begin error_type = 2'b10; error_addr = 7'd27; end
            7'b0110110 : begin error_type = 2'b10; error_addr = 7'd25; end
            7'b1111111 : begin error_type = 2'b10; error_addr = 7'd23; end
            7'b1010100 : begin error_type = 2'b10; error_addr = 7'd21; end
            7'b0010101 : begin error_type = 2'b10; error_addr = 7'd19; end
            7'b0111100 : begin error_type = 2'b10; error_addr = 7'd17; end
            7'b0001111 : begin error_type = 2'b10; error_addr = 7'd15; end
            7'b1110000 : begin error_type = 2'b10; error_addr = 7'd13; end
            7'b0011100 : begin error_type = 2'b10; error_addr = 7'd11; end
            7'b0000111 : begin error_type = 2'b10; error_addr = 7'd9; end
            7'b1001010 : begin error_type = 2'b10; error_addr = 7'd7; end
            7'b1100000 : begin error_type = 2'b10; error_addr = 7'd5; end
            7'b0011000 : begin error_type = 2'b10; error_addr = 7'd3; end
            7'b0000110 : begin error_type = 2'b10; error_addr = 7'd1; end

            // Uncorrectable
            default : begin error_type = 2'b11; error_addr = 7'd0; end
        endcase
    end
endmodule

module SCC_4LC_decoder(input [70:0] codeword, output [63:0] message, output [1:0] error_type);
    
    wire [6:0] syndrome;
    wire [7:0] error_addr;
    reg [70:0] decoded;

    SCC_4LC_syndrome_gen syndrome_gen(.codeword(codeword), .syndrome(syndrome));
    SCC_4LC_errorinfo_gen errorinfo_gen(.syndrome(syndrome), .error_addr(error_addr), .error_type(error_type));

    always @(error_type) begin
        if (error_type == 2'b00) begin       // No error
            decoded = codeword;
        end
        else if (error_type == 2'b01) begin  // SE
            decoded = codeword ^ (1 << error_addr);
        end
        else if (error_type == 2'b10) begin  // DE (DAE)
            decoded = codeword ^ (3 << error_addr);
        end
        else begin                          // Uncorrectable
            decoded = codeword;
        end
    end

    assign message = decoded[70:8];

endmodule