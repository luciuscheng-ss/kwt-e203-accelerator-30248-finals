`timescale 1ns/1ps
// The quantizer produces native BITS-bit signed results. Every matrix
// coordinate retains a fixed field, starting at bit BITS*(256*row+column).
module kwt_fixed_matrix_writer(
 input wire clk,rst_n,
 input wire in_valid,output wire in_ready,
 input wire signed [13:0] in_even,in_odd,input wire in_frame_end,
 output reg out_valid,input wire out_ready,
 output reg [31:0] out_data,output reg [13:0] out_word_index,
 output reg out_last,output reg fault
);
 localparam integer BITS=12;
 localparam [13:0] ROW_SKIP=14'(15*(BITS/2)+1);
 reg [6:0] row;
 reg [3:0] group_id;
 reg [2:0] pair_column;
 reg [13:0] write_index;
 reg complete;
 reg [31:0] pending;
 reg [4:0] occupied;

 wire end_row_group=pair_column==7;
 wire end_group=row==98&&end_row_group;
 wire [63:0] next_bits={32'd0,pending}
                     |({{(64-2*BITS){1'b0}},in_odd[BITS-1:0],in_even[BITS-1:0]}<<occupied);
 wire [5:0] next_count={1'b0,occupied}+6'(2*BITS);
 wire new_word=next_count>=32;
 assign in_ready=(!out_valid||out_ready)&&!complete;

 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
   row<=0;group_id<=0;pair_column<=0;write_index<=0;complete<=0;
   pending<=0;occupied<=0;
   out_valid<=0;out_data<=0;out_word_index<=0;out_last<=0;fault<=0;
  end else begin
   if(out_valid&&out_ready) out_valid<=0;
   if(in_valid&&in_ready) begin
    occupied<=next_count[4:0];
    pending<=new_word?next_bits[63:32]:next_bits[31:0];
    if(new_word) begin
     out_valid<=1;
     out_data<=next_bits[31:0];
     out_word_index<=write_index;
     out_last<=end_group;
     if(end_group) write_index<=14'((group_id+1)*(BITS/2));
     else write_index<=write_index+(end_row_group?ROW_SKIP:14'd1);
    end
    if(in_frame_end!=end_group ||
       in_even[13:BITS]!={(14-BITS){in_even[BITS-1]}} ||
       in_odd[13:BITS]!={(14-BITS){in_odd[BITS-1]}} ||
       (end_row_group&&next_count!=32)) fault<=1;
    if(end_row_group) begin
     pair_column<=0;
     if(row==98) begin
      row<=0;
      if(group_id==15) complete<=1;
      else group_id<=group_id+1'b1;
     end else row<=row+1'b1;
    end else pair_column<=pair_column+1'b1;
   end
  end
 end
endmodule
