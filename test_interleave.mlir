func.func @test_interleave(%a: vector<8xi16>, %b: vector<8xi16>) -> vector<16xi16> {
  %0 = vector.interleave %a, %b : vector<8xi16>
  return %0 : vector<16xi16>
}
