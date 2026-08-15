module {
  func.func private @llvm.x86.ssse3.pshuf.b.128(vector<16xi8>, vector<16xi8>) -> vector<16xi8>

  func.func @test_pshufb(%arg0: vector<16xi8>, %arg1: vector<16xi8>) -> vector<16xi8> {
    %0 = func.call @llvm.x86.ssse3.pshuf.b.128(%arg0, %arg1) : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
    return %0 : vector<16xi8>
  }
}
